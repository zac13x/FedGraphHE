import logging
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import tenseal as ts
from typing import List, Dict

from he_encryption_fixed import DAPHE, HEEncryptor
from models_fixed import HMAGT
from torch.nn.utils import parameters_to_vector, vector_to_parameters


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FedGraphHE")


class FRAHIP:
    def __init__(self, daphe: DAPHE, gamma=0.1, theta=0.6, delta=0.1,
                 beta=2.0, rep_max=2.0, rep_min=0.1):
          self.daphe = daphe
        self.gamma = gamma
        self.theta = theta
        self.delta = delta
        self.beta = beta
        self.rep_max = rep_max
        self.rep_min = rep_min
    
    def compute_homomorphic_inner_product(self, 
                                         encrypted_gi: List[ts.CKKSVector],
                                         encrypted_g_bar: List[ts.CKKSVector]) -> ts.CKKSVector:
         encrypted_sum = None
        
        for block_i, block_bar in zip(encrypted_gi, encrypted_g_bar):
               block_product = block_i * block_bar
            
            if encrypted_sum is None:
                encrypted_sum = block_product
            else:
                encrypted_sum += block_product
        
        return encrypted_sum
    
    def compute_consistency_score(self, si: float, norm_gi_squared: float, s0: float) -> float:
        denominator = norm_gi_squared * s0
        if denominator < 1e-10:
            return 0.0
        
        fs_i = np.sqrt(si / denominator)
        return fs_i
    
    def update_reputation(self, reputation: float, fs_i: float) -> float:
        if fs_i >= self.theta:
            return min(reputation + self.gamma, self.rep_max)
        elif fs_i < self.theta - self.delta:
            return max(reputation - self.gamma, self.rep_min)
        else:
            return reputation
    
    def compute_aggregation_weights(self, reputations: List[float]) -> np.ndarray:
        rep_array = np.array(reputations)
        exp_rep = np.exp(self.beta * rep_array)
        weights = exp_rep / np.sum(exp_rep)
        return weights
    
    def robust_aggregate(self, 
                        encrypted_gradients: List[List[ts.CKKSVector]],
                        gradient_norms_squared: List[float],
                        reputations: List[float],
                        tpa) -> tuple:
        K = len(encrypted_gradients)

        encrypted_consensus = self.daphe.aggregate_encrypted_gradients(encrypted_gradients)

        encrypted_inner_products = []
        
        for i in range(K):
            encrypted_si = self.compute_homomorphic_inner_product(
                encrypted_gradients[i],
                encrypted_consensus
            )
            encrypted_inner_products.append(encrypted_si)

        encrypted_s0 = self.compute_homomorphic_inner_product(
            encrypted_consensus,
            encrypted_consensus
        )

        scalars_to_decrypt = [encrypted_s0] + encrypted_inner_products

        decrypted_scalars = tpa.decrypt_scalars(scalars_to_decrypt)
        s0 = decrypted_scalars[0]
        inner_products = decrypted_scalars[1:]


        consistency_scores = []
        for i in range(K):
            fs_i = self.compute_consistency_score(
                inner_products[i],
                gradient_norms_squared[i],
                s0
            )
            consistency_scores.append(fs_i)
            logger.info(f"[FRAHIP] Client {i}: FSi={fs_i:.4f}, rep={reputations[i]:.4f}")

        updated_reputations = []
        for i in range(K):
            new_rep = self.update_reputation(reputations[i], consistency_scores[i])
            updated_reputations.append(new_rep)
            
            if new_rep > reputations[i]:
                logger.info(f"[FRAHIP] Client {i} Reputation rises: {reputations[i]:.4f} → {new_rep:.4f}")
            elif new_rep < reputations[i]:
                logger.warning(f"[FRAHIP] Client {i} Reputation declined: {reputations[i]:.4f} → {new_rep:.4f} (malicious)")

        weights = self.compute_aggregation_weights(updated_reputations)
        logger.info(f"[FRAHIP] Aggregate weights: {weights}")

        num_blocks = len(encrypted_gradients[0])
        aggregated_encrypted = []
        
        for block_idx in range(num_blocks):
            weighted_block = None
            for i in range(K):
                block = encrypted_gradients[i][block_idx]
                weighted = block * float(weights[i])
                
                if weighted_block is None:
                    weighted_block = weighted
                else:
                    weighted_block += weighted
            
            aggregated_encrypted.append(weighted_block)

        
        return aggregated_encrypted, updated_reputations, consistency_scores


class TrustedThirdParty:

    def __init__(self, daphe: DAPHE):
        self.daphe = daphe
        self.decryption_count = 0
    
    def decrypt_scalars(self, encrypted_scalars: List[ts.CKKSVector]) -> List[float]:
        decrypted = []
        for enc_scalar in encrypted_scalars:
            dec_array = np.array(enc_scalar.decrypt())
            scalar_value = dec_array[0]
            decrypted.append(scalar_value)
        
        self.decryption_count += len(encrypted_scalars)
        
        return decrypted


class Client:
    def __init__(self, client_id, dataloader, device, lr, num_classes=2):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.lr = lr

        self.model = HMAGT(num_classes=num_classes,
                          backbone_channels=64,
                          hidden_dim=128,
                          k_neighbors=8,
                          pool_ratio=0.5).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(self.dataloader, 
                                  desc=f"Client {self.client_id}",
                                  leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        
        return avg_loss
    
    def get_model_gradient(self, global_model):
        local_params = parameters_to_vector(self.model.parameters())
        global_params = parameters_to_vector(global_model.parameters())
        gradient = local_params - global_params
        return gradient.detach().cpu().numpy()
    
    def set_model(self, global_model):
        global_vec = parameters_to_vector(global_model.parameters())
        vector_to_parameters(global_vec, self.model.parameters())


class Server:
    
    def __init__(self, global_model, device, daphe: DAPHE, frahip: FRAHIP, tpa: TrustedThirdParty):
        self.global_model = global_model
        self.device = device
        self.daphe = daphe
        self.frahip = frahip
        self.tpa = tpa
    
    def update_global_model(self, aggregated_gradient_encrypted: List[ts.CKKSVector]):

        global_params = parameters_to_vector(self.global_model.parameters())
        original_shape = global_params.shape

        aggregated_gradient = self.daphe.decrypt_gradient(
            aggregated_gradient_encrypted,
            original_shape
        )

        grad_tensor = torch.from_numpy(aggregated_gradient).float().to(self.device)

        learning_rate = 1.0
        global_params_new = global_params - learning_rate * grad_tensor

        vector_to_parameters(global_params_new, self.global_model.parameters())





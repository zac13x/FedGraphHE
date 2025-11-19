"""
DAPHE: Dynamic Adaptive Partitioned Homomorphic Encryption
"""

import tenseal as ts
import numpy as np
from typing import Dict, List, Tuple
import torch


class DAPHE:
    def __init__(self, init_scale=2**40):
        self.scale = init_scale
        self.context = None
        self.N_star = None
        self.slots = None
        
    def compute_slot_utilization(self, gradient_dims: List[int], N: int) -> float:
        slots = N // 2
        utilizations = []
        for di in gradient_dims:
            eta_i = di / slots
            utilizations.append(min(eta_i, 1.0))
        return np.mean(utilizations)
    
    def select_optimal_ring_dimension(self, gradient_dims: List[int]) -> int:
        max_dim = max(gradient_dims)
        if max_dim <= 2867:
            eta_8192 = self.compute_slot_utilization(gradient_dims, 8192)
            if eta_8192 >= 0.7:
                return 8192
        if max_dim <= 5734:
            eta_16384 = self.compute_slot_utilization(gradient_dims, 16384)
            if eta_16384 >= 0.6:
                return 16384
        return 32768
    
    def compute_adaptive_block_size(self, di: int, slots: int) -> int:
        if di <= 1.2 * slots:
            return slots
        else:
            num_blocks = int(np.ceil(di / slots))
            return int(np.ceil(di / num_blocks))
    
    def initialize_context(self, gradient_dims: List[int]):
        self.N_star = self.select_optimal_ring_dimension(gradient_dims)
        self.slots = self.N_star // 2

        if self.N_star == 8192:
            coeff_mod_bit_sizes = [50, 30, 30, 50]
        elif self.N_star == 16384:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        else:  # 32768
            coeff_mod_bit_sizes = [60, 40, 40, 40, 60]

        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.N_star,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = self.scale
        

    
    def encrypt_gradient(self, gradient: np.ndarray, client_id: int) -> List[ts.CKKSVector]:
        if self.context is None:
            raise ValueError("initialize_context")
        
        g_flat = gradient.flatten()
        di = len(g_flat)

        block_size = self.compute_adaptive_block_size(di, self.slots)
        
        ciphertexts = []
        
        if di <= self.slots:
            ct = ts.ckks_vector(self.context, g_flat.tolist())
            ciphertexts.append(ct)
            print(f"[DAPHE] Client {client_id}: Single-block encryption, dim={di}")
        else:
            num_blocks = int(np.ceil(di / block_size))
            for j in range(num_blocks):
                start_idx = j * block_size
                end_idx = min((j + 1) * block_size, di)
                block = g_flat[start_idx:end_idx]

                if len(block) < block_size:
                    block = np.pad(block, (0, block_size - len(block)), 'constant')
                
                ct = ts.ckks_vector(self.context, block.tolist())
                ciphertexts.append(ct)
            
            print(f"[DAPHE] Client {client_id}: 分块加密, dim={di}, blocks={num_blocks}, block_size={block_size}")
        
        return ciphertexts
    
    def decrypt_gradient(self, ciphertexts: List[ts.CKKSVector], original_shape: tuple) -> np.ndarray:
        decrypted_blocks = []
        for ct in ciphertexts:
            block = np.array(ct.decrypt())
            decrypted_blocks.append(block)

        flat_gradient = np.concatenate(decrypted_blocks)

        total_elements = np.prod(original_shape)
        flat_gradient = flat_gradient[:total_elements]
        
        return flat_gradient.reshape(original_shape)
    
    def aggregate_encrypted_gradients(self, encrypted_gradients_list: List[List[ts.CKKSVector]]) -> List[ts.CKKSVector]:

        K = len(encrypted_gradients_list)
        num_blocks = len(encrypted_gradients_list[0])
        
        aggregated = []
        for block_idx in range(num_blocks):
            block_sum = encrypted_gradients_list[0][block_idx].copy()
            for client_idx in range(1, K):
                block_sum += encrypted_gradients_list[client_idx][block_idx]

            block_avg = block_sum * (1.0 / K)
            aggregated.append(block_avg)
        
        return aggregated


class HEEncryptor:
    def __init__(self, daphe: DAPHE):
        self.daphe = daphe
    
    def encrypt_model_update(self, model_update: torch.Tensor, client_id: int) -> List[ts.CKKSVector]:
        update_np = model_update.detach().cpu().numpy()
        return self.daphe.encrypt_gradient(update_np, client_id)
    
    def decrypt_model_update(self, ciphertexts: List[ts.CKKSVector], shape: tuple) -> torch.Tensor:
        update_np = self.daphe.decrypt_gradient(ciphertexts, shape)
        return torch.from_numpy(update_np).float()



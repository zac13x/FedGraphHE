import os
import sys
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from he_encryption_fixed import DAPHE, HEEncryptor
from models_fixed import HMAGT
from dataloader_fixed import create_federated_dataloaders, create_test_dataloader
from federated_learning_fixed import (
    FRAHIP, TrustedThirdParty, Client, Server
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("fedgraphhe.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FedGraphHE")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for p, l in zip(preds, labels):
                if p == 1 and l == 1:
                    TP += 1
                elif p == 1 and l == 0:
                    FP += 1
                elif p == 0 and l == 1:
                    FN += 1
                else:
                    TN += 1
    
    accuracy = correct / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


def run_federated_learning(config):
    device = torch.device(config["device"])
    client_loaders = create_federated_dataloaders(
        csv_file=config.get("train_csv"),
        img_dir=config.get("train_img_dir"),
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        alpha=config["non_iid_alpha"],
        use_real_data=config.get("use_real_data", False)
    )

    test_loader = None
    if config.get("test_csv") and config.get("test_img_dir"):
        test_loader = create_test_dataloader(
            csv_file=config["test_csv"],
            img_dir=config["test_img_dir"],
            batch_size=config["batch_size"]
        )

    global_model = HMAGT(
        num_classes=config["num_classes"],
        backbone_channels=config["backbone_channels"],
        hidden_dim=config["hidden_dim"],
        k_neighbors=config["k_neighbors"],
        pool_ratio=config["pool_ratio"]
    ).to(device)
    
    num_params = sum(p.numel() for p in global_model.parameters())


    gradient_dims = [num_params] * config["num_clients"]
    
    daphe = DAPHE(init_scale=config["ckks_scale"])
    daphe.initialize_context(gradient_dims)

    tpa = TrustedThirdParty(daphe)
    frahip = FRAHIP(
        daphe=daphe,
        gamma=config["gamma"],
        theta=config["theta"],
        delta=config["delta"],
        beta=config["beta"]
    )
    

    clients = [
        Client(
            client_id=i,
            dataloader=client_loaders[i],
            device=device,
            lr=config["client_lr"],
            num_classes=config["num_classes"]
        )
        for i in range(config["num_clients"])
    ]
    
    server = Server(
        global_model=global_model,
        device=device,
        daphe=daphe,
        frahip=frahip,
        tpa=tpa
    )

    reputations = [1.0] * config["num_clients"]

    training_history = {
        "round": [],
        "reputations": [],
        "consistency_scores": [],
        "round_time": [],
        "test_accuracy": []
    }
    
    total_start_time = time.time()
    
    for round_idx in range(config["num_rounds"]):
        round_start_time = time.time()
        
        logger.info(f"\n{'='*30} Round {round_idx + 1}/{config['num_rounds']} {'='*30}")

        for client in clients:
            client.set_model(global_model)

        encrypted_gradients = []
        gradient_norms_squared = []
        
        for i, client in enumerate(clients):
            for epoch in range(config["local_epochs"]):
                loss = client.train_one_epoch()

            gradient = client.get_model_gradient(global_model)
            norm_squared = np.sum(gradient ** 2)
            gradient_norms_squared.append(norm_squared)

            encrypted_grad = daphe.encrypt_gradient(gradient, client_id=i)
            encrypted_gradients.append(encrypted_grad)


        
        aggregated_encrypted, reputations, fs_scores = frahip.robust_aggregate(
            encrypted_gradients=encrypted_gradients,
            gradient_norms_squared=gradient_norms_squared,
            reputations=reputations,
            tpa=tpa
        )

        server.update_global_model(aggregated_encrypted)
        round_time = time.time() - round_start_time
        
        training_history["round"].append(round_idx + 1)
        training_history["reputations"].append(reputations.copy())
        training_history["consistency_scores"].append(fs_scores)
        training_history["round_time"].append(round_time)

        if test_loader is not None and (round_idx + 1) % config["eval_interval"] == 0:
            acc, prec, rec, f1 = evaluate_model(global_model, test_loader, device)
            training_history["test_accuracy"].append(acc)
            logger.info(f"    Accuracy  = {acc:.4f}")
            logger.info(f"    Precision = {prec:.4f}")
            logger.info(f"    Recall    = {rec:.4f}")
            logger.info(f"    F1 Score  = {f1:.4f}")


    total_time = time.time() - total_start_time

    
    if test_loader is not None:
        acc, prec, rec, f1 = evaluate_model(global_model, test_loader, device)
        logger.info(f"  Accuracy  = {acc:.4f}")
        logger.info(f"  Precision = {prec:.4f}")
        logger.info(f"  Recall    = {rec:.4f}")
        logger.info(f"  F1 Score  = {f1:.4f}")

    if config.get("save_model"):
        model_path = f"fedgraphhe_final_model.pth"
        torch.save(global_model.state_dict(), model_path)

    history_path = "training_history.json"
    with open(history_path, "w") as f:
        history_serializable = {
            "round": training_history["round"],
            "round_time": training_history["round_time"],
            "test_accuracy": training_history["test_accuracy"]
        }
        json.dump(history_serializable, f, indent=2)

    return global_model, training_history


from argparse import ArgumentParser
import os  
import torch
from torch.utils.data import DataLoader
from Model import StackedCAE, Predictor
from Dataset import LatentDataset
import torch.nn.functional as F
from tqdm import tqdm
from Dataset import get_args
import json

def train_predictor_for_one_epoch(dataloader, predictor, optimizer, device):
    
    predictor.train()
    
    total_loss = 0
    n = 0

    for latent in tqdm(dataloader, desc="Training"):
        
        latent = latent.to(device)
        outputs = predictor(latent)

        loss = F.mse_loss(outputs, latent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
    
    return total_loss / n

def evaluate_predictor(dataloader, predictor, device):
    
    predictor.eval()
    
    total_loss = 0
    n = 0

    with torch.no_grad():

        for latent in tqdm(dataloader, desc="Evaluating"):
            
            latent = latent.to(device)

            outputs = predictor(latent)
            loss = F.mse_loss(outputs, latent)

            total_loss += loss.item()
            n += 1
    
    return total_loss / n

def train_predictor(train_loader, eval_loader, predictor, optimizer, device, config):
    
    train_losses = []
    eval_losses = []

    best_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        
        train_predictor_for_one_epoch(train_loader, predictor, optimizer, device)
        train_loss = evaluate_predictor(train_loader, predictor, device)
        eval_loss = evaluate_predictor(eval_loader, predictor, device)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        print("Epoch %d: Train Loss: %.7f, Eval Loss: %.7f" % (epoch, train_loss, eval_loss))
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(predictor.state_dict(), config.predictor_path)

        losses = {"train": train_losses, "eval": eval_losses}
        filename = "predictor_losses" + "k=" + str(config.k) + ".json"
        path = os.path.join(config.result_data_path, filename)
        with open(path, "w") as f:
            json.dump(losses, f)

if __name__ == "__main__":
    
    config = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = StackedCAE().to(device)
    autoencoder.load_state_dict(torch.load(config.auto_encoder_path))

    predictor = Predictor(config).to(device)
    if os.path.exists(config.predictor_path):
        predictor.load_state_dict(torch.load(config.predictor_path))

    optimizer = torch.optim.Adam(predictor.parameters(), lr=config.lr)

    train_dataset = LatentDataset(config.root_path, config.train_split, autoencoder, device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    eval_dataset = LatentDataset(config.root_path, config.eval_split, autoencoder, device)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    train_predictor(train_loader, eval_loader, predictor, optimizer, device, config)
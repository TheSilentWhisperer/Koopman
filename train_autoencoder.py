import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from Dataset import VideoDataset
from torch.utils.data import DataLoader
from Model import StackedCAE
from Dataset import get_args
import json

def train_layer_for_one_epoch(model, optimizer, device, config, split):
    
    model.train()
    root_path = config.root_path
    nb_buckets = len(os.listdir(os.path.join(root_path, split)))

    total_loss = 0
    n = 0

    for bucket in tqdm(range(nb_buckets), desc="Training"):
        dataset = VideoDataset(root_path, split, bucket, is_image_dataset=True)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        for images in dataloader:

            images = images.to(device)

            model.eval()
            with torch.no_grad():
                images = model.precompute(images, config.layer)
            model.train()

            outputs = model(images, config.layer)
            loss = F.mse_loss(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1
    
    return total_loss / n

def evaluate_layer(model, device, config, split):
    
    model.eval()
    root_path = config.root_path
    nb_buckets = len(os.listdir(os.path.join(root_path, split)))

    total_loss = 0
    n = 0

    for bucket in tqdm(range(nb_buckets), desc="Evaluating"):
        dataset = VideoDataset(root_path, split, bucket, is_image_dataset=True)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        for images in dataloader:

            images = images.to(device)

            with torch.no_grad():
                images = model.precompute(images, config.layer)

            outputs = model(images, config.layer)
            loss = F.mse_loss(outputs, images)

            total_loss += loss.item()
            n += 1
    
    return total_loss / n

def train_autoencoder(model, optimizer, device, config):

    best_loss = float("inf")
    epochs = config.epochs

    train_losses = []
    eval_losses = []

    for epoch in range(1, epochs+1):
        
        train_layer_for_one_epoch(model, optimizer, device, config, config.train_split)
        train_loss = evaluate_layer(model, device, config, config.train_split)
        eval_loss = evaluate_layer(model, device, config, config.eval_split)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), config.auto_encoder_path)
        
        print("Epoch %d, Train Loss: %.7f, Eval Loss: %.7f" % (epoch, train_loss, eval_loss))

        #dump losses in a json file
        losses = dict()
        losses['train'] = train_losses
        losses['eval'] = eval_losses
        filename = "layer" + str(config.layer) + "_losses.json" if config.layer >= 0 else "all_layers_losses.json"
        path = os.path.join(config.result_data_path, filename)
        with open(path, 'w') as f:
            json.dump(losses, f)
    


if __name__ == "__main__":

    config = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    model = StackedCAE().to(device)

    if os.path.exists(config.auto_encoder_path):
        model.load_state_dict(torch.load(config.auto_encoder_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    train_autoencoder(model, optimizer, device, config)
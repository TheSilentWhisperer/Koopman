import argparse
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser("Train a stacked convolutional autoencoder")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to train, -1 for all layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of videos/images in a batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--root_path", type=str, default="Pendulum_data_10000", help="Path to the root directory of the dataset")
    parser.add_argument("--train_split", type=str, default="train", help="Split to train on")
    parser.add_argument("--eval_split", type=str, default="eval", help="Split to validate on")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save the model")

    args = parser.parse_args()
    return args

def train_layer_for_one_epoch(model, optimizer, device, config):
    
    model.train()
    root_path = config.root_path
    split = config.train_split
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

def evaluate_layer(model, device, config):
    
    model.eval()
    root_path = config.root_path
    split = config.eval_split
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
    
    for epoch in range(epochs):
        
        train_loss = train_layer_for_one_epoch(model, optimizer, device, config)
        eval_loss = evaluate_layer(model, device, config)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), config.model_path)
        
        print("Epoch %d, Train Loss: %.7f, Eval Loss: %.7f" % (epoch, train_loss, eval_loss))
    


if __name__ == "__main__":

    config = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    from Dataset import VideoDataset
    from torch.utils.data import DataLoader
    from Model import StackedCAE

    model = StackedCAE().to(device)

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    train_autoencoder(model, optimizer, device, config)
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from Dataset import VideoDataset
from torch.utils.data import DataLoader
from Model import FullModel
from Dataset import get_args
import json

def train_for_one_epoch(model, optimizer, device, config):

    model.train()
    root_path = config.root_path
    split = config.train_split
    nb_buckets = len(os.listdir(os.path.join(root_path, split)))

    total_loss = 0
    n = 0
    for bucket in tqdm(range(nb_buckets), desc="Training"):
        dataset = VideoDataset(root_path, split, bucket)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1
    
    optimizer.zero_grad(set_to_none=True)
    return total_loss / n

def evaluate(model, device, config, split):
    
    model.eval()
    root_path = config.root_path
    nb_buckets = len(os.listdir(os.path.join(root_path, split)))

    total_loss = 0
    n = 0

    with torch.no_grad():

        for bucket in tqdm(range(nb_buckets), desc="Evaluating"):
            dataset = VideoDataset(root_path, split, bucket)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            
            for images in dataloader:

                images = images.to(device)

                outputs = model(images)
                loss = F.mse_loss(outputs, images)

                total_loss += loss.item()
                n += 1
    
    return total_loss / n

def train(model, optimizer, device, config):

    best_loss = float("inf")
    epochs = config.epochs
    
    train_losses = []
    eval_losses = []

    for epoch in range(1, epochs+1):
        train_for_one_epoch(model, optimizer, device, config)
        train_loss = evaluate(model, device, config, config.train_split)
        eval_loss = evaluate(model, device, config, config.eval_split)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.auto_encoder.state_dict(), config.auto_encoder_path)
            torch.save(model.predictor.state_dict(), config.predictor_path)
        
        print("Epoch %d, Train Loss: %.7f, Eval Loss: %.7f" % (epoch, train_loss, eval_loss))
    
        #dump losses in a json file
        losses = dict()
        losses['train'] = train_losses
        losses['eval'] = eval_losses
        filename = "full_model_losses" + "k=" + str(config.k) + ".json"
        path = os.path.join(config.result_data_path, filename)
        with open(path, 'w') as f:
            json.dump(losses, f)


if __name__ == "__main__":

    config = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    model = FullModel(config).to(device)

    if os.path.exists(config.auto_encoder_path):
        model.auto_encoder.load_state_dict(torch.load(config.auto_encoder_path))
    
    if os.path.exists(config.predictor_path):
        model.predictor.load_state_dict(torch.load(config.predictor_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train(model, optimizer, device, config)
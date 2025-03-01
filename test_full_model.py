from Model import FullModel
from Model import get_args
import torch
import torch.nn.functional as F
from Dataset import VideoDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json

if __name__ == '__main__':

    config = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = 0
    losses = torch.zeros(300).to(device)

    model = FullModel(config).to(device)
    model.auto_encoder.load_state_dict(torch.load(config.auto_encoder_path))
    model.predictor.load_state_dict(torch.load(config.predictor_path))

    model.eval()
    root_path = config.root_path
    nb_buckets = len(os.listdir(os.path.join(root_path, "test")))

    with torch.no_grad():

        for bucket in tqdm(range(nb_buckets), desc="Evaluating"):
            dataset = VideoDataset(root_path, "test", bucket)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            
            for images in dataloader:

                images = images.to(device)

                outputs = model(images)
                loss = F.mse_loss(outputs, images, reduction="none").mean(dim=(2,3,4))
                losses += torch.sqrt(loss).sum(dim=0)
                n += loss.size(0)

    losses = (losses / n).cpu().numpy().tolist()

    print(losses)

    filename = "full_model_losses.json"
    path = config.result_data_path + "/" + filename
    with open(path, "w") as f:
        json.dump(losses, f)
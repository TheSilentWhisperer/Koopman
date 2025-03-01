import torch
import os
from torchvision.io import read_video
import numpy as np
from tqdm import tqdm
from Model import get_args
import json

def create_buckets(root_path, split, nb_frames_per_dataset = 10000, with_states = False):
    
    bucket_root_path = root_path + "_" + str(nb_frames_per_dataset)
    if not os.path.exists(bucket_root_path):
        os.mkdir(bucket_root_path)
    
    bucket_split_path = os.path.join(bucket_root_path, split)
    if not os.path.exists(bucket_split_path):
        os.mkdir(bucket_split_path)

    current_dataset = []
    current_states = []

    k = 0

    split_path = os.path.join(root_path, split)
    video_names = os.listdir(split_path)
    video_names = [video_name for video_name in video_names if video_name.endswith(".mp4")]
    np.random.shuffle(video_names)

    for video_name in tqdm(video_names):
        video_path = os.path.join(split_path, video_name)
        video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
        video = video[:, 0:1, :, :]

        if with_states:
            states_path = os.path.join(split_path, "q_" + video_name[:-4] + ".json")
            with open(states_path, "r") as f:
                states = json.load(f)
                states_list = [states[str(i)] for i in range(len(video))]
                current_states.append(states_list)

        if (len(current_dataset) + 1) * video.shape[0] > nb_frames_per_dataset:
            filename_path = os.path.join(bucket_split_path, str(k))
            current_dataset = torch.stack(current_dataset)
            torch.save(current_dataset, filename_path + ".pt")
            current_dataset = []
            if with_states:
                with open(filename_path + "_states.json", "w") as f:
                    json.dump(current_states, f)
                current_states = []
            k += 1
        current_dataset.append(video)

    filename_path = os.path.join(bucket_split_path, str(k))
    current_dataset = torch.stack(current_dataset)
    torch.save(current_dataset, filename_path + ".pt")

class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, root_path, split, bucket, is_image_dataset = False):
        
        self.root_path = root_path
        self.split_path = os.path.join(root_path, split)
        self.bucket_path = os.path.join(self.split_path, str(bucket) + ".pt")
        
        self.data = torch.load(self.bucket_path).float()

        if is_image_dataset:
            self.data = self.data.view(-1, *self.data.shape[2:])


    def __getitem__(self, index):
        sample = self.data[index]
        sample /= 255
        sample -= 0.5
        return sample
    
    def __len__(self):
        return len(self.data)

class LatentDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_path, split, autoencoder, device):
        
        self.root_path = root_path
        self.split_path = os.path.join(root_path, split)
        self.autoencoder = autoencoder
        self.device = device
        self.data = []

        nb_buckets = len(os.listdir(self.split_path))
        
        for bucket in range(nb_buckets):

            print("Precomputing bucket %d" % bucket)
            
            dataset = VideoDataset(root_path, split, bucket)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

            for video in dataloader:
                x = video.to(self.device).flatten(0, 1)
                autoencoder.eval()
                with torch.no_grad():
                    latent = autoencoder.encoder(x)
                latent = latent.view(*video.shape[:2], -1)   
                self.data.append(latent)
    
        self.data = torch.cat(self.data, dim=0)
                
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    config = get_args()

    root_path = config.original_root_path
    split = "test"
    create_buckets(root_path, split, 10000)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # autoencoder = StackedCAE().to(device)
    # autoencoder.load_state_dict(torch.load(config.auto_encoder_path))

    # latent_dataset = LatentDataset(root_path, split, autoencoder, device)

    
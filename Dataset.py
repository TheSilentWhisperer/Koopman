import torch
import os
from torchvision.io import read_video
from torchvision import transforms
import numpy as np

def create_buckets(root_path, split, nb_frames_per_dataset = 10000):
    
    bucket_root_path = root_path + "_" + str(nb_frames_per_dataset)
    if not os.path.exists(bucket_root_path):
        os.mkdir(bucket_root_path)
    
    bucket_split_path = os.path.join(bucket_root_path, split)
    if not os.path.exists(bucket_split_path):
        os.mkdir(bucket_split_path)

    current_dataset = []

    k = 0

    split_path = os.path.join(root_path, split)
    video_names = os.listdir(split_path)
    video_names = [video_name for video_name in video_names if video_name.endswith(".mp4")]
    np.random.shuffle(video_names)

    for video_name in video_names:
        video_path = os.path.join(split_path, video_name)
        video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
        video = video[:, 0:1, :, :]

        if (len(current_dataset) + 1) * video.shape[0] > nb_frames_per_dataset:
            filename_path = os.path.join(bucket_split_path, str(k))
            current_dataset = torch.stack(current_dataset)
            torch.save(current_dataset, filename_path + ".pt")
            current_dataset = []
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

if __name__ == "__main__":

    root_path = "Pendulum_data_10000"
    split = "train"
    bucket = 0

    dataset = VideoDataset(root_path, split, bucket)

    print(len(dataset))
    
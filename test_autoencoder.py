import os
import numpy as np
from Dataset import VideoDataset
from Model import StackedCAE
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def visualize_model(model, video, layer, device):
    
    model.eval()
    with torch.no_grad():
        video = video.to(device).flatten(0, 1)
        video = model.precompute(video, layer)
        reconstructed_video = model(video, layer)

    video = video[:, 0:1, :, :]
    reconstructed_video = reconstructed_video[:, 0:1, :, :]

    print(video.shape)
    print(reconstructed_video.shape)

    video = video.cpu().expand(-1, 3, -1, -1).permute(0, 2, 3, 1).clone()
    reconstructed_video = reconstructed_video.cpu().expand(-1, 3, -1, -1).permute(0, 2, 3, 1).clone()

    video += 0.5
    video *= 255

    reconstructed_video += 0.5
    reconstructed_video *= 255

    fig, ax = plt.subplots(1, 2)
    ax[0].axis("off")
    ax[1].axis("off")

    ax[0].set_title("Original Video")
    ax[1].set_title("Reconstructed Video")

    frames_video = []
    frames_reconstructed_video = []
    for i in range(video.shape[0]):
        frame_video = ax[0].imshow(video[i].numpy().astype(np.uint8))
        frame_reconstructed_video = ax[1].imshow(reconstructed_video[i].numpy().astype(np.uint8))
        frames_video.append([frame_video])
        frames_reconstructed_video.append([frame_reconstructed_video])

    ani_video = animation.ArtistAnimation(fig, frames_video, interval=1000/30, blit=True, repeat_delay=0)
    ani_reconstructed_video = animation.ArtistAnimation(fig, frames_reconstructed_video, interval=1000/30, blit=True, repeat_delay=0)
    
    plt.show()

    

if __name__ == "__main__":
    
    root_path = "Pendulum_data_10000"
    split = "test"
    layer = -1

    nb_buckets = len(os.listdir(os.path.join(root_path, split)))
    bucket = np.random.randint(nb_buckets)

    dataset = VideoDataset(root_path, split, bucket)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    video = next(iter(dataloader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = StackedCAE().to(device)
    model.load_state_dict(torch.load("model.pth"))

    visualize_model(model, video, layer, device)

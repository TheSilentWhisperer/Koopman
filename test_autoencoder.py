import os
import numpy as np
from Dataset import VideoDataset
from Model import StackedCAE, Predictor
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio as iio
from Dataset import get_args

def get_reconstructed_video(autoencoder, video):
    autoencoder.eval()

    with torch.no_grad():
        latent_video = autoencoder.encoder(video)
        reconstructed_video = autoencoder.decoder(latent_video)

    return reconstructed_video

def unnormalize(video):
    video = video.cpu().expand(-1, 3, -1, -1).permute(0, 2, 3, 1).clone()
    video += 0.5
    video *= 255
    return video

def visualize_model(autoencoder, video):

    reconstructed_video = get_reconstructed_video(autoencoder, video)

    video = unnormalize(video)
    reconstructed_video = unnormalize(reconstructed_video)

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

def save_gif(autoencoder, video):
    
    reconstructed_video = get_reconstructed_video(autoencoder, video)
    video = unnormalize(video)
    reconstructed_video = unnormalize(reconstructed_video)
    iio.mimwrite("original_video.gif", video.numpy().astype(np.uint8), fps=20, subrectangles=True)
    iio.mimwrite("reconstructed_video.gif", reconstructed_video.numpy().astype(np.uint8), fps=20, subrectangles=True)
    iio.help("gif")


if __name__ == "__main__":
    
    config = get_args()

    root_path = config.root_path
    split = "test"

    nb_buckets = len(os.listdir(os.path.join(root_path, split)))
    bucket = np.random.randint(nb_buckets)

    dataset = VideoDataset(root_path, split, bucket)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

video = next(iter(dataloader))[0].to(device)

autoencoder = StackedCAE().to(device)
autoencoder.load_state_dict(torch.load(config.auto_encoder_path))

visualize_model(autoencoder, video)
save_gif(autoencoder, video)

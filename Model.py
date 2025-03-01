import torch
from torch import nn
from torch.nn import functional as F
import argparse

def get_args():
    parser = argparse.ArgumentParser("Train a stacked convolutional autoencoder")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to train, -1 for all layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of videos/images in a batch")
    parser.add_argument("--lr", type=float, default=0.0001 , help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--root_path", type=str, default="cartpole_64_64_bw_10000", help="Path to the root directory of the dataset")
    parser.add_argument("--original_root_path", type=str, default="cartpole_64_64_bw", help="Path to the root directory of the original dataset")
    parser.add_argument("--train_split", type=str, default="train", help="Split to train on")
    parser.add_argument("--eval_split", type=str, default="eval", help="Split to validate on")
    parser.add_argument("--test_split", type=str, default="test", help="Split to test on")
    parser.add_argument("--auto_encoder_path", type=str, default="autoencoder.pth", help="Path to save the autoencoder model")
    parser.add_argument("--predictor_path", type=str, default="predictor.pth", help="Path to save the predictor model")
    parser.add_argument("--h", type=int, default=2, help="Number of time steps to use for prediction")
    parser.add_argument("--k", type=int, default=1, help="Number of time steps before resetting the prediction")
    parser.add_argument("--m", type=int, default=50, help="Number of points to use to estimate A")
    parser.add_argument("--A_method", type=str, default="Learned", choices=["Learned", "SVD", "Proximal"], help="Method to estimate A")
    parser.add_argument("--max_proximal_iter", type=int, default=100, help="Maximum number of iterations for the proximal method")
    parser.add_argument("--rho", type=float, default=1e-1, help="Regularization parameter for the proximal method")
    parser.add_argument("--result_data_path", type=str, default="results/data", help="Path to save the results data")

    args = parser.parse_args()
    return args

# class VideoPredictor(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.m = config["m"]
#         self.h = config["h"]
#         self.A_method = config["A_method"]
#         self.max_proximal_iter = config["max_proximal_iter"]
#         self.rho = config["rho"]

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1), # 1x64x64 -> 16x64x64
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(2), # 16x64x64 -> 16x32x32
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, padding=1), # 16x32x32 -> 32x32x32
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2), # 32x32x32 -> 32x16x16
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1), # 32x16x16 -> 64x16x16
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2), # 64x16x16 -> 64x8x8
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), # 64x8x8 -> 32x8x8
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2), # 32x8x8 -> 32x4x4
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1), # 32x4x4 -> 16x4x4
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2), # 16x4x4 -> 16x2x2
#             nn.ReLU(),
#             nn.Conv2d(32, 8, 3, padding=1), # 16x2x2 -> 8x2x2
#             nn.BatchNorm2d(8),
#             nn.MaxPool2d(2), # 8x2x2 -> 8x1x1
#             nn.Flatten()
#         )
            
#         # self.decoder = nn.Sequential(
#         #     nn.Unflatten(1, (16, 1, 1)),
#         #     nn.Upsample(2), # 16x1x1 -> 16x2x2
#         #     nn.ReLU(),
#         #     nn.ConvTranspose2d(16, 32, 3), # 16x2x2 -> 32x4x4
#         #     nn.ReLU(),
#         #     ResidualBlock(nn.ConvTranspose2d(32, 64, 3, padding=1), nn.ConvTranspose2d(64, 32, 3, padding=1), transpose=True), # 32x4x4 -> 32x16x16
#         #     nn.ReLU(),
#         #     ResidualBlock(nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ConvTranspose2d(16, 1, 3, padding=1), transpose=True), # 32x16x16 -> 1x64x64
#         # )
#         self.decoder = nn.Sequential(
#             nn.Unflatten(1, (8, 1, 1)),
#             nn.ConvTranspose2d(8, 32, 2, stride=2), # 8x1x1 -> 16x2x2
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 64, 2, stride=2), # 16x2x2 -> 32x4x4
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 64, 2, stride=2), # 32x4x4 -> 64x8x8
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 2, stride=2), # 64x8x8 -> 32x16x16
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 2, stride=2), # 32x16x16 -> 16x32x32
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, 2, stride=2), # 16x32x32 -> 1x64x64
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#     def encode(self, videos):
#         #videos of shape (batch_size, num_frames, num_channels, height, width)
#         #merge batch_size and num_frames dimensions to use 2D convolutions efficiently
#         batch_size, num_frames = videos.shape[:2]
#         videos = videos.flatten(0, 1)
#         #returns encoded videos of shape (batch_size, num_frames, num_channels, height, width)
#         return self.encoder(videos).unflatten(0, (batch_size, num_frames))
    
#     def get_data_to_fit(self, encoded_videos, t0=0):
#         Z = []
#         for i in range(self.h):
#             Z_i = encoded_videos[:,t0 + i:t0 + i + self.m - self.h, :]
#             Z.append(Z_i)
#         Z = torch.cat(Z, dim=2)
#         Zh = encoded_videos[:,t0 + self.h:t0 + self.m, :]   
#         return Z, Zh

#     def get_A_with_SVD(self, Z, Zh): 
#         U, S, V_t = torch.linalg.svd(Z, full_matrices=False)
#         U_t, S_inv, V = U.transpose(-2, -1), torch.diag_embed(1/S), V_t.transpose(-2, -1)
#         S_inv[S_inv > 1] = 0
#         A = V.bmm(S_inv.bmm(U_t.bmm(Zh)))
#         return A

#     def get_A_with_proximal(self, Z, Zh):
#         device = Z.device
#         bs = Z.shape[0]
#         n = Zh.shape[-1]
#         h = Z.shape[-1] // n

#         A_0 = torch.zeros(bs, n * h, n).to(device)
#         for i in range(h):
#             A_0[:, i * n:(i+1) * n, :] = torch.eye(n).unsqueeze(0).repeat(bs, 1, 1)
#         A_list = [A_0]
#         for i in range(self.max_proximal_iter):
#             L = torch.linalg.cholesky(Z.transpose(1, 2).bmm(Z) + self.rho * torch.eye(n * h).to(device))
#             B = Z.transpose(1, 2).bmm(Zh - Z.bmm(A_list[-1]))
#             A_list.append(torch.cholesky_solve(B, L))

#         best_loss, best_A = float("inf"), None
#         for i, A in enumerate(A_list):
#             loss = F.mse_loss(Z.bmm(A), Zh)
#             if loss < best_loss:
#                 best_loss = loss
#                 best_A = A

#         return best_A

#     def get_A(self, encoded_videos, t0 = 0):
#         Z, Zh = self.get_data_to_fit(encoded_videos, t0)
#         A = self.get_A_with_proximal(Z, Zh)
#         if self.A_method == "SVD":
#             return self.get_A_with_SVD(Z, Zh)
#         if self.A_method == "Proximal":
#             return self.get_A_with_proximal(Z, Zh)
        

#     def forward(self, videos, train_prediction = True):
#         batch_size = videos.shape[0]
#         seq_len = videos.shape[1]
#         encoded_videos = self.encode(videos)
#         self.A = self.get_A(encoded_videos) if train_prediction else None
#         predictions = []

#         for i in range(self.m):
#             predictions.append(encoded_videos[:, i:i+1, :])
        
#         if train_prediction:
#             for i in range(self.m, seq_len):
#                 Z = torch.cat(predictions[-self.h:], dim=2)
#                 predictions.append(Z.bmm(self.A).view(batch_size, 1, -1))
#         else:
#             for i in range(self.m, seq_len):
#                 predictions.append(encoded_videos[:, i:i+1, :])

#         predictions = torch.cat(predictions, dim=1)

#         decoded_videos = self.decoder(predictions.flatten(0, 1)).unflatten(0, (batch_size, seq_len))
#         return decoded_videos


class StackedCAE(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), # 1x64x64 -> 16x64x64
                nn.ReLU(),
                nn.MaxPool2d(2), # 16x64x64 -> 16x32x32
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1), # 16x32x32 -> 32x32x32
                nn.ReLU(),
                nn.MaxPool2d(2), # 32x32x32 -> 32x16x16
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), # 32x16x16 -> 64x16x16
                nn.ReLU(),
                nn.MaxPool2d(2), # 64x16x16 -> 64x8x8
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), # 64x8x8 -> 32x8x8
                nn.ReLU(),
                nn.MaxPool2d(2), # 32x8x8 -> 32x4x4
            ),
            nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1), # 32x4x4 -> 16x4x4
                nn.ReLU(),
                nn.MaxPool2d(2), # 16x4x4 -> 16x2x2
            ),
            nn.Sequential(
                nn.Conv2d(16, 8, 3, padding=1), # 16x2x2 -> 8x2x2
                nn.ReLU(),
                nn.MaxPool2d(2), # 8x2x2 -> 8x1x1
            )
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 8x1x1 -> 8x2x2
                nn.ConvTranspose2d(8, 16, 3, padding=1), # 8x2x2 -> 16x2x2
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 16x2x2 -> 16x4x4
                nn.ConvTranspose2d(16, 32, 3, padding=1), # 16x4x4 -> 32x4x4
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 32x4x4 -> 32x8x8
                nn.ConvTranspose2d(32, 64, 3, padding=1), # 32x8x8 -> 64x8x8
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 64x8x8 -> 64x16x16
                nn.ConvTranspose2d(64, 32, 3, padding=1), # 64x16x16 -> 32x16x16
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 32x16x16 -> 32x32x32
                nn.ConvTranspose2d(32, 16, 3, padding=1), # 32x32x32 -> 16x32x32
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2), # 16x32x32 -> 16x64x64
                nn.ConvTranspose2d(16, 1, 3, padding=1), # 16x64x64 -> 1x64x64
            )
        )

    def forward(self, x, layer):
        
        if layer == -1:
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
        nb_layers = len(self.encoder)
        x = self.encoder[layer](x)
        x = self.decoder[nb_layers - layer - 1](x)
        return x
    
    def precompute(self, x, layer):
        if layer == -1:
            return x
        for i in range(layer):
            x = self.encoder[i](x)
        return x

class Predictor(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.h = config.h
        self.k = config.k
        self.m = config.m
        self.A_method = config.A_method
        self.max_proximal_iter = config.max_proximal_iter
        self.rho = config.rho
        
        self.encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.A = nn.Linear(8 * self.h, 8, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
    def get_A_with_SVD(self, Z, Zh):
        U, S, V_t = torch.linalg.svd(Z, full_matrices=False)
        U_t, S_inv, V = U.transpose(-2, -1), torch.diag_embed(1/S), V_t.transpose(-2, -1)
        A = V.bmm(S_inv.bmm(U_t.bmm(Zh)))
        return A
    
    def get_A_with_proximal(self, Z, Zh):
        device = Z.device
        bs = Z.shape[0]
        n = Zh.shape[-1]
        h = Z.shape[-1] // n

        A_0 = torch.zeros(bs, n * h, n).to(device)
        for i in range(h):
            A_0[:, i * n:(i+1) * n, :] = torch.eye(n).unsqueeze(0).repeat(bs, 1, 1)
        A_list = [A_0]
        for i in range(self.max_proximal_iter):
            L = torch.linalg.cholesky(Z.transpose(1, 2).bmm(Z) + self.rho * torch.eye(n * h).to(device))
            B = Z.transpose(1, 2).bmm(Zh - Z.bmm(A_list[-1]))
            A_list.append(torch.cholesky_solve(B, L))

        best_loss, best_A = float("inf"), None
        for i, A in enumerate(A_list):
            loss = F.mse_loss(Z.bmm(A), Zh)
            if loss < best_loss:
                best_loss = loss
                best_A = A

        return best_A

    def get_data_to_fit(self, x):
        Z = []
        for i in range(self.h):
            Z_i = x[:, i:i + self.m - self.h, :]
            Z.append(Z_i)
        Z = torch.cat(Z, dim=2)
        Zh = x[:, self.h:self.m, :]   
        return Z, Zh

    def forward(self, x):

        device = x.device

        x = self.encoder(x)
        batch_size, seq_len, n = x.shape

        #find A, A^2, ..., A^k
        match self.A_method:
            case "Learned":
                A = self.A.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            case "SVD":
                Z, Zh = self.get_data_to_fit(x)
                A = self.get_A_with_SVD(Z, Zh).permute(0, 2, 1)
            case "Proximal":
                Z, Zh = self.get_data_to_fit(x)
                A = self.get_A_with_proximal(Z, Zh).permute(0, 2, 1)

        A_aug_list = [None] * (self.k + 1)

        A_aug = torch.zeros(batch_size, n * self.h, n * self.h).to(device)

        A_aug[:, -n:, :] = A

        batch_eye = torch.eye(n * self.h).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        A_aug[:, :-n, :] = batch_eye[:, n:, :]

        A_aug_list[1] = A_aug
        for i in range(2, self.k+1):
            A_aug_list[i] = A_aug_list[i - 1].bmm(A_aug)

        #find predictions
        predictions = [None] * seq_len

        for i in range(self.m):
            predictions[i] = x[:, i, :]

        for i in range(self.m, seq_len):
            l = min(self.k, i - self.m + 1)
            Z = x[:, i-self.h:i, :].flatten(1, 2).unsqueeze(-1)
            A_l = A_aug_list[l]
            predictions[i] = A_l.bmm(Z)[:, -n:, 0]
        
        predictions = torch.stack(predictions, dim=1)
        
        output = self.decoder(predictions)
        return output

class FullModel(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.auto_encoder = StackedCAE()
        self.predictor = Predictor(config)

    def forward(self, videos):
        
        x = self.auto_encoder.encoder(videos.flatten(0, 1))
        latent = x.view(*videos.shape[:2], -1)  
        prediction = self.predictor(latent)
        prediction = prediction.view(*x.shape)
        x = self.auto_encoder.decoder(prediction).view(*videos.shape)

        return x

if __name__ == "__main__":
    
    import torch
    from Dataset import VideoDataset
    from torch.utils.data import DataLoader

    config = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    model = FullModel(config).to(device)
    print(model)

    root_path = "Pendulum_data_10000"
    split = "train"
    bucket = 0

    dataset = VideoDataset(root_path, split, bucket, is_image_dataset=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    images = next(iter(dataloader)).float().to(device)
    print(model(images).shape)
    
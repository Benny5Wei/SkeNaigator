import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):

    def __init__(self, goal_id, hidden_size, hiddens=[16, 32, 64, 128, 256] , latent_dim=128) -> None:
        super().__init__()
        self.goal_id = goal_id
        self. hidden_size = hidden_size
        prev_channels = 3
        modules = []
        img_length = 64
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.projector = nn.Linear(latent_dim, self.hidden_size)
        nn.init.xavier_normal_(self.projector.weight)

    def forward(self, observations):
        
        goal_observations = observations[self.goal_id]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        goal_observations = goal_observations.permute(0, 3, 1, 2)
        goal_observations = goal_observations / 255.0  # normalize map
        goal_observations = F.interpolate(goal_observations, size=(64, 64), mode='bilinear', align_corners=False)
        encoded = self.encoder(goal_observations)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        return self.projector(z)
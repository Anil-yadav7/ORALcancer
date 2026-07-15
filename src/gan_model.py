import torch
import torch.nn as nn

class Generator(nn.Module):
    """ 
    13-Layer Conditional Generator based on the clinical dcGAN baseline.
    Output: Synthetic 256x256 H&E Patch
    """
    def __init__(self, noise_dim=128, num_classes=5, features_g=64):
        super(Generator, self).__init__()        
        self.features_g = features_g
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.init_proj = nn.Sequential(
            # bias=False because it is followed by BatchNorm
            nn.Linear(noise_dim + num_classes, features_g * 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(features_g * 16 * 8 * 8),
            nn.ReLU(True)
        )
        
        self.up_blocks = nn.Sequential(
            # Block 1 (8x8 -> 16x16)
            nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # Block 2 (16x16 -> 32x32)
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # Block 3 (32x32 -> 64x64)
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # Block 4 (64x64 -> 128x128)
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # Block 5 (128x128 -> 256x256)
            # bias=True here because there is no BatchNorm after it
            nn.ConvTranspose2d(features_g, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh() 
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        x = self.init_proj(x)
        x = x.view(-1, self.features_g * 16, 8, 8)
        return self.up_blocks(x)


class Critic(nn.Module):
    """
    12-Layer Conditional Discriminator (Critic for WGAN-GP).
    Optimized with Spatial Broadcasting and dynamic GroupNorm.
    """
    def __init__(self, features_d=64, num_classes=5):
        super(Critic, self).__init__()
        
        # Memory Optimized: Embed to a single channel rather than 256*256 pixels
        self.label_emb = nn.Embedding(num_classes, 1)
        
        self.model = nn.Sequential(
            # Input: 4 channels (3 RGB + 1 Label Map)
            nn.Conv2d(4, features_d, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # GroupNorm(1, C) is mathematically identical to LayerNorm but adapts to any spatial size
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, features_d * 2), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 8, features_d * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, features_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(features_d * 16 * 8 * 8, 1, bias=True)
        )

    def forward(self, img, labels):
        # Broadcast the 1D embedding to match the image's spatial dimensions (H, W)
        c = self.label_emb(labels).view(-1, 1, 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        
        x = torch.cat([img, c], dim=1)
        return self.model(x)
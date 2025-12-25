"""
CycleGAN implementation for image synthesis
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """Generator network for CycleGAN"""
    
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, 
                                  padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

class CycleGAN(nn.Module):
    """Complete CycleGAN model"""
    
    def __init__(self):
        super(CycleGAN, self).__init__()
        
        # Generators
        self.G_s2t = Generator()  # Source to Target (T1 to T2)
        self.G_t2s = Generator()  # Target to Source (T2 to T1)
        
        # Discriminators
        self.D_s = Discriminator()  # Discriminator for source domain
        self.D_t = Discriminator()  # Discriminator for target domain
    
    def forward(self, source, target):
        """
        Forward pass for training
        Returns generated images
        """
        # Generate fake images
        fake_target = self.G_s2t(source)  # T1 -> T2
        fake_source = self.G_t2s(target)  # T2 -> T1
        
        # Reconstruct images (cycle consistency)
        reconstructed_source = self.G_t2s(fake_target)  # T1 -> T2 -> T1
        reconstructed_target = self.G_s2t(fake_source)  # T2 -> T1 -> T2
        
        return {
            'fake_target': fake_target,
            'fake_source': fake_source,
            'reconstructed_source': reconstructed_source,
            'reconstructed_target': reconstructed_target
        }
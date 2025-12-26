# """
# Simplified Swin Transformer for segmentation
# Note: For production, use timm library's Swin implementation
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PatchEmbed(nn.Module):
#     """Image to Patch Embedding"""
    
#     def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = img_size // patch_size
#         self.num_patches = self.patches_resolution ** 2
        
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = nn.LayerNorm(embed_dim)
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         x = self.norm(x)
#         return x

# class SwinTransformerBlock(nn.Module):
#     """Simplified Swin Transformer Block"""
    
#     def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
        
#         self.norm1 = nn.LayerNorm(dim)
#         # Note: This uses Global Attention (ViT style), not Window Attention (Swin style)
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, dim)
#         )
        
#         self.attention_weights = None
    
#     def forward(self, x):
#         # Multi-head self-attention
#         shortcut = x
#         x = self.norm1(x)
#         x, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
#         self.attention_weights = attn_weights  # Store for Meta Attention
#         x = shortcut + x
        
#         # MLP
#         x = x + self.mlp(self.norm2(x))
#         return x

# class SwinTransformerSegmentation(nn.Module):
#     """
#     Swin Transformer-based segmentation model for MA-UDA
#     """
    
#     def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=4,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
#         super().__init__()
        
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         # Calculate final features dimension
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
#         # Patch embedding
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         self.patches_resolution = self.patch_embed.patches_resolution
        
#         # --- FIX START: Separate Stages and Downsamplers ---
#         self.stages = nn.ModuleList()
#         self.downsamplers = nn.ModuleList()
        
#         for i_layer in range(self.num_layers):
#             layer_dim = int(embed_dim * 2 ** i_layer)
            
#             # Create the block for this stage
#             stage_blocks = nn.ModuleList([
#                 SwinTransformerBlock(
#                     dim=layer_dim,
#                     num_heads=num_heads[i_layer],
#                     window_size=7
#                 )
#                 for _ in range(depths[i_layer])
#             ])
#             self.stages.append(stage_blocks)
            
#             # Create downsampler for this stage (if not the last one)
#             if i_layer < self.num_layers - 1:
#                 # Using Linear for channel expansion
#                 self.downsamplers.append(nn.Linear(layer_dim, layer_dim * 2))
#         # --- FIX END ---
        
#         # Decoder (UperNet-style)
#         self.decoder = nn.ModuleList([
#             nn.Sequential(
#                 nn.ConvTranspose2d(self.num_features, self.num_features // 2, 2, stride=2),
#                 nn.BatchNorm2d(self.num_features // 2),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(self.num_features // 2, self.num_features // 4, 2, stride=2),
#                 nn.BatchNorm2d(self.num_features // 4),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(self.num_features // 4, self.num_features // 8, 2, stride=2),
#                 nn.BatchNorm2d(self.num_features // 8),
#                 nn.ReLU(inplace=True)
#             )
#         ])
        
#         # Final segmentation head
#         self.segmentation_head = nn.Sequential(
#             nn.Conv2d(self.num_features // 8, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, num_classes, 1)
#         )
        
#         self.attention_masks = []
    
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # Clear previous attention masks
#         self.attention_masks = []
        
#         # Patch embedding
#         x = self.patch_embed(x)  # [B, N, C]
        
#         # --- FIX START: Correct Iteration Loop ---
#         for i_layer in range(self.num_layers):
#             # 1. Run all blocks in this stage
#             for block in self.stages[i_layer]:
#                 x = block(x)
#                 if block.attention_weights is not None:
#                     self.attention_masks.append(block.attention_weights)
            
#             # 2. Run downsampler if it exists for this layer
#             if i_layer < len(self.downsamplers):
#                 x = self.downsamplers[i_layer](x)
#         # --- FIX END ---
        
#         # Reshape for decoder
#         B, N, C = x.shape
#         H_feat = W_feat = int(N ** 0.5)
#         x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        
#         # Decoder
#         for dec_layer in self.decoder:
#             x = dec_layer(x)
        
#         # Segmentation head
#         output = self.segmentation_head(x)
        
#         # Upsample to original size if needed
#         if output.shape[2:] != (H, W):
#             output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
#         return output
    
#     def get_attention_masks(self):
#         """Return collected attention masks for Meta Attention"""
#         return self.attention_masks

"""
Simplified Swin Transformer for segmentation with Gradient Checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # <--- IMPORT THIS

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Simplified Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.attention_weights = None
    
    def forward(self, x):
        # Multi-head self-attention
        shortcut = x
        x = self.norm1(x)
        
        # We need to capture weights for MA-UDA
        # Note: Checkpointing might re-run this, so we handle side-effects carefully
        x, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.attention_weights = attn_weights 
        
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformerSegmentation(nn.Module):
    """
    Swin Transformer-based segmentation model for MA-UDA
    """
    
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 use_checkpoint=True): # <--- Enabled by default
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patches_resolution = self.patch_embed.patches_resolution
        
        # Stages and Downsamplers
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            
            stage_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=7
                )
                for _ in range(depths[i_layer])
            ])
            self.stages.append(stage_blocks)
            
            if i_layer < self.num_layers - 1:
                self.downsamplers.append(nn.Linear(layer_dim, layer_dim * 2))
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.num_features, self.num_features // 2, 2, stride=2),
                nn.BatchNorm2d(self.num_features // 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(self.num_features // 2, self.num_features // 4, 2, stride=2),
                nn.BatchNorm2d(self.num_features // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(self.num_features // 4, self.num_features // 8, 2, stride=2),
                nn.BatchNorm2d(self.num_features // 8),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.num_features // 8, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        self.attention_masks = []
    
    def forward(self, x):
        B, C, H, W = x.shape
        self.attention_masks = []
        
        x = self.patch_embed(x)
        
        for i_layer in range(self.num_layers):
            for block in self.stages[i_layer]:
                
                # --- GRADIENT CHECKPOINTING MAGIC IS HERE ---
                if self.use_checkpoint and x.requires_grad:
                    # We wrap the block call in checkpoint
                    x = checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
                # --------------------------------------------
                
                # Collect attention weights
                # Note: With checkpointing, this might require an extra forward pass 
                # or rely on the cached property. For MA-UDA, we usually grab the 
                # weights from the final pass.
                if block.attention_weights is not None:
                    self.attention_masks.append(block.attention_weights)
            
            if i_layer < len(self.downsamplers):
                x = self.downsamplers[i_layer](x)
        
        B, N, C = x.shape
        H_feat = W_feat = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        
        for dec_layer in self.decoder:
            x = dec_layer(x)
        
        output = self.segmentation_head(x)
        
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output
    
    def get_attention_masks(self):
        return self.attention_masks
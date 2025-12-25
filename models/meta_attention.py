"""
Meta Attention implementation for MA-UDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAttention(nn.Module):
    """
    Meta Attention module that computes attention of multi-head attention masks
    As described in the paper: "Attention in Attention"
    """
    
    def __init__(self, num_heads, hidden_dim):
        super(MetaAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Learnable parameters for meta attention
        self.theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, mha_masks):
        """
        Args:
            mha_masks: Multi-head attention masks from Transformer
                      Shape: [B, num_heads, N, N] where N is number of patches
        Returns:
            meta_attention: Aggregated attention map
                           Shape: [B, N, N]
        """
        B, num_heads, N, _ = mha_masks.shape
        
        # Compute cross-attention for each head
        # For each position (i,j), compute attention using same row and column
        meta_attention_list = []
        
        for k in range(num_heads):
            mask_k = mha_masks[:, k, :, :]  # [B, N, N]
            
            # Cross-attention: row and column products
            # For position (i,j), use mask_k[i,:] * mask_k[:,j]
            row_attention = mask_k.unsqueeze(3)  # [B, N, N, 1]
            col_attention = mask_k.unsqueeze(2)  # [B, N, 1, N]
            
            # Element-wise product
            cross_attn = row_attention * col_attention  # [B, N, N, N]
            
            # Sum over one dimension to get [B, N, N]
            cross_attn = cross_attn.sum(dim=2)  # [B, N, N]
            
            meta_attention_list.append(cross_attn)
        
        # Aggregate across all heads
        stacked_attention = torch.stack(meta_attention_list, dim=1)  # [B, num_heads, N, N]
        
        # Sum across heads
        aggregated = stacked_attention.sum(dim=1)  # [B, N, N]
        
        # Apply learnable transformation
        # Reshape for linear layer
        B, N, N = aggregated.shape
        aggregated_flat = aggregated.view(B * N * N, -1)
        
        # Apply theta transformation
        # For simplicity, we apply softmax directly
        meta_attention = F.softmax(aggregated.view(B, -1), dim=1).view(B, N, N)
        
        return meta_attention

class MultiHeadAttentionExtractor(nn.Module):
    """
    Extract multi-head attention masks from Transformer blocks
    """
    
    def __init__(self):
        super(MultiHeadAttentionExtractor, self).__init__()
        self.attention_masks = []
    
    def register_hooks(self, model):
        """
        Register forward hooks to extract attention from Swin Transformer blocks
        """
        def hook_fn(module, input, output):
            # Store attention masks during forward pass
            if hasattr(module, 'attn'):
                # Extract attention from the module
                self.attention_masks.append(output)
        
        # Register hooks for all attention layers
        for name, module in model.named_modules():
            if 'attn' in name or isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(hook_fn)
    
    def get_attention_masks(self):
        """Get collected attention masks"""
        return self.attention_masks
    
    def clear_masks(self):
        """Clear stored masks"""
        self.attention_masks = []

class AttentionAlignmentDiscriminator(nn.Module):
    """
    Discriminator for attention-level alignment
    Simplified discriminator for attention maps
    """
    
    def __init__(self, input_size):
        super(AttentionAlignmentDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Attention map [B, N, N]
        Returns:
            Discriminator output
        """
        # Add channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # [B, 1, N, N]
        
        return self.model(x)
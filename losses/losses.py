"""
Loss functions for MA-UDA framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Generalized Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - predicted logits
            target: [B, H, W] - ground truth labels
        """
        # Convert target to one-hot
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax on predictions
        pred = F.softmax(pred, dim=1)
        
        # Flatten
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], 
                                                           target_one_hot.shape[1], -1)
        
        # Compute Dice
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean dice loss across classes
        return 1 - dice.mean()

class FastDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(FastDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        
        # 🟢 THE FIX: CALM DOWN WEIGHTS (10.0 -> 4.0)
        # We lower the penalty so it stops predicting tumor everywhere
        weights = torch.tensor([1.0, 4.0, 4.0, 4.0]).cuda()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = FastDiceLoss() 
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return ce + (2.0 * dice)

class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for CycleGAN"""
    
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real, reconstructed):
        return self.l1_loss(real, reconstructed)

class AdversarialLoss(nn.Module):
    """GAN adversarial loss"""
    
    def __init__(self, loss_type='lsgan'):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, is_real):
        if self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        else:
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        
        return self.loss(pred, target)

class PredictionConsistencyLoss(nn.Module):
    """Prediction consistency loss for unlabeled target domain"""
    
    def __init__(self):
        super(PredictionConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred1, pred2):
        """
        Args:
            pred1: Prediction for target image
            pred2: Prediction for generated target image
        """
        # Apply softmax to get probabilities
        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)
        
        return self.mse_loss(pred1, pred2)

class AttentionAlignmentLoss(nn.Module):
    """Adversarial loss for attention alignment"""
    
    def __init__(self):
        super(AttentionAlignmentLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward_generator(self, disc_output):
        """Loss for generator (trying to fool discriminator)"""
        target = torch.ones_like(disc_output)
        return self.bce_loss(disc_output, target)
    
    def forward_discriminator(self, real_output, fake_output):
        """Loss for discriminator"""
        real_target = torch.ones_like(real_output)
        fake_target = torch.zeros_like(fake_output)
        
        real_loss = self.bce_loss(real_output, real_target)
        fake_loss = self.bce_loss(fake_output, fake_target)
        
        return (real_loss + fake_loss) * 0.5

class MAUDALoss:
    """Complete loss for MA-UDA framework"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize all losses
        self.seg_loss = SegmentationLoss()
        self.adv_loss = AdversarialLoss(loss_type='lsgan')
        self.cycle_loss = CycleConsistencyLoss()
        self.pred_consistency_loss = PredictionConsistencyLoss()
        self.attn_alignment_loss = AttentionAlignmentLoss()
    
    def compute_segmentation_loss(self, pred_source, target_source, 
                                   pred_source_to_target, target_source_to_target):
        """Compute segmentation loss for source domain"""
        loss_s = self.seg_loss(pred_source, target_source)
        loss_s2t = self.seg_loss(pred_source_to_target, target_source_to_target)
        
        return loss_s + loss_s2t
    
    def compute_prediction_consistency(self, pred_target, pred_target_to_source):
        """Compute prediction consistency for target domain"""
        return self.pred_consistency_loss(pred_target, pred_target_to_source)
    
    def compute_total_loss(self, **kwargs):
        """
        Compute total loss for MA-UDA
        
        Required kwargs:
        - pred_source, target_source
        - pred_source_to_target, target_source_to_target
        - pred_target, pred_target_to_source
        - ma_source, ma_target_to_source, disc_s_output
        - ma_target, ma_source_to_target, disc_t_output
        """
        config = self.config
        
        # Segmentation loss
        loss_seg = self.compute_segmentation_loss(
            kwargs['pred_source'], kwargs['target_source'],
            kwargs['pred_source_to_target'], kwargs['target_source_to_target']
        )
        
        # Prediction consistency
        loss_pred = self.compute_prediction_consistency(
            kwargs['pred_target'], kwargs['pred_target_to_source']
        )
        
        # Meta attention alignment (if available)
        loss_ma = 0
        if 'disc_s_output' in kwargs and 'disc_t_output' in kwargs:
            loss_ma += self.attn_alignment_loss.forward_generator(kwargs['disc_s_output'])
            loss_ma += self.attn_alignment_loss.forward_generator(kwargs['disc_t_output'])
        
        # Total loss
        total_loss = (config.lambda_seg * loss_seg + 
                     config.lambda_pred * loss_pred + 
                     config.lambda_ma * loss_ma)
        
        return {
            'total': total_loss,
            'seg': loss_seg,
            'pred': loss_pred,
            'ma': loss_ma
        }


# """
# Loss functions for MA-UDA framework (Optimized for Speed)
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FastDiceLoss(nn.Module):
#     """Memory-efficient Dice Loss"""
#     def __init__(self, smooth=1e-5):
#         super(FastDiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         """
#         pred: [B, C, H, W] logits
#         target: [B, H, W] labels
#         """
#         # 1. Get Softmax probabilities
#         pred = F.softmax(pred, dim=1)
        
#         # 2. Create One-Hot efficiently (avoid permute/view overhead if possible)
#         num_classes = pred.shape[1]
        
#         # Optimized One-Hot: [B, H, W] -> [B, C, H, W] directly
#         target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
#         # 3. Compute stats without flattening huge tensors
#         # Sum over spatial dims (H, W) -> [B, C]
#         intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
#         union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
#         # 4. Dice Score
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
#         # 5. Average over classes and batch
#         return 1.0 - dice.mean()

# class SegmentationLoss(nn.Module):
#     """Combined Cross Entropy and Dice Loss"""
#     def __init__(self):
#         super(SegmentationLoss, self).__init__()
#         # ignore_index=0 is often used for background, but BraTS usually counts background.
#         # If your background dominates, consider class weights.
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dice_loss = FastDiceLoss()
    
#     def forward(self, pred, target):
#         # CE is fast and stable
#         ce = self.ce_loss(pred, target)
#         # Dice helps with class imbalance
#         dice = self.dice_loss(pred, target)
#         return ce + dice

# class AdversarialLoss(nn.Module):
#     """GAN adversarial loss"""
#     def __init__(self, loss_type='lsgan'):
#         super(AdversarialLoss, self).__init__()
#         self.loss_type = loss_type
#         if loss_type == 'lsgan':
#             self.loss = nn.MSELoss()
#         else:
#             self.loss = nn.BCEWithLogitsLoss()
    
#     def forward(self, pred, is_real):
#         # Use register_buffer trick or just creating simplified targets
#         target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
#         return self.loss(pred, target)

# class PredictionConsistencyLoss(nn.Module):
#     """Prediction consistency loss for unlabeled target domain"""
#     def __init__(self):
#         super(PredictionConsistencyLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
    
#     def forward(self, pred1, pred2):
#         # Optimized: Detach one branch to stop gradients flowing back into the "target"
#         # usually in consistency loss, one side is the 'teacher' (detached) and one is 'student'.
#         # However, MA-UDA paper might update both. If OOM persists, detach pred2.
#         p1 = F.softmax(pred1, dim=1)
#         p2 = F.softmax(pred2, dim=1)
#         return self.mse_loss(p1, p2)

# class AttentionAlignmentLoss(nn.Module):
#     def __init__(self):
#         super(AttentionAlignmentLoss, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
    
#     def forward_generator(self, disc_output):
#         target = torch.ones_like(disc_output)
#         return self.bce_loss(disc_output, target)
    
#     def forward_discriminator(self, real_output, fake_output):
#         real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
#         fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
#         return (real_loss + fake_loss) * 0.5

# class MAUDALoss:
#     """Complete loss for MA-UDA framework"""
#     def __init__(self, config):
#         self.config = config
#         self.seg_loss = SegmentationLoss()
#         self.pred_consistency_loss = PredictionConsistencyLoss()
#         self.attn_alignment_loss = AttentionAlignmentLoss()
    
#     def compute_total_loss(self, **kwargs):
#         config = self.config
        
#         # 1. Segmentation Loss (Source Domain)
#         # Combine items to save function calls
#         loss_s = self.seg_loss(kwargs['pred_source'], kwargs['target_source'])
#         loss_s2t = self.seg_loss(kwargs['pred_source_to_target'], kwargs['target_source_to_target'])
#         loss_seg = loss_s + loss_s2t
        
#         # 2. Prediction Consistency (Target Domain)
#         loss_pred = self.pred_consistency_loss(
#             kwargs['pred_target'], kwargs['pred_target_to_source']
#         )
        
#         # 3. Meta Attention (Only compute if needed)
#         loss_ma = torch.tensor(0.0, device=config.device)
#         if kwargs.get('disc_s_output') is not None:
#             loss_ma += self.attn_alignment_loss.forward_generator(kwargs['disc_s_output'])
#             loss_ma += self.attn_alignment_loss.forward_generator(kwargs['disc_t_output'])
        
#         # 4. Total Loss
#         total_loss = (config.lambda_seg * loss_seg + 
#                      config.lambda_pred * loss_pred + 
#                      config.lambda_ma * loss_ma)
        
#         return {
#             'total': total_loss,
#             'seg': loss_seg,
#             'pred': loss_pred,
#             'ma': loss_ma
#         }
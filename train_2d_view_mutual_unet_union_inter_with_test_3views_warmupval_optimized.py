import os
import argparse
import random
import numpy as np
import torch
import shutil
import pdb
import tifffile as tiff
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pandas as pd

from dataset import FullDataset_160
from model import UNet2D
from eval_volumes import eval_two_volumes_maxpool


def set_random_seed(seed):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic Python/NumPy seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
    

def soft_skeletonize_2d(x, thresh_width=5):
    '''
    Approximate differentiable skeletonization in 2D.
    x: Input tensor of shape (Batch, Channel, H, W), preferably in [0, 1].
    thresh_width: Expected maximum axon width, expressed as an iteration count.
    '''
    for i in range(thresh_width):
        # 1. Find horizontal and vertical local minima using max-pooling on negated input.
        # p1: vertical (3, 1); p2: horizontal (1, 3).
        p1 = F.max_pool2d(x * -1, (3, 1), 1, (1, 0)) * -1
        p2 = F.max_pool2d(x * -1, (1, 3), 1, (0, 1)) * -1
        
        # 2. Take the minimum of the two directional pooling results.
        min_pool_x = torch.min(p1, p2)
        
        # 3. Compute the contour as the local maximum minus the local minimum.
        # Find the local maximum of min_pool_x with a 3x3 window.
        contour = F.relu(F.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        
        # 4. Subtract the contour from the input to thin it.
        x = F.relu(x - contour)
        
    return x


def clDice_loss_2d(y_pred, y_true, iters=5):
    '''
    y_pred: Model prediction after sigmoid.
    y_true: Skeleton label.
    '''
    # Soft-skeletonize the prediction.
    skel_pred = soft_skeletonize_2d(y_pred, thresh_width=iters)
    # Soft-skeletonize the label; fewer iterations may suffice for a one-pixel skeleton.
    skel_true = soft_skeletonize_2d(y_true, thresh_width=iters)
    
    # Compute Tprec and Tsens from the overlap between skeletons and masks.
    t_prec = (torch.sum(skel_pred * y_true) + 1e-6) / (torch.sum(skel_pred) + 1e-6)
    t_sens = (torch.sum(skel_true * y_pred) + 1e-6) / (torch.sum(skel_true) + 1e-6)
    
    cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return cl_dice


def plot_loss_curves_with_3d_cldice(train_losses_axial, train_losses_coronal, train_losses_sagittal,
                                     val_losses_axial, val_losses_coronal, val_losses_sagittal,
                                     val_cldices_axial, val_cldices_coronal, val_cldices_sagittal,
                                     mutual_losses, output_path, show_best_annotation=True):
    """Plot and save training and validation losses, annotating the best epoch with its 3D ClDice."""
    epochs_axial = range(1, len(train_losses_axial) + 1)
    epochs_coronal = range(1, len(train_losses_coronal) + 1)
    epochs_sagittal = range(1, len(train_losses_sagittal) + 1)

    # Figure 1: axial training and validation loss (two panels).
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    ax1a.plot(epochs_axial, train_losses_axial, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax1a.legend(fontsize=11)
    ax1a.set_xlabel('Epoch', fontsize=12)
    ax1a.set_ylabel('Loss', fontsize=12)
    ax1a.set_title('Axial: Training Loss', fontsize=14)
    ax1a.grid(True, alpha=0.3)

    if val_losses_axial:
        ax1b.plot(epochs_axial, val_losses_axial, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
        if show_best_annotation:
            # Mark the point with the highest 3D ClDice.
            best_idx = val_cldices_axial.index(max(val_cldices_axial)) if val_cldices_axial else val_losses_axial.index(min(val_losses_axial))
            best_cldice = val_cldices_axial[best_idx] if val_cldices_axial else 0.0
            best_value = val_losses_axial[best_idx]
            best_epoch = best_idx + 1
            ax1b.scatter([best_epoch], [best_value], color='green', s=150, marker='*', zorder=5)
            
            # Adjust the annotation position based on the best epoch.
            if best_epoch > len(val_losses_axial) * 0.7:
                xytext_offset = (-80, 30)
                ha = 'right'
            else:
                xytext_offset = (10, 30)
                ha = 'left'
            
            # Annotate the validation loss and 3D ClDice at the best point.
            annotation_text = f'Best 3D ClDice: {best_cldice:.4f}\nVal Loss: {best_value:.4f}\n(Epoch {best_epoch})'
            
            ax1b.annotate(annotation_text,
                         xy=(best_epoch, best_value), xytext=xytext_offset,
                         textcoords='offset points', fontsize=9, color='green',
                         fontweight='bold', ha=ha,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color='green'))

    ax1b.set_xlabel('Epoch', fontsize=12)
    ax1b.set_ylabel('Loss', fontsize=12)
    ax1b.set_title('Axial: Validation Loss (Lower is Better)', fontsize=14)
    ax1b.legend(fontsize=11)
    ax1b.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_path, 'loss_curves_axial_unet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: coronal training and validation loss (two panels).
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    ax2a.plot(epochs_coronal, train_losses_coronal, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax2a.legend(fontsize=11)
    ax2a.set_xlabel('Epoch', fontsize=12)
    ax2a.set_ylabel('Loss', fontsize=12)
    ax2a.set_title('Coronal: Training Loss', fontsize=14)
    ax2a.grid(True, alpha=0.3)

    if val_losses_coronal:
        ax2b.plot(epochs_coronal, val_losses_coronal, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
        if show_best_annotation:
            # Mark the point with the highest 3D ClDice.
            best_idx = val_cldices_coronal.index(max(val_cldices_coronal)) if val_cldices_coronal else val_losses_coronal.index(min(val_losses_coronal))
            best_cldice = val_cldices_coronal[best_idx] if val_cldices_coronal else 0.0
            best_value = val_losses_coronal[best_idx]
            best_epoch = best_idx + 1
            ax2b.scatter([best_epoch], [best_value], color='green', s=150, marker='*', zorder=5)
            
            # Adjust the annotation position based on the best epoch.
            if best_epoch > len(val_losses_coronal) * 0.7:
                xytext_offset = (-80, 30)
                ha = 'right'
            else:
                xytext_offset = (10, 30)
                ha = 'left'
            
            # Annotate the validation loss and 3D ClDice at the best point.
            annotation_text = f'Best 3D ClDice: {best_cldice:.4f}\nVal Loss: {best_value:.4f}\n(Epoch {best_epoch})'
            
            ax2b.annotate(annotation_text,
                         xy=(best_epoch, best_value), xytext=xytext_offset,
                         textcoords='offset points', fontsize=9, color='green',
                         fontweight='bold', ha=ha,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color='green'))

    ax2b.set_xlabel('Epoch', fontsize=12)
    ax2b.set_ylabel('Loss', fontsize=12)
    ax2b.set_title('Coronal: Validation Loss (Lower is Better)', fontsize=14)
    ax2b.legend(fontsize=11)
    ax2b.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_path, 'loss_curves_coronal_unet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Figure 3: sagittal training and validation loss (two panels).
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    ax3a.plot(epochs_sagittal, train_losses_sagittal, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax3a.legend(fontsize=11)
    ax3a.set_xlabel('Epoch', fontsize=12)
    ax3a.set_ylabel('Loss', fontsize=12)
    ax3a.set_title('Sagittal: Training Loss', fontsize=14)
    ax3a.grid(True, alpha=0.3)

    if val_losses_sagittal:
        ax3b.plot(epochs_sagittal, val_losses_sagittal, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
        if show_best_annotation:
            # Mark the point with the highest 3D ClDice.
            best_idx = val_cldices_sagittal.index(max(val_cldices_sagittal)) if val_cldices_sagittal else val_losses_sagittal.index(min(val_losses_sagittal))
            best_cldice = val_cldices_sagittal[best_idx] if val_cldices_sagittal else 0.0
            best_value = val_losses_sagittal[best_idx]
            best_epoch = best_idx + 1
            ax3b.scatter([best_epoch], [best_value], color='green', s=150, marker='*', zorder=5)
            
            # Adjust the annotation position based on the best epoch.
            if best_epoch > len(val_losses_sagittal) * 0.7:
                xytext_offset = (-80, 30)
                ha = 'right'
            else:
                xytext_offset = (10, 30)
                ha = 'left'
            
            # Annotate the validation loss and 3D ClDice at the best point.
            annotation_text = f'Best 3D ClDice: {best_cldice:.4f}\nVal Loss: {best_value:.4f}\n(Epoch {best_epoch})'
            
            ax3b.annotate(annotation_text,
                         xy=(best_epoch, best_value), xytext=xytext_offset,
                         textcoords='offset points', fontsize=9, color='green',
                         fontweight='bold', ha=ha,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color='green'))

    ax3b.set_xlabel('Epoch', fontsize=12)
    ax3b.set_ylabel('Loss', fontsize=12)
    ax3b.set_title('Sagittal: Validation Loss (Lower is Better)', fontsize=14)
    ax3b.legend(fontsize=11)
    ax3b.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_path, 'loss_curves_sagittal_unet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # Figure 4: mutual loss, if mutual_losses is nonempty.
    if mutual_losses:
        epochs_mutual = range(1, len(mutual_losses) + 1)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(epochs_mutual, mutual_losses, 'r-', label='Mutual Loss', linewidth=2, marker='^', markersize=4)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.set_title('Mutual Learning Loss', fontsize=14)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        fig4.savefig(os.path.join(output_path, 'loss_curves_mutual_unet.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)

    print(f"Saved loss curves to {output_path}")


def structure_loss(pred, mask, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
    """BCE + Dice structure loss."""
    # BCE with logits (numerically stable)
    bce = F.binary_cross_entropy_with_logits(pred, mask)

    # Dice loss
    pred_sig = torch.sigmoid(pred)
    pred_flat = pred_sig.view(pred_sig.size(0), -1)
    mask_flat = mask.view(mask.size(0), -1)

    intersection = (pred_flat * mask_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return bce_weight * bce + dice_weight * dice_loss


def reconstruct_and_eval(pred_slices_list, dataset, image_index, view, device):
    pred_slices_sorted = [p for _, p in sorted(pred_slices_list, key=lambda x: x[0])]
    pred_stack = np.stack([p.squeeze() for p in pred_slices_sorted], axis=0)

    label_3d = dataset.get_original_label_3d(image_index)
    if view == "coronal":
        label_3d = np.transpose(label_3d, (1, 0, 2))

    if view == "sagittal":
        label_3d = np.transpose(label_3d, (2, 0, 1))

    cldice_res = eval_two_volumes_maxpool(pred_stack, label_3d, pool_kernel=3, threshold=0.5, device=device)
    
    return cldice_res, pred_stack, label_3d


def build_teacher_prob_volume(model, data_loader, device, teacher_view, D=160, H=160, W=160):
    """
    Build a 3D probability volume for the teacher.
    Return: dict {volume_idx: probability_volume (H, W, D)}.
    """
    model.eval()
    teacher_probs = {}  # {volume_idx: np.ndarray of shape (H, W, D)}

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            image_indices = batch.get('image_index', None)
            slice_idxs = batch.get('slice_idx', None)

            # UNet2D returns a single output.
            pred = model(images)
            probs = torch.sigmoid(pred).detach().cpu()  # (B, 1, H, W)

            for i in range(probs.shape[0]):
                vid = int(image_indices[i].item())
                sidx = int(slice_idxs[i].item())
                slice2d = probs[i, 0]  # (160,160)

                if vid not in teacher_probs:
                    teacher_probs[vid] = np.zeros((D, H, W), dtype=np.float32)

                # -----------------------------
                # Place predictions in common DHW coordinates.
                # -----------------------------
                if teacher_view == "axial":
                    # slice_idx = d
                    # slice2d is (H,W)
                    teacher_probs[vid][sidx, :, :] = slice2d

                elif teacher_view == "coronal":
                    # slice_idx = h
                    # slice2d is (D,W)
                    teacher_probs[vid][:, sidx, :] = slice2d

                elif teacher_view == "sagittal":
                    # slice_idx = w
                    # slice2d is (D,H)
                    teacher_probs[vid][:, :, sidx] = slice2d

                else:
                    raise ValueError(f"Unknown teacher_view: {teacher_view}")

    return teacher_probs


def convert_view_probabilities(prob_3d, student_view):
    """
    prob_3d: (D,H,W)

    returns volume where slice axis is last:
        axial    -> (H, W, D)
        coronal  -> (D, W, H)
        sagittal -> (D, H, W)
    """

    if student_view == "axial":
        return np.transpose(prob_3d, (1, 2, 0))   # (H,W,D)

    elif student_view == "coronal":
        return np.transpose(prob_3d, (0, 2, 1))   # (D,W,H)

    elif student_view == "sagittal":
        return prob_3d                            # (D,H,W)

    else:
        raise ValueError(student_view)


def main(args):
    if not 0.0 <= args.prob_threshold <= 1.0:
        raise ValueError("--prob_threshold must be between 0 and 1")
    if args.rotation_interval < 1:
        raise ValueError("--rotation_interval must be a positive integer")

    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    val_image_dir = os.path.join(args.output_path, "val_image")
    os.makedirs(val_image_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ Build three-view datasets (same paths, different views) ============
    train_dataset_axial = FullDataset_160(
        image_root=args.train_image_path, gt_root=args.train_mask_path,
        size=args.image_size, mode='train', view='axial'
    )
    train_dataset_coronal = FullDataset_160(
        image_root=args.train_image_path, gt_root=args.train_mask_path,
        size=args.image_size, mode='train', view='coronal'
    )
    train_dataset_sagittal = FullDataset_160(
        image_root=args.train_image_path, gt_root=args.train_mask_path,
        size=args.image_size, mode='train', view='sagittal'
    )

    val_dataset_axial = FullDataset_160(
        image_root=args.val_image_path, gt_root=args.val_mask_path,
        size=args.image_size, mode='val', view='axial'
    )
    val_dataset_coronal = FullDataset_160(
        image_root=args.val_image_path, gt_root=args.val_mask_path,
        size=args.image_size, mode='val', view='coronal'
    )
    val_dataset_sagittal = FullDataset_160(
        image_root=args.val_image_path, gt_root=args.val_mask_path,
        size=args.image_size, mode='val', view='sagittal'
    )

    # Test dataset.
    test_dataset_axial = FullDataset_160(
        image_root=args.test_image_path, gt_root=args.test_mask_path,
        size=args.image_size, mode='val', view='axial'
    )
    test_dataset_coronal = FullDataset_160(
        image_root=args.test_image_path, gt_root=args.test_mask_path,
        size=args.image_size, mode='val', view='coronal'
    )
    test_dataset_sagittal = FullDataset_160(
        image_root=args.test_image_path, gt_root=args.test_mask_path,
        size=args.image_size, mode='val', view='sagittal'
    )

    train_loader_axial = DataLoader(train_dataset_axial, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True,
                                    worker_init_fn=seed_worker, generator=make_generator(args.seed))
    train_loader_coronal = DataLoader(train_dataset_coronal, batch_size=args.batch_size,
                                      shuffle=True, num_workers=4, pin_memory=True,
                                      worker_init_fn=seed_worker, generator=make_generator(args.seed))
    train_loader_sagittal = DataLoader(train_dataset_sagittal, batch_size=args.batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True,
                                       worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader_axial = DataLoader(val_dataset_axial, batch_size=args.val_batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader_coronal = DataLoader(val_dataset_coronal, batch_size=args.val_batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True,
                                    worker_init_fn=seed_worker, generator=make_generator(args.seed))
    val_loader_sagittal = DataLoader(val_dataset_sagittal, batch_size=args.val_batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True,
                                     worker_init_fn=seed_worker, generator=make_generator(args.seed))

    # Test-set DataLoader.
    test_loader_axial = DataLoader(test_dataset_axial, batch_size=args.val_batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True,
                                    worker_init_fn=seed_worker, generator=make_generator(args.seed))
    test_loader_coronal = DataLoader(test_dataset_coronal, batch_size=args.val_batch_size,
                                      shuffle=False, num_workers=4, pin_memory=True,
                                      worker_init_fn=seed_worker, generator=make_generator(args.seed))
    test_loader_sagittal = DataLoader(test_dataset_sagittal, batch_size=args.val_batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True,
                                       worker_init_fn=seed_worker, generator=make_generator(args.seed))

    # ============ Create three UNet2D models ============
    model_axial = UNet2D(in_ch=3, num_classes=1).to(device)
    model_coronal = UNet2D(in_ch=3, num_classes=1).to(device)
    model_sagittal = UNet2D(in_ch=3, num_classes=1).to(device)

    # Optimizers.
    optimizer_axial = optim.AdamW(model_axial.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_coronal = optim.AdamW(model_coronal.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_sagittal = optim.AdamW(model_sagittal.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning-rate schedulers.
    scheduler_axial = optim.lr_scheduler.CosineAnnealingLR(optimizer_axial, T_max=args.epochs, eta_min=1e-7)
    scheduler_coronal = optim.lr_scheduler.CosineAnnealingLR(optimizer_coronal, T_max=args.epochs, eta_min=1e-7)
    scheduler_sagittal = optim.lr_scheduler.CosineAnnealingLR(optimizer_sagittal, T_max=args.epochs, eta_min=1e-7)

    # Checkpoint paths.
    ckpt_axial_path = os.path.join(args.save_path, "UNet2D-axial-latest.pth")
    ckpt_coronal_path = os.path.join(args.save_path, "UNet2D-coronal-latest.pth")
    ckpt_sagittal_path = os.path.join(args.save_path, "UNet2D-sagittal-latest.pth")

    start_epoch = 0
    best_val_loss_axial = float('inf')
    best_val_loss_coronal = float('inf')
    best_val_loss_sagittal = float('inf')
    best_3d_cldice_axial = 0.0
    best_3d_cldice_coronal = 0.0
    best_3d_cldice_sagittal = 0.0

    # Early-stopping state.
    patience = 5  # Early-stopping patience.
    no_improvement_count_axial = 0
    no_improvement_count_coronal = 0
    no_improvement_count_sagittal = 0

    # Record losses and metrics.
    train_losses_axial = []
    train_losses_coronal = []
    train_losses_sagittal = []
    val_losses_axial = []
    val_losses_coronal = []
    val_losses_sagittal = []
    val_cldices_axial = []  # 3D ClDice for each validation epoch.
    val_cldices_coronal = []  # 3D ClDice for each validation epoch.
    val_cldices_sagittal = []  # 3D ClDice for each validation epoch.
    val_dices_axial = []  # 3D Dice for each validation, including all-zero labels.
    val_dices_coronal = []
    val_dices_sagittal = []
    val_cldice_epochs_axial = []
    val_cldice_epochs_coronal = []
    val_cldice_epochs_sagittal = []
    mutual_losses = []

    # ============ Resume from checkpoints if available ============
    if os.path.exists(ckpt_axial_path) and os.path.exists(ckpt_coronal_path) and os.path.exists(ckpt_sagittal_path):
        print(f"Resuming from checkpoints...")
        ckpt_axial = torch.load(ckpt_axial_path, map_location=device)
        ckpt_coronal = torch.load(ckpt_coronal_path, map_location=device)
        ckpt_sagittal = torch.load(ckpt_sagittal_path, map_location=device)

        model_axial.load_state_dict(ckpt_axial['state_dict'], strict=False)
        model_coronal.load_state_dict(ckpt_coronal['state_dict'], strict=False)
        model_sagittal.load_state_dict(ckpt_sagittal['state_dict'], strict=False)

        optimizer_axial.load_state_dict(ckpt_axial['optimizer'])
        optimizer_coronal.load_state_dict(ckpt_coronal['optimizer'])
        optimizer_sagittal.load_state_dict(ckpt_sagittal['optimizer'])

        scheduler_axial.load_state_dict(ckpt_axial['scheduler'])
        scheduler_coronal.load_state_dict(ckpt_coronal['scheduler'])
        scheduler_sagittal.load_state_dict(ckpt_sagittal['scheduler'])

        start_epoch = max(ckpt_axial.get('epoch', 0), ckpt_coronal.get('epoch', 0), ckpt_sagittal.get('epoch', 0))
        best_val_loss_axial = ckpt_axial.get('best_val_loss', float('inf'))
        best_val_loss_coronal = ckpt_coronal.get('best_val_loss', float('inf'))
        best_val_loss_sagittal = ckpt_sagittal.get('best_val_loss', float('inf'))
        best_3d_cldice_axial = ckpt_axial.get('best_3d_cldice', 0.0)
        best_3d_cldice_coronal = ckpt_coronal.get('best_3d_cldice', 0.0)
        best_3d_cldice_sagittal = ckpt_sagittal.get('best_3d_cldice', 0.0)

        train_losses_axial = ckpt_axial.get('train_losses', [])
        train_losses_coronal = ckpt_coronal.get('train_losses', [])
        train_losses_sagittal = ckpt_sagittal.get('train_losses', [])
        val_losses_axial = ckpt_axial.get('val_losses', [])
        val_losses_coronal = ckpt_coronal.get('val_losses', [])
        val_losses_sagittal = ckpt_sagittal.get('val_losses', [])

        val_cldices_axial = ckpt_axial.get('val_cldices', [])
        val_cldices_coronal = ckpt_coronal.get('val_cldices', [])
        val_cldices_sagittal = ckpt_sagittal.get('val_cldices', [])
        # Use None placeholders when older checkpoints lack Dice history to preserve epoch alignment.
        val_dices_axial = ckpt_axial.get('val_dices', [None] * len(val_cldices_axial))
        val_dices_coronal = ckpt_coronal.get('val_dices', [None] * len(val_cldices_coronal))
        val_dices_sagittal = ckpt_sagittal.get('val_dices', [None] * len(val_cldices_sagittal))
        # Support older checkpoints without saved validation epochs.
        val_cldice_epochs_axial = ckpt_axial.get(
            'val_cldice_epochs',
            [args.warmup_epochs + 1 + 3 * i for i in range(len(val_cldices_axial))]
        )
        val_cldice_epochs_coronal = ckpt_coronal.get(
            'val_cldice_epochs',
            [args.warmup_epochs + 2 + 3 * i for i in range(len(val_cldices_coronal))]
        )
        val_cldice_epochs_sagittal = ckpt_sagittal.get(
            'val_cldice_epochs',
            [args.warmup_epochs + 3 + 3 * i for i in range(len(val_cldices_sagittal))]
        )

        if 'mutual_losses' in ckpt_axial:
            mutual_losses = ckpt_axial.get('mutual_losses', [])
        else:
            mutual_losses = []

        print(f"\n ====> Resumed from epoch {start_epoch}, best_val_loss_axial: {best_val_loss_axial:.4f}, best_val_loss_coronal: {best_val_loss_coronal:.4f}, best_val_loss_sagittal: {best_val_loss_sagittal:.4f}, best_3d_cldice_axial: {best_3d_cldice_axial:.4f}, best_3d_cldice_coronal: {best_3d_cldice_coronal:.4f}, best_3d_cldice_sagittal: {best_3d_cldice_sagittal:.4f} <====")

    # ============ Warmup phase: train each view independently ============
    warmup_epochs = args.warmup_epochs
    mutual_start_epoch = warmup_epochs + 1

    if start_epoch < warmup_epochs:
        print(f"\n{'='*60}")
        print(f"WARMUP PHASE: Epochs 1 - {warmup_epochs}")
        print(f"{'='*60}")

        # Warm up the axial model.
        warmup_end = min(warmup_epochs, args.epochs)
        for epoch in range(start_epoch + 1, mutual_start_epoch):
            print("\n>>> Warming up Axial model...")
            # Train the axial model.
            model_axial.train()
            total_loss = 0.0
            for b_idx, batch in enumerate(train_loader_axial):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer_axial.zero_grad()
                # UNet2D returns a single output.
                pred = model_axial(images)
                loss = structure_loss(pred, labels)
                loss.backward()
                optimizer_axial.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader_axial)
            print(f"[Axial Warmup Epoch {epoch}/{warmup_end}] Train loss: {avg_loss:.6f}")

            scheduler_axial.step()


            # Warm up the coronal model.
            print("\n>>> Warming up Coronal model...")
            model_coronal.train()
            total_loss = 0.0
            for b_idx, batch in enumerate(train_loader_coronal):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer_coronal.zero_grad()
                # UNet2D returns a single output.
                pred = model_coronal(images)
                loss = structure_loss(pred, labels)
                loss.backward()
                optimizer_coronal.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader_coronal)
            print(f"[Coronal Warmup Epoch {epoch}/{warmup_epochs}] Train loss: {avg_loss:.6f}")
            scheduler_coronal.step()


            # ============ Warm up the sagittal model ============
            print("\n>>> Warming up Sagittal model...")
            model_sagittal.train()
            total_loss = 0.0
            for b_idx, batch in enumerate(train_loader_sagittal):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer_sagittal.zero_grad()
                # UNet2D returns a single output.
                pred = model_sagittal(images)
                loss = structure_loss(pred, labels)
                loss.backward()
                optimizer_sagittal.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader_sagittal)
            print(f"[Sagittal Warmup Epoch {epoch}/{warmup_epochs}] Train loss: {avg_loss:.6f}")

            scheduler_sagittal.step()

        print("\n>>> Warmup completed! Starting mutual learning phase...")

    # ============ Mutual-learning phase ============
    print(f"\n{'='*60}")
    print(f"MUTUAL LEARNING PHASE: Epochs {mutual_start_epoch} - {args.epochs}")
    print(f"{'='*60}")


    for epoch in range(max(mutual_start_epoch, start_epoch + 1), args.epochs + 1):
        # Rotate the student view after rotation_interval consecutive epochs.
        student_view_idx = (
            (epoch - mutual_start_epoch) // args.rotation_interval
        ) % 3

        # Build dictionaries of teacher models and loaders.
        teacher_models = {
            'axial': model_axial,
            'coronal': model_coronal,
            'sagittal': model_sagittal
        }
        teacher_loaders = {
            'axial': train_loader_axial,
            'coronal': train_loader_coronal,
            'sagittal': train_loader_sagittal
        }

        if student_view_idx == 0:
            # Use axial as student and coronal and sagittal as teachers.
            print(f"\n============> [Epoch {epoch}] Axial as STUDENT, Coronal + Sagittal as TEACHER")
            mutual_loss = train_with_mutual_learning_3views(
                model_student=model_axial,
                optimizer_student=optimizer_axial,
                train_loader_student=train_loader_axial,
                teacher_models=teacher_models,
                teacher_loaders=teacher_loaders,
                device=device,
                student_view='axial',
                kl_weight=args.kl_weight,
                union_loss_weight=args.union_loss_weight,
                intersection_loss_weight=args.intersection_loss_weight,
                prob_threshold=args.prob_threshold,
                use_skeleton_for_intersection=args.use_skeleton_for_intersection
                
            )
            train_losses_axial.append(mutual_loss['student_loss'])

            # Validate.
            mean_val_loss, epoch_3d_cldice, epoch_3d_dice = validate_and_compute_metrics(
                model_axial, val_loader_axial, val_dataset_axial,
                'axial', epoch, val_image_dir, device
            )
            val_losses_axial.append(mean_val_loss)
            val_cldices_axial.append(epoch_3d_cldice)
            val_dices_axial.append(epoch_3d_dice)
            val_cldice_epochs_axial.append(epoch)

            # Track early stopping using validation loss.
            if mean_val_loss < best_val_loss_axial:
                best_val_loss_axial = mean_val_loss
                no_improvement_count_axial = 0
                print(f"New best val_loss (axial): {best_val_loss_axial:.4f}")
            else:
                no_improvement_count_axial += 1
                print(f"Val loss no improvement (axial): {no_improvement_count_axial}/{patience}")

            # Save the best model according to 3D ClDice (higher is better).
            if epoch_3d_cldice > best_3d_cldice_axial:
                best_3d_cldice_axial = epoch_3d_cldice
                save_checkpoint(model_axial, optimizer_axial, scheduler_axial, epoch,
                              best_val_loss_axial, val_losses_axial, train_losses_axial,
                              val_cldices_axial, val_dices_axial, val_cldice_epochs_axial,
                              mutual_losses, best_3d_cldice_axial,
                              args.save_path, 'axial', 'best')
                print(f"Saved best axial model (by ClDice) at epoch {epoch} with 3D ClDice: {best_3d_cldice_axial:.4f}")

                # Evaluate on the test set when validation ClDice reaches a new best.
                print(f">>> Evaluating on test set at best epoch {epoch}...")
                test_metrics = evaluate_on_test(model_axial, test_loader_axial, test_dataset_axial, 'axial', device, args.output_path)
                save_test_metrics(test_metrics, args.output_path, 'axial', epoch, best_3d_cldice_axial)

            scheduler_axial.step()
            save_checkpoint(model_axial, optimizer_axial, scheduler_axial, epoch,
                          best_val_loss_axial, val_losses_axial, train_losses_axial,
                          val_cldices_axial, val_dices_axial, val_cldice_epochs_axial,
                          mutual_losses, best_3d_cldice_axial,
                          args.save_path, 'axial', 'latest')

        elif student_view_idx == 1:
            # Use coronal as student and axial and sagittal as teachers.
            print(f"\n============> [Epoch {epoch}] Coronal as STUDENT, Axial + Sagittal as TEACHER")
            mutual_loss = train_with_mutual_learning_3views(
                model_student=model_coronal,
                optimizer_student=optimizer_coronal,
                train_loader_student=train_loader_coronal,
                teacher_models=teacher_models,
                teacher_loaders=teacher_loaders,
                device=device,
                student_view='coronal',
                kl_weight=args.kl_weight,
                union_loss_weight=args.union_loss_weight,
                intersection_loss_weight=args.intersection_loss_weight,
                prob_threshold=args.prob_threshold,
                use_skeleton_for_intersection=args.use_skeleton_for_intersection
            )
            train_losses_coronal.append(mutual_loss['student_loss'])

            # Validate.
            mean_val_loss, epoch_3d_cldice, epoch_3d_dice = validate_and_compute_metrics(
                model_coronal, val_loader_coronal, val_dataset_coronal,
                'coronal', epoch, val_image_dir, device
            )
            val_losses_coronal.append(mean_val_loss)
            val_cldices_coronal.append(epoch_3d_cldice)
            val_dices_coronal.append(epoch_3d_dice)
            val_cldice_epochs_coronal.append(epoch)

            # Track early stopping using validation loss.
            if mean_val_loss < best_val_loss_coronal:
                best_val_loss_coronal = mean_val_loss
                no_improvement_count_coronal = 0
                print(f"New best val_loss (coronal): {best_val_loss_coronal:.4f}")
            else:
                no_improvement_count_coronal += 1
                print(f"Val loss no improvement (coronal): {no_improvement_count_coronal}/{patience}")

            # Save the best model according to 3D ClDice (higher is better).
            if epoch_3d_cldice > best_3d_cldice_coronal:
                best_3d_cldice_coronal = epoch_3d_cldice
                save_checkpoint(model_coronal, optimizer_coronal, scheduler_coronal, epoch,
                              best_val_loss_coronal, val_losses_coronal, train_losses_coronal,
                              val_cldices_coronal, val_dices_coronal, val_cldice_epochs_coronal,
                              mutual_losses, best_3d_cldice_coronal,
                              args.save_path, 'coronal', 'best')
                print(f"Saved best coronal model (by ClDice) at epoch {epoch} with 3D ClDice: {best_3d_cldice_coronal:.4f}")

                # Evaluate on the test set when validation ClDice reaches a new best.
                print(f">>> Evaluating on test set at best epoch {epoch}...")
                test_metrics = evaluate_on_test(model_coronal, test_loader_coronal, test_dataset_coronal, 'coronal', device, args.output_path)
                save_test_metrics(test_metrics, args.output_path, 'coronal', epoch, best_3d_cldice_coronal)

            scheduler_coronal.step()
            save_checkpoint(model_coronal, optimizer_coronal, scheduler_coronal, epoch,
                          best_val_loss_coronal, val_losses_coronal, train_losses_coronal,
                          val_cldices_coronal, val_dices_coronal, val_cldice_epochs_coronal,
                          mutual_losses, best_3d_cldice_coronal,
                          args.save_path, 'coronal', 'latest')

        else:
            # Use sagittal as student and axial and coronal as teachers.
            print(f"\n============> [Epoch {epoch}] Sagittal as STUDENT, Axial + Coronal as TEACHER")
            mutual_loss = train_with_mutual_learning_3views(
                model_student=model_sagittal,
                optimizer_student=optimizer_sagittal,
                train_loader_student=train_loader_sagittal,
                teacher_models=teacher_models,
                teacher_loaders=teacher_loaders,
                device=device,
                student_view='sagittal',
                kl_weight=args.kl_weight,
                union_loss_weight=args.union_loss_weight,
                intersection_loss_weight=args.intersection_loss_weight,
                prob_threshold=args.prob_threshold,
                use_skeleton_for_intersection=args.use_skeleton_for_intersection
            )
            train_losses_sagittal.append(mutual_loss['student_loss'])

            # Validate.
            mean_val_loss, epoch_3d_cldice, epoch_3d_dice = validate_and_compute_metrics(
                model_sagittal, val_loader_sagittal, val_dataset_sagittal,
                'sagittal', epoch, val_image_dir, device
            )
            val_losses_sagittal.append(mean_val_loss)
            val_cldices_sagittal.append(epoch_3d_cldice)
            val_dices_sagittal.append(epoch_3d_dice)
            val_cldice_epochs_sagittal.append(epoch)

            # Track early stopping using validation loss.
            if mean_val_loss < best_val_loss_sagittal:
                best_val_loss_sagittal = mean_val_loss
                no_improvement_count_sagittal = 0
                print(f"New best val_loss (sagittal): {best_val_loss_sagittal:.4f}")
            else:
                no_improvement_count_sagittal += 1
                print(f"Val loss no improvement (sagittal): {no_improvement_count_sagittal}/{patience}")

            # Save the best model according to 3D ClDice (higher is better).
            if epoch_3d_cldice > best_3d_cldice_sagittal:
                best_3d_cldice_sagittal = epoch_3d_cldice
                save_checkpoint(model_sagittal, optimizer_sagittal, scheduler_sagittal, epoch,
                              best_val_loss_sagittal, val_losses_sagittal, train_losses_sagittal,
                              val_cldices_sagittal, val_dices_sagittal, val_cldice_epochs_sagittal,
                              mutual_losses, best_3d_cldice_sagittal,
                              args.save_path, 'sagittal', 'best')
                print(f"Saved best sagittal model (by ClDice) at epoch {epoch} with 3D ClDice: {best_3d_cldice_sagittal:.4f}")

                # Evaluate on the test set when validation ClDice reaches a new best.
                print(f">>> Evaluating on test set at best epoch {epoch}...")
                test_metrics = evaluate_on_test(model_sagittal, test_loader_sagittal, test_dataset_sagittal, 'sagittal', device, args.output_path)
                save_test_metrics(test_metrics, args.output_path, 'sagittal', epoch, best_3d_cldice_sagittal)

            scheduler_sagittal.step()
            save_checkpoint(model_sagittal, optimizer_sagittal, scheduler_sagittal, epoch,
                          best_val_loss_sagittal, val_losses_sagittal, train_losses_sagittal,
                          val_cldices_sagittal, val_dices_sagittal, val_cldice_epochs_sagittal,
                          mutual_losses, best_3d_cldice_sagittal,
                          args.save_path, 'sagittal', 'latest')

        # Record mutual-learning loss.
        mutual_losses.append(mutual_loss.get('mutual_kd_loss', 0.0))

        # Plot the curves.
        plot_loss_curves_with_3d_cldice(train_losses_axial, train_losses_coronal, train_losses_sagittal,
                        val_losses_axial, val_losses_coronal, val_losses_sagittal,
                        val_cldices_axial, val_cldices_coronal, val_cldices_sagittal,
                        mutual_losses, args.output_path)

    print("\nTraining finished!")


def train_with_mutual_learning_3views(model_student, optimizer_student,
                                       train_loader_student,
                                       teacher_models,
                                       teacher_loaders,
                                       device,
                                       student_view='axial',
                                       kl_weight=0.5,
                                       union_loss_weight=1.0, intersection_loss_weight=1.0,
                                       prob_threshold=0.5,
                                       use_skeleton_for_intersection=False):
    """
    Three-view mutual learning:
    1. Train the student.
    2. Read corresponding probabilities from two teacher caches.
    3. Compute the union loss from the student and two teachers.
    teacher_models: dict, {'axial': model_axial, 'coronal': model_coronal, 'sagittal': model_sagittal}
    teacher_loaders: dict, {'axial': train_loader_axial, ...}
    """
    model_student.train()

    # Choose the two teacher views other than the student view.
    all_views = ['axial', 'coronal', 'sagittal']
    teacher_view1, teacher_view2 = [v for v in all_views if v != student_view]

    # Teacher 1: infer and convert to student_view.
    teacher_probs1_cache = {}
    teacher_models[teacher_view1].eval()
    with torch.no_grad():
        probs1 = build_teacher_prob_volume(
            teacher_models[teacher_view1], teacher_loaders[teacher_view1], device, teacher_view=teacher_view1
        )
    for vol_idx in probs1:
        teacher_probs1_cache[vol_idx] = convert_view_probabilities(probs1[vol_idx], student_view=student_view)

    # Teacher 2: infer and convert to student_view.
    teacher_probs2_cache = {}
    teacher_models[teacher_view2].eval()
    with torch.no_grad():
        probs2 = build_teacher_prob_volume(
            teacher_models[teacher_view2], teacher_loaders[teacher_view2], device, teacher_view=teacher_view2
        )
    for vol_idx in probs2:
        teacher_probs2_cache[vol_idx] = convert_view_probabilities(probs2[vol_idx], student_view=student_view)

    total_student_loss = 0.0
    total_mutual_kd_loss = 0.0
    num_batches = 0

    for b_idx, batch in enumerate(train_loader_student):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        image_indices = batch.get('image_index', None)
        slice_idxs = batch.get('slice_idx', None)

        optimizer_student.zero_grad()

        # Student forward pass (UNet2D returns one output).
        pred = model_student(images)
        student_loss = structure_loss(pred, labels)

        # Mutual-learning loss.
        batch_mutual_loss = 0.0
        batch_size = images.shape[0]

        for i in range(batch_size):
            img_idx = int(image_indices[i].item()) if image_indices is not None else 0
            sl_idx = int(slice_idxs[i].item()) if slice_idxs is not None else i

            # Apply sigmoid to the student prediction.
            student_prob = torch.sigmoid(pred[i, 0])  # (H, W)
            student_bin = (student_prob > prob_threshold)  # Boolean tensor.

            if img_idx in teacher_probs1_cache and img_idx in teacher_probs2_cache:
                teacher_vol1 = teacher_probs1_cache[img_idx]
                teacher_vol2 = teacher_probs2_cache[img_idx]
                teacher_prob_slice1 = teacher_vol1[:, :, sl_idx] if sl_idx < teacher_vol1.shape[2] else None
                teacher_prob_slice2 = teacher_vol2[:, :, sl_idx] if sl_idx < teacher_vol2.shape[2] else None
                if teacher_prob_slice1 is not None and teacher_prob_slice2 is not None:
                    teacher_prob1 = torch.from_numpy(teacher_prob_slice1).float().to(device)
                    teacher_prob2 = torch.from_numpy(teacher_prob_slice2).float().to(device)
                    teacher_bin1 = (teacher_prob1 > prob_threshold)
                    teacher_bin2 = (teacher_prob2 > prob_threshold)

                    # Build the union label from foreground predicted by any model.
                    union_label = (student_bin | teacher_bin1 | teacher_bin2).float()  # (H, W)

                    # Compute union loss.
                    union_loss = structure_loss(
                        pred[i, 0].unsqueeze(0),  # (1, H, W)
                        union_label.unsqueeze(0)  # (1, H, W)
                    )

                    # Build the intersection label from foreground predicted by all three models.
                    intersection_label = (student_bin & teacher_bin1 & teacher_bin2).float()  # (H, W)

                    # Skeletonize with differentiable soft_skeletonize_2d.
                    if use_skeleton_for_intersection:
                        student_prob_4d = student_prob.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        intersection_label_4d = intersection_label.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        intersection_loss = clDice_loss_2d(student_prob_4d, intersection_label_4d)
                    else:
                        intersection_loss = structure_loss(
                            pred[i, 0].unsqueeze(0),  # (1, H, W)
                            intersection_label.unsqueeze(0)  # (1, H, W)
                        )

                    # Combine the two weighted losses.
                    batch_mutual_loss += union_loss_weight * union_loss + intersection_loss_weight * intersection_loss

        # Average mutual-learning loss.
        batch_mutual_loss /= batch_size

        # Total loss.
        total_loss = student_loss + kl_weight * batch_mutual_loss

        total_loss.backward()
        optimizer_student.step()

        total_student_loss += student_loss.item()
        total_mutual_kd_loss += batch_mutual_loss.item()
        num_batches += 1

        if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(train_loader_student):
            avg_so_far = total_student_loss / (b_idx + 1)
            print(f"  Iter {b_idx + 1}/{len(train_loader_student)} - "
                  f"student_loss: {student_loss.item():.6f}, "
                  f"mutual_kd: {batch_mutual_loss.item():.9f}, "
                  f"avg_student_loss: {avg_so_far:.6f}")

    return {
        'student_loss': total_student_loss / max(1, num_batches),
        'mutual_kd_loss': total_mutual_kd_loss / max(1, num_batches)
    }




def validate_and_compute_metrics(model, val_loader, val_dataset, view, epoch, val_image_dir, device):
    """
    Validate and compute metrics:
    - Compute 2D validation loss (BCE + Dice).
    - Collect predictions for 3D reconstruction.
    - Save validation images every five epochs.
    - Compute 3D ClDice and 3D Dice.
    Return: (mean_val_loss, epoch_3d_cldice, epoch_3d_dice).
    """
    model.eval()
    with torch.no_grad():
        pred_slices = {}
        total_val_loss = 0.0
        val_batch_count = 0

        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_indices = batch.get('image_index', None)
            slice_idxs = batch.get('slice_idx', None)

            # UNet2D returns a single output.
            pred = model(images)
            preds_sig = torch.sigmoid(pred).detach().cpu()

            # Compute 2D validation loss.
            val_loss = structure_loss(pred, labels)
            total_val_loss += val_loss.item()
            val_batch_count += 1

            # Collect predicted slices for 3D reconstruction.
            for i in range(preds_sig.shape[0]):
                img_idx = int(image_indices[i].item()) if image_indices is not None else 0
                sl_idx = int(slice_idxs[i].item()) if slice_idxs is not None else i

                if img_idx not in pred_slices:
                    pred_slices[img_idx] = []
                pred_slices[img_idx].append((sl_idx, preds_sig[i].squeeze(0).numpy()))

        # Compute mean validation loss.
        mean_val_loss = total_val_loss / max(1, val_batch_count)

        # Save validation predictions as TIFF at the configured intervals.
        if epoch == 1 or epoch % 10 == 0:
            for img_idx in sorted(pred_slices.keys()):
                preds_list = pred_slices[img_idx]
                cldice_res, pred_stack, _ = reconstruct_and_eval(preds_list, val_dataset, img_idx, view, device)
                out_filename = os.path.join(val_image_dir, f"Val_volume_view_{view}_img_{img_idx}_epoch{epoch}.tiff")
                tiff.imwrite(out_filename, (pred_stack * 255).astype(np.uint8))

        # Compute 3D ClDice and 3D Dice.
        total_3d_cldice = 0.0
        total_3d_dice = 0.0
        cldice_count = 0
        dice_count = 0
        for img_idx in sorted(pred_slices.keys()):
            preds_list = pred_slices[img_idx]
            cldice_res, _, label_3d = reconstruct_and_eval(preds_list, val_dataset, img_idx, view, device)
            total_3d_dice += float(cldice_res['dice'])
            dice_count += 1
            if not np.any(label_3d > 0):
                continue
            total_3d_cldice += float(cldice_res['cldice'])
            cldice_count += 1
        epoch_3d_cldice = total_3d_cldice / max(1, cldice_count)
        epoch_3d_dice = total_3d_dice / max(1, dice_count)

        print(
            f"[Epoch {epoch}] Val Loss ({view}): {mean_val_loss:.4f}, "
            f"3D ClDice: {epoch_3d_cldice:.4f}, 3D Dice: {epoch_3d_dice:.4f}"
        )

    return mean_val_loss, epoch_3d_cldice, epoch_3d_dice


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                   val_losses, train_losses, val_cldices, val_dices, val_cldice_epochs,
                   mutual_losses, best_3d_cldice,
                   save_path, view, suffix):
    """Save a checkpoint."""
    ckpt = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'view': view,
        'best_val_loss': best_val_loss,
        'best_3d_cldice': best_3d_cldice,
        'val_losses': val_losses,
        'train_losses': train_losses,
        'val_cldices': val_cldices,
        'val_dices': val_dices,
        'val_cldice_epochs': val_cldice_epochs,
        'mutual_losses': mutual_losses,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    ckpt_path = os.path.join(save_path, f"UNet2D-{view}-{suffix}.pth")
    torch.save(ckpt, ckpt_path)


def evaluate_on_test(model, test_loader, test_dataset, view, device, output_path, save_predictions=True):
    """
    Evaluate on the test set and return all metrics.
    """
    model.eval()
    test_pred_slices = {}
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            image_indices = batch.get('image_index', None)
            slice_idxs = batch.get('slice_idx', None)

            pred = model(images)
            preds_sig = torch.sigmoid(pred).detach().cpu()

            # Collect predicted slices for 3D reconstruction.
            for i in range(preds_sig.shape[0]):
                img_idx = int(image_indices[i].item()) if image_indices is not None else 0
                sl_idx = int(slice_idxs[i].item()) if slice_idxs is not None else i

                if img_idx not in test_pred_slices:
                    test_pred_slices[img_idx] = []
                test_pred_slices[img_idx].append((sl_idx, preds_sig[i].squeeze(0).numpy()))

    # Compute 3D metrics.
    all_metrics = []
    for img_idx in sorted(test_pred_slices.keys()):
        preds_list = test_pred_slices[img_idx]
        cldice_res, pred_stack, label_3d = reconstruct_and_eval(preds_list, test_dataset, img_idx, view, device)
        all_metrics.append({
            'image_index': img_idx,
            'cldice': float(cldice_res['cldice']),
            'dice': float(cldice_res['dice']),
            'iou': float(cldice_res['iou']),
            't_iou': float(cldice_res['t_iou']),
            'clprecision': float(cldice_res['clprecision']),
            'clrecall': float(cldice_res['clrecall']),
            'precision': float(cldice_res['precision']),
            'recall': float(cldice_res['recall']),
            'acc': float(cldice_res['acc'])
        })
        
        # Save test predictions when requested.
        if save_predictions:
            test_image_dir = os.path.join(output_path, "test_image")
            os.makedirs(test_image_dir, exist_ok=True)
            out_filename = os.path.join(test_image_dir, f"Test_volume_view_{view}_img_{img_idx}.tiff")
            tiff.imwrite(out_filename, (pred_stack * 255).astype(np.uint8))

    return all_metrics


def save_test_metrics(metrics, output_path, view, epoch, best_3d_cldice):
    """Save test metrics to Excel and TXT files."""
    
    # ========== Save an Excel file ==========
    excel_path = os.path.join(output_path, f"test_metrics_view_{view}.xlsx")
    
    # Prepare per-sample metrics.
    df_details = pd.DataFrame([
        {
            'Idx': m['image_index'],
            'IOU': m['iou'],
            'T-IOU': m['t_iou'],
            'ClDice': m['cldice'],
            'ClPrecision': m['clprecision'],
            'ClRecall': m['clrecall'],
            'Dice': m['dice'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'Acc': m['acc']
        }
        for m in metrics
    ])
    
    # Compute averages.
    avg_iou = np.mean([m['iou'] for m in metrics])
    avg_t_iou = np.mean([m['t_iou'] for m in metrics])
    avg_cldice = np.mean([m['cldice'] for m in metrics])
    avg_clprecision = np.mean([m['clprecision'] for m in metrics])
    avg_clrecall = np.mean([m['clrecall'] for m in metrics])
    avg_dice = np.mean([m['dice'] for m in metrics])
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    avg_acc = np.mean([m['acc'] for m in metrics])
    
    # Add the average row.
    df_avg = pd.DataFrame([{
        'Idx': 'AVG',
        'IOU': avg_iou,
        'T-IOU': avg_t_iou,
        'ClDice': avg_cldice,
        'ClPrecision': avg_clprecision,
        'ClRecall': avg_clrecall,
        'Dice': avg_dice,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'Acc': avg_acc
    }])
    
    # Combine individual metrics and averages.
    df_all = pd.concat([df_details, df_avg], ignore_index=True)
    
    # Create an Excel writer.
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Write individual metrics and averages.
        df_all.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Create summary information.
        df_summary = pd.DataFrame({
            'Metric': ['View', 'Best Epoch', 'Best Val 3D ClDice', 'Evaluation Time',
                       'Avg IOU', 'Avg T-IOU', 'Avg ClDice', 'Avg ClPrecision', 
                       'Avg ClRecall', 'Avg Dice', 'Avg Precision', 'Avg Recall', 'Avg Acc'],
            'Value': [view, epoch, f'{best_3d_cldice:.5f}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      f'{avg_iou:.5f}', f'{avg_t_iou:.5f}', f'{avg_cldice:.5f}', f'{avg_clprecision:.5f}',
                      f'{avg_clrecall:.5f}', f'{avg_dice:.5f}', f'{avg_precision:.5f}', f'{avg_recall:.5f}', f'{avg_acc:.5f}']
        })
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Set column widths.
        worksheet = writer.sheets['Metrics']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Saved test metrics to Excel: {excel_path}")
    
    # ========== Also save a TXT file for compatibility ==========
    txt_path = os.path.join(output_path, f"test_metrics_view_{view}.txt")
    
    with open(txt_path, 'w') as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"Test Set Evaluation Results\n")
        f.write(f"View: {view}\n")
        f.write(f"Best Epoch: {epoch}\n")
        f.write(f"Best Val 3D ClDice: {best_3d_cldice:.4f}\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        # Write metrics for each sample.
        f.write("Individual Volume Metrics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Idx':<6} {'IOU':<10} {'T-IOU':<10} {'ClDice':<10} {'ClPrecision':<10} {'ClRecall':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10} {'Acc':<10}\n")
        f.write("-" * 100 + "\n")
        
        for m in metrics:
            f.write(f"{m['image_index']:<6} {m['iou']:<10.5f} {m['t_iou']:<10.5f} {m['cldice']:<10.5f} {m['clprecision']:<10.5f} {m['clrecall']:<10.5f} {m['dice']:<10.5f} {m['precision']:<10.5f} {m['recall']:<10.5f} {m['acc']:<10.5f}\n")
        
        # Compute averages.
        f.write("-" * 100 + "\n")
        f.write(f"{'AVG':<6} {avg_iou:<10.5f} {avg_t_iou:<10.5f} {avg_cldice:<10.5f} {avg_clprecision:<10.5f} {avg_clrecall:<10.5f} {avg_dice:<10.5f} {avg_precision:<10.5f} {avg_recall:<10.5f} {avg_acc:<10.5f}\n")
        f.write("=" * 100 + "\n")
        
    print(f"Saved test metrics to txt: {txt_path}")
    
    return avg_cldice, avg_dice, avg_iou, avg_t_iou, avg_clprecision, avg_clrecall, avg_precision, avg_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mutual Learning Training for Three-View Segmentation (UNet2D)")

    # Data paths (shared across views; the dataset handles view-specific slicing).
    parser.add_argument("--train_image_path", type=str, default="/path/to/train/image")
    parser.add_argument("--train_mask_path", type=str, default="/path/to/train/label")
    parser.add_argument("--val_image_path", type=str, default="/path/to/val/image")
    parser.add_argument("--val_mask_path", type=str, default="/path/to/val/label")
    parser.add_argument("--test_image_path", type=str, default="/path/to/test/image")
    parser.add_argument("--test_mask_path", type=str, default="/path/to/test/label")

    # Output paths.
    parser.add_argument("--save_path", type=str, default="/path/to/output/model")
    parser.add_argument("--output_path", type=str, default="/path/to/output")

    # Training parameters.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible training")

    # Mutual-learning parameters.
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs for each view")
    parser.add_argument("--kl_weight", type=float, default=0.5, help="Weight for KL divergence loss")
    parser.add_argument("--union_loss_weight", type=float, default=1.0, help="Weight for union loss")
    parser.add_argument("--intersection_loss_weight", type=float, default=1.0, help="Weight for intersection loss")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Probability threshold used to binarize student and teacher predictions for mutual learning")
    parser.add_argument("--rotation_interval", type=int, default=1, help="Number of consecutive epochs for each student view before rotating to the next view")
    parser.add_argument("--use_skeleton_for_intersection", action="store_true", help="Use skeletonized intersection label for intersection loss")

    args = parser.parse_args()
    main(args)

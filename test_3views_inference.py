import argparse
import inspect
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import DataLoader

from dataset import FullDataset_160
from model import UNet2D
from eval_volumes import eval_two_volumes_maxpool


# ==================== 1. Random seeds (same as training) ====================

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


# ==================== 2. Slice reconstruction and metrics (from training) ====================

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


# ==================== 3. Test-set inference (from training) ====================

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


# ==================== 4. Save TXT and Excel metrics (from training) ====================

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


# ==================== 5. Validate input data ====================

def validate_dataset(dataset):
    # FullDataset_160 assumes 160 slices and recursively skips samples on read errors.
    # Check inputs early to avoid silently skipping data or mismatching label and prediction shapes.
    if not dataset.images:
        raise ValueError("测试图像目录中没有 .tif / .tiff 文件")
    for image_path, label_path in zip(dataset.images, dataset.gts):
        for path in (image_path, label_path):
            array = tiff.imread(path)
            if array.shape != (160, 160, 160):
                raise ValueError(f"原训练测试流程要求 160x160x160，{path}: {array.shape}")


# ==================== 6. Main flow: data -> model -> inference -> metrics ====================

def main(args):
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size 必须 >= 1，num_workers 必须 >= 0")
    if args.image_size != 160:
        raise ValueError("原训练测试流程使用 FullDataset_160，image_size 必须为 160")

    set_random_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}; threshold=0.5; pool_kernel=3")

    for view in args.views:
        # Step 1: Load the target data with the training preprocessing and DataLoader settings.
        test_dataset = FullDataset_160(
            image_root=args.test_image_path,
            gt_root=args.test_mask_path,
            size=args.image_size,
            mode="val",
            view=view,
        )
        validate_dataset(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed),
        )

        # Step 2: Load the checkpoint for the requested view.
        # Prefer three-view checkpoint names, then single-view names; preserve best/latest.
        checkpoint_path = os.path.join(args.model_path, f"UNet2D-{view}-{args.checkpoint}.pth")
        single_view_path = os.path.join(args.model_path, f"UNet2D-view-{view}-{args.checkpoint}.pth")
        if not os.path.isfile(checkpoint_path):
            if os.path.isfile(single_view_path):
                checkpoint_path = single_view_path
            else:
                raise FileNotFoundError(
                    f"找不到模型权重，请检查以下两个路径：\n{checkpoint_path}\n{single_view_path}"
                )
        # Newer PyTorch needs weights_only=False for checkpoints containing NumPy metrics.
        # Support the older PyTorch used in training; load trusted checkpoints only.
        if "weights_only" in inspect.signature(torch.load).parameters:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("view", view) != view:
            raise ValueError(f"检查点视图 {checkpoint['view']} 与请求的 {view} 不一致")

        model = UNet2D(in_ch=3, num_classes=1).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        epoch = checkpoint.get("epoch", "unknown")
        best_3d_cldice = float(checkpoint.get("best_3d_cldice", float("nan")))
        del checkpoint
        print(f"[{view}] {checkpoint_path} (epoch={epoch})", flush=True)

        # Step 3: Run the training-time test pipeline and compute nine metrics per volume.
        metrics = evaluate_on_test(
            model, test_loader, test_dataset, view, device,
            args.output_dir, save_predictions=True,
        )

        # Step 4: Save per-volume metrics and the test-set mean.
        save_test_metrics(metrics, args.output_dir, view, epoch, best_3d_cldice)
        mean_cldice = np.mean([m["cldice"] for m in metrics])
        print(f"[{view}] 测试体积数={len(metrics)}, Mean ClDice={mean_cldice:.8f}")
        del model, test_loader, test_dataset

    print(f"推理及指标计算完成：{args.output_dir}")


# ==================== 7. Command-line arguments ====================

def parse_args():
    parser = argparse.ArgumentParser(description="UNet2D 推理及指标计算：固定 threshold=0.5，pool_kernel=3")
    parser.add_argument("--model_path", type=str, required=True, help="源模型的 model 目录")
    parser.add_argument("--test_image_path", type=str, required=True, help="目标测试图像目录")
    parser.add_argument("--test_mask_path", type=str, required=True, help="目标测试标签目录")
    parser.add_argument("--output_dir", "--output_path", dest="output_dir", type=str, required=True)
    parser.add_argument("--val_batch_size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views", nargs="+", choices=["axial", "coronal", "sagittal"], default=["axial"])
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

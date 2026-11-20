import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from model import UNet2D
from ptflops import get_model_complexity_info
from train_2d_view_mutual_unet_union_inter_with_test_3views_warmupval_optimized import train_with_mutual_learning_3views


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("UNet2D Computation Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (160, 3, 160, 160)
    batch_size = 24

    print(f"\nDevice: {device}")
    print(f"Input size: {input_size}")
    print(f"Slice batch size: {batch_size}")

    # Create the model
    model = UNet2D(in_ch=3, num_classes=1).to(device)
    model.eval()

    # ========== 1. Parameter count ==========
    print("-" * 60)
    print("1. Parameter Count")
    print("-" * 60)
    total_params, trainable_params = count_parameters(model)
    print(f"  Params updated per step: {trainable_params:,}")
    print(f"  Total (one model):       {total_params:,}")
    print(f"  Size:      {total_params * 4 / 1024**2:.2f} MB")

    # ========== 2. FLOPs ==========
    print("\n" + "-" * 60)
    print("2. FLOPs")
    print("-" * 60)

    # ptflops returns MACs for one slice; a complete volume contains 160 slices.
    macs, params = get_model_complexity_info(model, (3, 160, 160), as_strings=False, verbose=False)
    flops = macs * 2 * 160
    print(f"  FLOPs (one 160^3 volume): {flops / 1e9:.2f} G")
    print("  Using ptflops to count FLOPs")

    # ========== 3. Training GPU memory ==========
    train_mem = None
    print("\n" + "-" * 60)
    print("3. Training Memory (with gradients)")
    print("-" * 60)

    if device.type == 'cuda':
        del model  # Release the parameter-counting model before measuring training memory.
        torch.cuda.empty_cache()

        model_axial = UNet2D(in_ch=3, num_classes=1).to(device)
        model_coronal = UNet2D(in_ch=3, num_classes=1).to(device)
        model_sagittal = UNet2D(in_ch=3, num_classes=1).to(device)
        models = {'axial': model_axial, 'coronal': model_coronal, 'sagittal': model_sagittal}

        # As in training, each step updates one student while two teachers run without gradients.
        optimizer = torch.optim.AdamW(model_axial.parameters(), lr=0.0003, weight_decay=1e-3)

        # Build the same 160^3 volume in three views; each loader batches 24 slices.
        volume = torch.randn(160, 160, 160)
        labels = (volume > 2).float()
        train_loaders = {}
        for view, axis in (('axial', 0), ('coronal', 1), ('sagittal', 2)):
            samples = []
            for i in range(160):
                neighbors = (max(i - 1, 0), i, min(i + 1, 159))
                samples.append({
                    'image': torch.stack([volume.select(axis, j) for j in neighbors]),
                    'label': labels.select(axis, i).unsqueeze(0),
                    'image_index': 0,
                    'slice_idx': i,
                })
            train_loaders[view] = DataLoader(samples, batch_size=batch_size, shuffle=False)

        torch.cuda.reset_peak_memory_stats()
        train_with_mutual_learning_3views(
            model_student=model_axial,
            optimizer_student=optimizer,
            train_loader_student=train_loaders['axial'],
            teacher_models=models,
            teacher_loaders=train_loaders,
            device=device,
            student_view='axial',
            kl_weight=0.5,
            union_loss_weight=1.0,
            intersection_loss_weight=1.0,
            prob_threshold=0.5,
            use_skeleton_for_intersection=True,
        )
        torch.cuda.synchronize()
        train_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak: {train_mem:.2f} GiB")
        print("  (1 student + 2 teachers; full mutual-learning procedure)")

        del model_axial, model_coronal, model_sagittal, models, optimizer
        del volume, labels, train_loaders, samples
        torch.cuda.empty_cache()
    else:
        print("  CUDA not available")

    # ========== 4. Inference GPU memory ==========
    infer_mem = None
    print("\n" + "-" * 60)
    print("4. Inference Memory (no gradients)")
    print("-" * 60)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        model = UNet2D(in_ch=3, num_classes=1).to(device)
        model.eval()
        dummy_input = torch.randn(input_size)  # Keep the complete volume on CPU and send batches to the GPU.

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                pred_slices = []
                for batch in dummy_input.split(batch_size):
                    pred_slices.append(torch.sigmoid(model(batch.to(device))).cpu())
                pred_stack = torch.cat(pred_slices, dim=0)

        torch.cuda.synchronize()
        infer_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak: {infer_mem:.2f} GiB")

        del model, dummy_input, pred_slices, pred_stack
        torch.cuda.empty_cache()
    else:
        print("  CUDA not available")

    # ========== 5. Single-volume inference time ==========
    print("\n" + "-" * 60)
    print("5. Single Volume Inference Time")
    print("-" * 60)

    model = UNet2D(in_ch=3, num_classes=1).to(device)
    model.eval()
    dummy_input = torch.randn(input_size)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            pred_slices = []
            for batch in dummy_input.split(batch_size):
                pred_slices.append(torch.sigmoid(model(batch.to(device))).cpu())
            pred_stack = torch.cat(pred_slices, dim=0)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure one complete 160^3 volume per iteration
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            pred_slices = []
            for batch in dummy_input.split(batch_size):
                pred_slices.append(torch.sigmoid(model(batch.to(device))).cpu())
            pred_stack = torch.cat(pred_slices, dim=0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms

    print("  One 160^3 volume:")
    print(f"    Mean: {times.mean():.4f} ms")
    print(f"    Std:  {times.std():.4f} ms")
    print(f"    Min:  {times.min():.4f} ms")
    print(f"    Max:  {times.max():.4f} ms")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Params per step':<25}: {trainable_params:,}")
    print(f"{'FLOPs per volume':<25}: {flops / 1e9:.2f} G")
    print(f"{'Training Memory':<25}: {train_mem:.2f} GiB" if train_mem is not None else "Training Memory: N/A")
    print(f"{'Inference Memory':<25}: {infer_mem:.2f} GiB" if infer_mem is not None else "Inference Memory: N/A")
    print(f"{'Inference Time (mean)':<25}: {times.mean():.2f} ms")
    print("=" * 60)

    print("\nDone!")
    print("=" * 60)


if __name__ == "__main__":
    main()

import os
import sys
import re
import time
import json
import argparse
import resource
import numpy as np
import torch
import tifffile as tiff

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from UNet import UNet2D


def find_model_checkpoint(model_dir, view):
    best_path = os.path.join(model_dir, f"UNet2D-{view}-best.pth")
    latest_path = os.path.join(model_dir, f"UNet2D-{view}-latest.pth")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(latest_path):
        return latest_path
    raise FileNotFoundError(f"找不到 {view} 模型权重：{model_dir}")


def sliding_window_starts(length, window_size, overlap):
    """Move the last window to cover the boundary; pad axes shorter than the window."""
    length = max(length, window_size)
    starts = list(range(0, length - window_size + 1, window_size - overlap))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return starts


def convert_3d_patch_to_8bit(patch3d, low=0.05, high=99.95):
    """Normalize each 3D cube by percentiles to floating-point values in [0, 1]."""
    patch3d = patch3d.astype(np.float32)
    p_low = np.percentile(patch3d, low)
    p_high = np.percentile(patch3d, high)
    img_clipped = np.clip(patch3d, p_low, p_high)
    return (img_clipped - p_low) / (p_high - p_low + 1e-8)


def load_whole_brain(input_dir, cache_path):
    """Reuse an existing raw cache, or decompress TIFF slices to create one."""
    filenames = [name for name in os.listdir(input_dir)
                 if name.lower().endswith((".tif", ".tiff"))]
    filenames.sort(key=lambda name: [int(s) if s.isdigit() else s.lower()
                                     for s in re.split(r"(\d+)", name)])
    if not filenames:
        raise ValueError(f"没有找到 TIFF 切片：{input_dir}")

    # OME-TIFF series[0] may span files as one ZYX volume; use page for a single slice.
    with tiff.TiffFile(os.path.join(input_dir, filenames[0])) as tif:
        page = tif.pages[0]
        plane_shape = page.shape
        dtype = np.dtype(page.dtype)
        if len(tif.pages) != 1 or len(plane_shape) != 2 or page.samplesperpixel != 1:
            raise ValueError(f"每个 TIFF 必须只有一张二维灰度切片：{filenames[0]}")
    shape = (len(filenames), plane_shape[0], plane_shape[1])
    print(f"Whole brain shape (Z, Y, X): {shape}, dtype: {dtype}", flush=True)
    expected_bytes = int(np.prod(shape)) * dtype.itemsize

    if os.path.exists(cache_path):
        if os.path.getsize(cache_path) != expected_bytes:
            raise ValueError(f"缓存大小与输入不匹配，请检查缓存或更换 cache_dir：{cache_path}")
        print(f"Reuse cache: {cache_path}（跳过 TIFF 解压和缓存写入）", flush=True)
        # Raw caches have no provenance; matching size does not guarantee matching contents.
        return shape, dtype

    # Write a new cache as .partial and rename it to .raw only when complete.
    partial_path = cache_path + ".partial"
    if os.path.exists(partial_path):
        raise FileExistsError(f"存在未完成的缓存，请检查或更换 cache_dir：{partial_path}")
    with open(partial_path, "xb") as f:
        f.truncate(expected_bytes)

    for z, name in enumerate(filenames):
        with tiff.TiffFile(os.path.join(input_dir, name)) as tif:
            page = tif.pages[0]
            if len(tif.pages) != 1 or len(page.shape) != 2 or page.samplesperpixel != 1:
                raise ValueError(f"每个 TIFF 必须只有一张二维灰度切片：{name}")
            if page.shape != shape[1:] or page.dtype != dtype:
                raise ValueError(f"切片尺寸或类型不一致：{name}")
            image = page.asarray(maxworkers=1)
        volume = np.memmap(partial_path, dtype=dtype, mode="r+", shape=shape)
        volume[z] = image
        volume.flush()
        # Close each mapping so previously read volumes do not remain in process memory.
        volume._mmap.close()
        del image
        if (z + 1) % 100 == 0 or z + 1 == shape[0]:
            print(f"Loaded slices: {z + 1}/{shape[0]}", flush=True)
    with open(partial_path, "rb") as f:
        os.fsync(f.fileno())
    os.rename(partial_path, cache_path)
    return shape, dtype


def inference_cube(model, cube, view, device, patch_size, overlap, batch_size):
    """Infer one cube and return prediction sums and counts for averaging overlaps."""
    if view == "coronal":
        cube = cube.transpose(1, 0, 2)
    elif view == "sagittal":
        cube = cube.transpose(1, 2, 0)  # Keep the sagittal transpose used by the original script.

    cube_norm = convert_3d_patch_to_8bit(cube)
    pred_cube = np.zeros(cube.shape, dtype=np.float32)
    count_cube = np.zeros(cube.shape, dtype=np.float32)
    y_starts = sliding_window_starts(cube.shape[1], patch_size, overlap)
    x_starts = sliding_window_starts(cube.shape[2], patch_size, overlap)
    # After transposition, coronal axis 0 and sagittal axis 1 correspond to the original Z axis.
    # The original Z axis is concatenated without overlap even when patch_size is smaller than cube_size.
    if view == "coronal":
        y_starts = range(0, cube.shape[1], patch_size)
    elif view == "sagittal":
        x_starts = range(0, cube.shape[2], patch_size)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    with torch.no_grad():
        for sy in y_starts:
            for sx in x_starts:
                for z in range(0, cube.shape[0], batch_size):
                    patches = cube_norm[z:z+batch_size, sy:sy+patch_size, sx:sx+patch_size]
                    height, width = patches.shape[1:]
                    # Pad an incomplete patch at the Z end and crop away the padding after prediction.
                    if height < patch_size or width < patch_size:
                        patches = np.pad(patches, ((0, 0), (0, patch_size-height),
                                                   (0, patch_size-width)), mode="edge")
                    patches = np.repeat(patches[:, None], 3, axis=1)
                    patches = (patches - mean) / std
                    batch_tensor = torch.from_numpy(patches).float().to(device)
                    batch_pred = torch.sigmoid(model(batch_tensor)).cpu().numpy()[:, 0]
                    batch_pred = batch_pred[:, :height, :width]
                    pred_cube[z:z+batch_size, sy:sy+patch_size, sx:sx+patch_size] += batch_pred
                    count_cube[z:z+batch_size, sy:sy+patch_size, sx:sx+patch_size] += 1

    if view == "coronal":
        return pred_cube.transpose(1, 0, 2), count_cube.transpose(1, 0, 2)
    if view == "sagittal":
        return pred_cube.transpose(2, 0, 1), count_cube.transpose(2, 0, 1)
    return pred_cube, count_cube


def inference_block(model, view, cache_path, shape, dtype, device, args, z0, y0, x0):
    """Compute one output block, keeping only its predictions and counts in memory."""
    D, H, W = shape
    z1 = min(z0 + args.block_size, D)
    y1 = min(y0 + args.block_size, H)
    x1 = min(x0 + args.block_size, W)
    pred_block = np.zeros((z1-z0, y1-y0, x1-x0), dtype=np.float32)
    count_block = np.zeros_like(pred_block)
    size = args.cube_size

    # Compute window positions over the whole brain and select those intersecting this output block.
    # Step by cube_size along Z; retain overlap and edge-aligned windows in X and Y.
    z_starts = [z for z in range(0, D, size) if z < z1 and z+size > z0]
    y_starts = [y for y in sliding_window_starts(H, size, args.overlap) if y < y1 and y+size > y0]
    x_starts = [x for x in sliding_window_starts(W, size, args.overlap) if x < x1 and x+size > x0]

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                volume = np.memmap(cache_path, dtype=dtype, mode="r", shape=shape)
                cube = volume[z:z+size, y:y+size, x:x+size].copy()
                volume._mmap.close()
                # Pad the last Z segment or any short axis and write back only the original extent.
                cube = np.pad(cube, ((0, size-cube.shape[0]),
                                     (0, size-cube.shape[1]), (0, size-cube.shape[2])), mode="edge")
                pred_cube, count_cube = inference_cube(
                    model, cube, view, device, args.patch_size, args.overlap, args.batch_size)

                # Keep only the intersection of the cube and current output block.
                za, zb = max(z, z0), min(z+size, z1)
                ya, yb = max(y, y0), min(y+size, y1)
                xa, xb = max(x, x0), min(x+size, x1)
                pred_block[za-z0:zb-z0, ya-y0:yb-y0, xa-x0:xb-x0] += pred_cube[za-z:zb-z, ya-y:yb-y, xa-x:xb-x]
                count_block[za-z0:zb-z0, ya-y0:yb-y0, xa-x0:xb-x0] += count_cube[za-z:zb-z, ya-y:yb-y, xa-x:xb-x]
                del cube, pred_cube, count_cube

    pred_block /= count_block
    pred_block *= 255
    return pred_block.astype(np.uint8)


def inference_whole_brain(model, view, cache_path, shape, dtype, device, args):
    """Process output blocks in Z, Y, X order and write each to the BigTIFF."""
    output_path = os.path.join(args.output_dir, f"{args.output_prefix}_{view}.tiff")
    partial_path = output_path + ".partial"
    if os.path.exists(output_path) or os.path.exists(partial_path):
        raise FileExistsError(f"输出已存在，请更换 output_dir：{output_path}")
    output = tiff.memmap(partial_path, shape=shape, dtype=np.uint8,
                         bigtiff=True, photometric="minisblack", metadata={"axes": "ZYX"})
    offset = output.offset
    output._mmap.close()

    D, H, W = shape
    block_size = args.block_size
    for z0 in range(0, D, block_size):
        for y0 in range(0, H, block_size):
            for x0 in range(0, W, block_size):
                pred = inference_block(model, view, cache_path, shape, dtype, device, args, z0, y0, x0)
                dz, dy, dx = pred.shape
                output = np.memmap(partial_path, dtype=np.uint8, mode="r+", shape=shape, offset=offset)
                output[z0:z0+dz, y0:y0+dy, x0:x0+dx] = pred
                output.flush()
                output._mmap.close()
                del pred
                print(f"{view}: finished block (z={z0}, y={y0}, x={x0})", flush=True)

    with open(partial_path, "rb") as f:
        os.fsync(f.fileno())  # Flush output before recording end-to-end time.
    os.rename(partial_path, output_path)
    print(f"Saved: {output_path}", flush=True)


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "whole_brain_input.raw")

    # 1. Find the checkpoint for the selected view only.
    checkpoint_path = find_model_checkpoint(args.model_path, args.view)

    # 2. Reuse a raw cache or build one from TIFF files; exclude this from wall-clock time.
    shape, dtype = load_whole_brain(args.input_dir, cache_path)

    # 3. Start timing after cache preparation and before model loading.
    # Include model loading, cache reads, preprocessing, inference, and output writes.
    print(f"\nLoading {args.view} model...", flush=True)
    start_time = time.perf_counter()
    model = UNet2D(in_ch=3, num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model = model.to(device)
    model.eval()

    # Read blocks from the prepared cache and infer the selected view.
    print(f"\nRunning {args.view} inference...", flush=True)
    inference_whole_brain(model, args.view, cache_path, shape, dtype, device, args)
    del model

    # 4. Synchronize GPU and disk writes, then record resource usage.
    # os.remove(cache_path)
    peak_gpu_allocated = None
    peak_gpu_reserved = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_gpu_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    wall_clock_time = time.perf_counter() - start_time
    # On Linux, ru_maxrss is in KiB and tracks peak RSS over the full process lifetime.
    peak_cpu_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    metrics = {
        "wall_clock_seconds": wall_clock_time,
        "peak_gpu_memory_allocated_gib": peak_gpu_allocated,
        "peak_gpu_memory_reserved_gib": peak_gpu_reserved,
        "peak_cpu_memory_rss_gib": peak_cpu_memory,
        "volume_shape_zyx": list(shape),
        "cube_overlap_zyx": [0, args.overlap, args.overlap],
        "device": str(device),
        "configuration": vars(args),
        "note": "耗时从缓存准备完成后的模型初始化/权重加载前开始，包含推理时读取缓存、预处理、推理和结果写盘，"
                "结束前同步 GPU；不包含原始 TIFF 读取、解压、缓存生成及前置准备；"
                "CPU 为进程整个生命周期的峰值 RSS，仍包含缓存准备阶段；"
                "GPU 为 PyTorch allocated/reserved 峰值，不包含 CUDA 上下文等额外占用。",
    }
    metrics_path = os.path.join(args.output_dir, "inference_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\nWall-clock time: {wall_clock_time:.2f} s")
    print(f"Peak GPU memory (allocated / reserved): {peak_gpu_allocated} / {peak_gpu_reserved} GiB")
    print(f"Peak CPU memory (RSS): {peak_cpu_memory:.3f} GiB")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whole Brain Single-View UNet2D Inference")
    parser.add_argument("--input_dir", default="/media/root/HDD31/Developmental/P28/241204_P28-TH-RI-B2-647_12-09-37")
    parser.add_argument("--model_path", required=True, help="所选视角的模型权重目录")
    parser.add_argument("--view", choices=["axial", "coronal", "sagittal"], default="axial", help="只推理这个视角，默认 axial")
    parser.add_argument("--output_dir", required=True, help="结果保存目录")
    parser.add_argument("--cache_dir", default=None, help="临时缓存目录，默认使用 output_dir")
    parser.add_argument("--output_prefix", default="pred_whole_brain")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cube_size", type=int, default=160, help="归一化 cube 的边长")
    parser.add_argument("--patch_size", type=int, default=160, help="模型输入的二维 patch 边长")
    parser.add_argument("--overlap", type=int, default=20, help="仅原始 X/Y 方向的 cube 和 patch 重叠大小；Z 无重叠")
    parser.add_argument("--batch_size", type=int, default=24, help="一次送入 GPU 的切片数量")
    parser.add_argument("--block_size", type=int, default=512, help="输出块边长，调小可降低 CPU 内存占用")
    args = parser.parse_args()
    if not (8 <= args.patch_size <= args.cube_size and 0 <= args.overlap < args.patch_size):
        parser.error("需要满足 8 <= patch_size <= cube_size，0 <= overlap < patch_size")
    if args.batch_size < 1 or args.block_size < 1:
        parser.error("batch_size 和 block_size 必须大于 0")
    main(args)

## Contents

- `train_2d_view_mutual_unet_union_inter_with_test_3views_warmupval_optimized.py`: training and pseudo-label generation.
- `dataset.py`, `model/`, `myutils/`, `eval_volumes.py`: data loading, network, and evaluation utilities.
- `test_3views_inference.py`: volume inference and test metrics.
- `test_whole_brain_inference.py`: whole-brain inference from TIFF slices.
- `computation_test_3views_new.py`: computational efficiency test.
- `output/train/`: trained checkpoints, organized by dataset and seed. Each `model/` directory contains `best` and `latest` checkpoints for the three views.

## Training

```bash
python train_2d_view_mutual_unet_union_inter_with_test_3views_warmupval_optimized.py \
  --train_image_path /path/to/train/images \
  --train_mask_path /path/to/train/masks \
  --val_image_path /path/to/val/images \
  --val_mask_path /path/to/val/masks \
  --test_image_path /path/to/test/images \
  --test_mask_path /path/to/test/masks \
  --save_path ./runs/example/model \
  --output_path ./runs/example \
  --epochs 50 --batch_size 24 --warmup_epochs 2 --seed 123 \
  --use_skeleton_for_intersection
```

## Volume evaluation

```bash
python test_3views_inference.py \
  --model_path /path/to/checkpoint/model \
  --test_image_path /path/to/test/images \
  --test_mask_path /path/to/test/masks \
  --output_dir ./runs/evaluation \
  --views axial --checkpoint best
```

Set `--views` to `coronal` or `sagittal` to evaluate another network.

## Whole-brain inference

```bash
python test_whole_brain_inference.py \
  --input_dir /path/to/tiff_slices \
  --model_path /path/to/checkpoint/model \
  --output_dir ./runs/whole_brain \
  --view axial
```

## Computational efficiency

```bash
python computation_test_3views_new.py
```

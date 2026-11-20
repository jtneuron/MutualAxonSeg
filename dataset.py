import os
import numpy as np
from PIL import Image
import tifffile as tiff  # Load 3D TIFF images.
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': F.resize(image, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        }

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # RGB image mean and standard deviation.
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class RandomFlip(object):
    """Apply a random horizontal or vertical flip."""
    def __call__(self, data):
        image, label = data['image'], data['label']
        
        # Random horizontal flip.
        if torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        
        # Random vertical flip.
        if torch.rand(1).item() < 0.5:
            image = F.vflip(image)
            label = F.vflip(label)
        
        return {'image': image, 'label': label}


class RandomBrightnessGamma(object):
    """Apply random brightness and gamma adjustments."""
    def __init__(self, brightness_factor=0.2, gamma_factor=0.2):
        self.brightness_factor = brightness_factor
        self.gamma_factor = gamma_factor
    
    def __call__(self, data):
        image, label = data['image'], data['label']
        
        # Random brightness adjustment (0.8 to 1.2).
        brightness = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness_factor
        image = F.adjust_brightness(image, brightness)
        
        # Random gamma adjustment (0.8 to 1.2).
        gamma = 1.0 + (torch.rand(1).item() * 2 - 1) * self.gamma_factor
        image = F.adjust_gamma(image, gamma)
        
        return {'image': image, 'label': label}


class RandomCrop(object):
    """Apply a random crop."""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, data):
        image, label = data['image'], data['label']
        
        # PIL images use .size (width, height); tensors use .shape (C, H, W).
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            _, h, w = image.shape
        
        crop_h, crop_w = self.size
        
        # Ensure the image is at least as large as the crop.
        if h < crop_h or w < crop_w:
            image = F.resize(image, (crop_h, crop_w))
            label = F.resize(label, (crop_h, crop_w), interpolation=InterpolationMode.NEAREST)
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                _, h, w = image.shape
        
        # Choose a random crop position.
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        
        image = F.crop(image, top, left, crop_h, crop_w)
        label = F.crop(label, top, left, crop_h, crop_w)
        
        return {'image': image, 'label': label}

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, slice_nums, mode, view='axial', subset_start=None, subset_size=None):
        """
        image_root: Directory of 3D TIFF images.
        gt_root: Directory of 3D TIFF labels.
        size: Output size of each 2D image.
        mode: 'train' or 'val'.
        view: Slice orientation: 'axial', 'coronal', or 'sagittal'.
        subset_start: Starting index of the subset.
        subset_size: Number of items in the subset.
        """
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tiff') or f.endswith('.tif')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tiff') or f.endswith('.tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # Select the requested subset.
        if subset_start is not None and subset_size is not None:
            end_idx = min(subset_start + subset_size, len(self.images))
            self.images = self.images[subset_start:end_idx]
            self.gts = self.gts[subset_start:end_idx]
            print(f"Loading subset from {subset_start} to {end_idx}, total {len(self.images)} images")

        if len(self.images) != len(self.gts):
            raise ValueError(f"Number of images ({len(self.images)}) and labels ({len(self.gts)}) are not the same!")

        self.size = size
        self.mode = mode
        self.view = view
        self.slice_nums = slice_nums

        # Define transforms and normalization.
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __len__(self):
        # Compute the dataset length.
        num_images = len(self.images)
        slices_per_image = self.slice_nums  # Number of slices per 3D image.
        total_slices = num_images * slices_per_image
        return total_slices

    def load_tiff(self, path):
        """Load a 3D TIFF image as a NumPy array."""
        with tiff.TiffFile(path) as tif:
            return tif.asarray()

    def __getitem__(self, idx):
        # Validate the data.
        file_idx = idx // self.slice_nums
        slice_idx = idx % self.slice_nums

        # Load and validate the data.
        try:
            image_3d = self.load_tiff(self.images[file_idx])
            label_3d = self.load_tiff(self.gts[file_idx])
            
            # Check the data dimensions.
            if image_3d.ndim != 3 or label_3d.ndim != 3:
                raise ValueError(f"Invalid dimensions: image {image_3d.shape}, label {label_3d.shape}")

            # Extract the slice for the selected view.
            if self.view == 'axial':
                image_slice = image_3d[slice_idx, :, :]
                label_slice = label_3d[slice_idx, :, :]
            elif self.view == 'coronal':
                image_slice = image_3d[:, slice_idx, :]
                label_slice = label_3d[:, slice_idx, :]
            elif self.view == 'sagittal':
                image_slice = image_3d[:, :, slice_idx]
                label_slice = label_3d[:, :, slice_idx]
            else:
                raise ValueError(f"Invalid view: {self.view}")

            # Ensure the data type and range are valid.
            image_slice = image_slice.astype(np.uint8)
            label_slice = label_slice.astype(np.uint8)

            # Convert to PIL images.
            image_slice = Image.fromarray(image_slice).convert("RGB")
            label_slice = Image.fromarray(label_slice).convert("L")

            # Apply transforms.
            data = {'image': image_slice, 'label': label_slice}
            data = self.transform(data)

            # Add metadata.
            data['name'] = os.path.basename(self.images[file_idx])
            data['image_index'] = file_idx
            data['slice_idx'] = slice_idx  # Record slice_idx to restore slice order after shuffling.

            return data

        except Exception as e:
            print(f"Error loading data at index {idx}: {str(e)}")
            # Return the next valid sample.
            return self.__getitem__((idx + 1) % self.__len__())

    def get_original_3d(self, image_index):
        """
        Return the complete, unprocessed 3D image at the given index.
        """
        if image_index < 0 or image_index >= len(self.images):
            raise IndexError(f"Image index {image_index} out of range.")
        return self.load_tiff(self.images[image_index])

    def get_original_label_3d(self, image_index):
        """
        Return the complete, unprocessed 3D label at the given index.
        """
        if image_index < 0 or image_index >= len(self.gts):
            raise IndexError(f"Label index {image_index} out of range.")
        return self.load_tiff(self.gts[image_index])


class FullDataset_160(Dataset):
    def __init__(self, image_root, gt_root, size, mode, view='axial', subset_start=None, subset_size=None):
        """
        image_root: Directory of 3D TIFF images.
        gt_root: Directory of 3D TIFF labels.
        size: Output size of each 2D image.
        mode: 'train' or 'val'.
        view: Slice orientation: 'axial', 'coronal', or 'sagittal'.
        subset_start: Starting index of the subset.
        subset_size: Number of items in the subset.
        """
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tiff') or f.endswith('.tif')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tiff') or f.endswith('.tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # Select the requested subset.
        if subset_start is not None and subset_size is not None:
            end_idx = min(subset_start + subset_size, len(self.images))
            self.images = self.images[subset_start:end_idx]
            self.gts = self.gts[subset_start:end_idx]
            print(f"Loading subset from {subset_start} to {end_idx}, total {len(self.images)} images")

        if len(self.images) != len(self.gts):
            raise ValueError(f"Number of images ({len(self.images)}) and labels ({len(self.gts)}) are not the same!")

        self.size = size
        self.mode = mode
        self.view = view

        # Define transforms and normalization.
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                # Resize((size + 40, size + 40)),   # First resize to a larger size.
                # RandomCrop((size, size)),         # Randomly crop to the target size.
                ToTensor(),                       # Convert to a tensor.
                # RandomFlip(),                     # Random horizontal or vertical flip.
                # RandomBrightnessGamma(            # Random brightness and gamma adjustment.
                #     brightness_factor=0.2,
                #     gamma_factor=0.2
                # ),
                Normalize()                       # Normalize.
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __len__(self):
        # Compute the dataset length.
        num_images = len(self.images)
        slices_per_image = 160  # Number of slices per 3D image.
        total_slices = num_images * slices_per_image
        return total_slices

    def load_tiff(self, path):
        """Load a 3D TIFF image as a NumPy array."""
        with tiff.TiffFile(path) as tif:
            return tif.asarray()

    def __getitem__(self, idx):
        # Validate the data.
        file_idx = idx // 160
        slice_idx = idx % 160

        # Load and validate the data.
        try:
            image_3d = self.load_tiff(self.images[file_idx])
            label_3d = self.load_tiff(self.gts[file_idx])
            
            # Check the data dimensions.
            if image_3d.ndim != 3 or label_3d.ndim != 3:
                raise ValueError(f"Invalid dimensions: image {image_3d.shape}, label {label_3d.shape}")

            # Extract the slice for the selected view.
            if self.view == 'axial':
                image_slice = image_3d[slice_idx, :, :]
                label_slice = label_3d[slice_idx, :, :]
            elif self.view == 'coronal':
                image_slice = image_3d[:, slice_idx, :]
                label_slice = label_3d[:, slice_idx, :]
            elif self.view == 'sagittal':
                image_slice = image_3d[:, :, slice_idx]
                label_slice = label_3d[:, :, slice_idx]
            else:
                raise ValueError(f"Invalid view: {self.view}")

            # Ensure the data type and range are valid.
            image_slice = image_slice.astype(np.uint8)
            label_slice = label_slice.astype(np.uint8)

            # Convert to PIL images.
            image_slice = Image.fromarray(image_slice).convert("RGB")
            label_slice = Image.fromarray(label_slice).convert("L")

            # Apply transforms.
            data = {'image': image_slice, 'label': label_slice}
            data = self.transform(data)

            # Add metadata.
            data['name'] = os.path.basename(self.images[file_idx])
            data['image_index'] = file_idx
            data['slice_idx'] = slice_idx  # Record slice_idx to restore slice order after shuffling.

            return data

        except Exception as e:
            print(f"Error loading data at index {idx}: {str(e)}")
            # Return the next valid sample.
            return self.__getitem__((idx + 1) % self.__len__())

    def get_original_3d(self, image_index):
        """
        Return the complete, unprocessed 3D image at the given index.
        """
        if image_index < 0 or image_index >= len(self.images):
            raise IndexError(f"Image index {image_index} out of range.")
        return self.load_tiff(self.images[image_index])

    def get_original_label_3d(self, image_index):
        """
        Return the complete, unprocessed 3D label at the given index.
        """
        if image_index < 0 or image_index >= len(self.gts):
            raise IndexError(f"Label index {image_index} out of range.")
        return self.load_tiff(self.gts[image_index])


class TestDataset(Dataset):
    def __init__(self, image_root, label_root=None, size=352, view='axial'):
        """
        Load 3D test images and slice them in the selected orientation.
        :param image_root: Path to a 3D image directory or one TIFF file.
        :param label_root: Path to a 3D label directory or one TIFF file; load labels if supplied.
        :param size: Target size of each slice.
        :param view: Slice orientation: 'axial', 'coronal', or 'sagittal'.。
        """
        if os.path.isdir(image_root):
            self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tiff') or f.endswith('.tif')]
        elif os.path.isfile(image_root) and (image_root.endswith('.tiff') or image_root.endswith('.tif')):
            self.images = [image_root]
        else:
            raise ValueError(f"[ERROR] Invalid image_root: {image_root}. It should be a directory or a .tiff file.")

        if not self.images:
            raise ValueError(f"[ERROR] No valid .tiff or .tif files found in {image_root}.")

        self.images = sorted(self.images)
        print(f"[INFO] Loaded {len(self.images)} 3D images from {image_root}")
        
        # Load labels if provided.
        self.has_labels = label_root is not None
        if self.has_labels:
            if os.path.isdir(label_root):
                self.labels = [os.path.join(label_root, f) for f in os.listdir(label_root) if f.endswith('.tiff') or f.endswith('.tif')]
            elif os.path.isfile(label_root) and (label_root.endswith('.tiff') or label_root.endswith('.tif')):
                self.labels = [label_root]
            else:
                raise ValueError(f"[ERROR] Invalid label_root: {label_root}. It should be a directory or a .tiff file.")
            
            if not self.labels:
                raise ValueError(f"[ERROR] No valid .tiff or .tif files found in {label_root}.")
            
            self.labels = sorted(self.labels)
            print(f"[INFO] Loaded {len(self.labels)} 3D labels from {label_root}")
            
            if len(self.images) != len(self.labels):
                raise ValueError(f"[ERROR] Number of images ({len(self.images)}) does not match number of labels ({len(self.labels)}).")

        self.size = size
        self.view = view

        # Define transforms consistent with training.
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define the label transform.
        self.label_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def load_tiff(self, path):
        """Load a 3D TIFF image as a NumPy array."""
        with tiff.TiffFile(path) as tif:
            return tif.asarray()

    def __getitem__(self, idx):
        """
        Return all slices of one 3D image and their labels, if available.
        """
        try:
            # Load the 3D image.
            image_3d = self.load_tiff(self.images[idx])
            name = os.path.basename(self.images[idx]).split('.')[0]
            
            # Ensure name is a string.
            if isinstance(name, tuple):
                name = name[0]
            elif isinstance(name, list):
                name = name[0]
            
            # Process slices according to the selected view.
            if self.view == 'axial':
                depth, height, width = image_3d.shape
            elif self.view == 'coronal':
                height, depth, width = image_3d.shape
            elif self.view == 'sagittal':
                height, width, depth = image_3d.shape
            else:
                raise ValueError(f"[ERROR] Unsupported view: {self.view}")
            
            # Retain the original 3D image for visualization.
            original_3d = image_3d
            
            # Process all slices.
            all_slices = []
            original_slices = []
            
            for slice_idx in range(depth):
                # Extract the slice for the selected view.
                if self.view == 'axial':
                    image_slice = image_3d[slice_idx, :, :]
                elif self.view == 'coronal':
                    image_slice = image_3d[:, slice_idx, :]
                elif self.view == 'sagittal':
                    image_slice = image_3d[:, :, slice_idx]
                
                # Save the original slice.
                original_slices.append(image_slice)
                
                # Convert to a PIL image and apply transforms.
                image_slice_pil = Image.fromarray(image_slice).convert('RGB')
                image_slice_tensor = self.transform(image_slice_pil)
                all_slices.append(image_slice_tensor)
            
            # Stack all slices.
            all_slices_tensor = torch.stack(all_slices)
            
            # Process labels if available.
            if self.has_labels:
                label_3d = self.load_tiff(self.labels[idx])
                
                # Process labels according to the selected view.
                all_label_slices = []
                for slice_idx in range(depth):
                    if self.view == 'axial':
                        label_slice = label_3d[slice_idx, :, :]
                    elif self.view == 'coronal':
                        label_slice = label_3d[:, slice_idx, :]
                    elif self.view == 'sagittal':
                        label_slice = label_3d[:, :, slice_idx]
                    
                    label_slice_pil = Image.fromarray(label_slice).convert('L')
                    label_slice_tensor = self.label_transform(label_slice_pil)
                    all_label_slices.append(label_slice_tensor)
                
                all_label_slices_tensor = torch.stack(all_label_slices)
                
                return all_slices_tensor, torch.tensor(image_3d.shape), name, original_slices, all_label_slices_tensor
            else:
                return all_slices_tensor, torch.tensor(image_3d.shape), name, original_slices
            
        except Exception as e:
            print(f"[ERROR] Failed to load item {idx}: {str(e)}")
            # On error, try the next sample unless this is the last one.
            if idx < len(self.images) - 1:
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"[CRITICAL] Failed to load any valid data from dataset.")

    def __len__(self):
        """Return the number of 3D images in the dataset."""
        return len(self.images)
        
    def get_original_3d(self, image_index):
        """
        Return the complete, unprocessed 3D image at the given index.
        """
        if image_index < 0 or image_index >= len(self.images):
            raise IndexError(f"Image index {image_index} out of range.")
        return self.load_tiff(self.images[image_index])
    
    def get_original_label_3d(self, image_index):
        """
        Return the complete, unprocessed 3D label at the given index.
        """
        if not self.has_labels:
            raise ValueError("Dataset does not have labels.")
        if image_index < 0 or image_index >= len(self.labels):
            raise IndexError(f"Label index {image_index} out of range.")
        return self.load_tiff(self.labels[image_index])
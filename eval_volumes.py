from myutils import *
import pdb


def read_file(file_path):
    """
    Read an image file by extension (NIfTI or TIFF).
    """
    if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
        img = nib.load(file_path)
        data = img.get_fdata()
        return data
    elif file_path.endswith('.tiff') or file_path.endswith('.tif'):
        if tifffile is not None:
            data = tifffile.imread(file_path)
        else:
            # Use PIL as a fallback.
            from PIL import Image
            img = Image.open(file_path)
            data = np.array(img)
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path}")



def eval_two_volumes_maxpool(pre, label, pool_kernel, threshold, device):
    # pre = read_nifti(root)
    # pre = read_tiff_stack(root)
    # label = read_nifti(target)
    pre = pre.copy()
    label = label.copy()
    k = pool_kernel
    s = max(1, k - 1)
    kernel = (k, k, k)
    stride = (s, s, s)
    pre[pre < threshold] = 0
    pre[pre >= threshold] = 1
    label[label > 0] = 1

    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    clrecall, clprecision, recall, precision = soft_cldice_f1(pre, label)
    cldice = (2. * clrecall * clprecision) / (clrecall + clprecision)
    acc = calculate_accuracy(pre, label)

    return {'iou': total_loss_iou,
            't_iou': total_loss_tiou,
            'cldice': cldice,
            'clprecision': clprecision,
            'clrecall': clrecall,
            'dice': dice_score,
            'precision': precision,
            'recall': recall,
            'acc': acc}




def eval_two_volume_dirs_maxpool(target, root, data, pool_kernel, device):
   
    if 'P28' in root or 'P14' in root:
        label = read_file(join(target, data))
        pre = read_file(join(root, 'case_' + data.split('_')[0].split('-')[1] + '.nii.gz'))       

    else:
        label = read_file(join(target, data))
        pre = read_file(join(root, 'case_' + data.split('.')[0] + '.nii.gz'))

    # label = read_file(join(target, data))
    # pre = read_file(join(root, data.split('.')[0] + '_prediction.tiff'))
    # pre[pre < 125] = 0
    # pre[pre >= 125] = 1  

    k = pool_kernel
    kernel = (k, k, k)
    
    pre[pre < 0.5] = 0
    pre[pre >= 0.5] = 1
    label[label > 0.5] = 1
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    # pre = np.transpose(pre, (1, 0, 2))

    # pre = np.transpose(pre, (1, 2, 0))

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    clrecall, clprecision, recall, precision = soft_cldice_f1(pre, label)
    cldice = (2. * clrecall * clprecision) / (clrecall + clprecision)
    acc = calculate_accuracy(pre, label)

    return {'iou': total_loss_iou,
            'tiou': total_loss_tiou,
            'cldice': cldice,
            'clprecision': clprecision,
            'clrecall': clrecall,
            'dice': dice_score,
            'precision': precision,
            'recall': recall,
            'acc': acc}


if __name__ == "__main__":
    val_root = "./3D_predictions_test2Dloss_cldice/1_epoch_0_view_axial_label.tiff"
    val_target = "./3D_predictions_test2Dloss_cldice/1_epoch_0_view_axial_prediction.tiff"
    kernel_size = 2  # maxpooling kernel size
    threshold = 125  # probability threshold of the positive class, no need for nnunet prediction
    device = torch.device('cuda:1')
    loss = eval_two_volumes_maxpool(val_target, val_root, kernel_size, device)

import albumentations as A
import random as _random

# WAM config 기반 aug weight
_AUG_WEIGHTS = {
    'identity': 1, 'jpeg': 1, 'resize': 1, 'crop': 1,
    'rotate': 1, 'hflip': 1, 'perspective': 1,
    'gaussian_blur': 1, 'median_filter': 1,
    'brightness': 1, 'contrast': 1, 'saturation': 1, 'hue': 1,
    'crop_resize_pad': 2,
}


def sample_random_aug_transform(image_size=256):
    """
    WAM config 기반으로 aug 종류와 파라미터를 무작위 샘플링.
    identity 포함, weight에 비례한 확률로 선택.
    Returns: A.Compose transform or None (identity)
    """
    aug_types = list(_AUG_WEIGHTS.keys())
    weights   = list(_AUG_WEIGHTS.values())
    aug_type  = _random.choices(aug_types, weights=weights, k=1)[0]

    if aug_type == 'identity':
        return None

    elif aug_type == 'jpeg':
        q = _random.randint(40, 80)
        return A.Compose([A.ImageCompression(quality_range=(q, q), p=1.0)])

    elif aug_type == 'resize':
        scale = _random.uniform(0.7, 1.5)
        sz = max(1, int(image_size * scale))
        return A.Compose([
            A.Resize(height=sz, width=sz, p=1.0),
            A.Resize(height=image_size, width=image_size, p=1.0),
        ])

    elif aug_type == 'crop':
        ratio = _random.uniform(0.33, 1.0)
        sz = max(1, int(image_size * ratio))
        return A.Compose([
            A.CenterCrop(height=sz, width=sz, p=1.0),
            A.Resize(height=image_size, width=image_size, p=1.0),
        ])

    elif aug_type == 'rotate':
        angle = _random.uniform(-10, 10)
        return A.Compose([A.Rotate(limit=(angle, angle), border_mode=0, p=1.0)])

    elif aug_type == 'hflip':
        return A.Compose([A.HorizontalFlip(p=1.0)])

    elif aug_type == 'perspective':
        scale = _random.uniform(0.1, 0.5)
        return A.Compose([A.Perspective(scale=(scale, scale), keep_size=True, p=1.0)])

    elif aug_type == 'gaussian_blur':
        k = _random.randrange(3, 18, 2)  # odd: 3, 5, ..., 17
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        return A.Compose([A.GaussianBlur(blur_limit=(k, k), sigma_limit=(sigma, sigma), p=1.0)])

    elif aug_type == 'median_filter':
        k = _random.randrange(3, 8, 2)  # odd: 3, 5, 7
        return A.Compose([A.MedianBlur(blur_limit=(k, k), p=1.0)])

    elif aug_type == 'brightness':
        factor = _random.uniform(0.5, 2.0)
        limit = factor - 1.0  # [-0.5, 1.0]
        return A.Compose([A.RandomBrightnessContrast(
            brightness_limit=(limit, limit), contrast_limit=0, p=1.0)])

    elif aug_type == 'contrast':
        factor = _random.uniform(0.5, 2.0)
        limit = factor - 1.0  # [-0.5, 1.0]
        return A.Compose([A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(limit, limit), p=1.0)])

    elif aug_type == 'saturation':
        factor = _random.uniform(0.5, 2.0)
        shift = int((factor - 1.0) * 100)  # [-50, 100] (S채널 [0,255] 기준)
        return A.Compose([A.HueSaturationValue(
            hue_shift_limit=0, sat_shift_limit=(shift, shift), val_shift_limit=0, p=1.0)])

    elif aug_type == 'hue':
        factor = _random.uniform(-0.1, 0.1)
        shift = int(factor * 180)  # [-18, 18] (OpenCV H채널 [0,180] 기준, ≈±36°)
        return A.Compose([A.HueSaturationValue(
            hue_shift_limit=(shift, shift), sat_shift_limit=0, val_shift_limit=0, p=1.0)])

    elif aug_type == 'crop_resize_pad':
        resize_scale = _random.uniform(0.7, 1.5)
        crop_ratio   = _random.uniform(0.5, 0.66)
        resized_sz   = max(1, int(image_size * resize_scale))
        crop_sz      = max(1, int(resized_sz * crop_ratio))
        return A.Compose([
            A.Resize(height=resized_sz, width=resized_sz, p=1.0),
            A.CenterCrop(height=crop_sz, width=crop_sz, p=1.0),
            A.Resize(height=image_size, width=image_size, p=1.0),
        ])

    return None


def get_robustness_transform(aug_type, aug_param, image_size=512):
    if aug_type is None or aug_param is None:
        return None

    transforms_list = []

    if aug_type == 'brightness':
        # aug_param: -1.0 ~ 1.0 (1.5, 2.0)
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=(aug_param, aug_param), contrast_limit=0, p=1.0))
    
    elif aug_type == 'contrast':
        # aug_param: -1.0 ~ 1.0 (1.5, 2.0)
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(aug_param, aug_param), p=1.0))
        
    elif aug_type == 'hue':
        # aug_param: -0.5 ~ 0.5 (-0.1, 0.1)
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=(aug_param*20, aug_param*20), sat_shift_limit=0, val_shift_limit=0, p=1.0))
        
    elif aug_type == 'saturation':
        # aug_param: -1.0 ~ 1.0 (1.5, 2.0)
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(aug_param*30, aug_param*30), val_shift_limit=0, p=1.0))
        
    elif aug_type == 'jpeg':
        # aug_param: quality (50, 80)
        quality = int(aug_param)
        transforms_list.append(A.ImageCompression(quality_range=(quality, quality), p=1.0))
        
    elif aug_type == 'gaussian_blur':
        # aug_param: kernel size (3, 17)
        k = int(aug_param)
        if k % 2 == 0: k += 1
        # OpenCV formula: sigma derived from kernel size
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        transforms_list.append(A.GaussianBlur(blur_limit=(k, k), sigma_limit=(sigma, sigma), p=1.0))

    elif aug_type == 'gaussian_noise':
        # aug_param: sigma in pixel space (e.g. 1, 3, 5)
        std_normalized = aug_param / 255.0
        transforms_list.append(A.GaussNoise(std_range=(std_normalized, std_normalized), p=1.0))
        
    elif aug_type == 'median_filter':
        # aug_param: kernel size (3, 7)
        k = int(aug_param)
        if k % 2 == 0: k += 1
        transforms_list.append(A.MedianBlur(blur_limit=(k, k), p=1.0))
        
    elif aug_type == 'horizontal_flip': 
        transforms_list.append(A.HorizontalFlip(p=1.0))
        
    elif aug_type == 'crop':
        # aug_param: crop ratio (0.33, 0.5)
        crop_size = int(image_size * aug_param)
        transforms_list.append(A.CenterCrop(height=crop_size, width=crop_size, p=1.0))
        transforms_list.append(A.Resize(height=image_size, width=image_size, p=1.0))
        
    elif aug_type == 'resize':
        # aug_param: scale ratio (0.5)
        scaled_size = int(image_size * aug_param)
        transforms_list.append(A.Resize(height=scaled_size, width=scaled_size, p=1.0))
        transforms_list.append(A.Resize(height=image_size, width=image_size, p=1.0))
        
    elif aug_type == 'rotation':
        # aug_param: degrees (-10, 10)
        transforms_list.append(A.Rotate(limit=(aug_param, aug_param), p=1.0, border_mode=0)) # border_mode=0 (CONSTANT/Black)
        
    elif aug_type == 'perspective':
        # aug_param: scale (0.1, 0.5)
        transforms_list.append(A.Perspective(scale=(aug_param, aug_param), keep_size=True, p=1.0))

    return A.Compose(transforms_list)
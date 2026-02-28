import albumentations as A

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
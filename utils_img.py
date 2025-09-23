import torch


def normalize_img(x, image_mean = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1), image_std = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x, image_mean = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1), image_std = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def round_pixel(x):
    """ 
    Round pixel values to nearest integer. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255) - x_pixel.detach() + x_pixel
    y = normalize_img(y/255.0)
    return y



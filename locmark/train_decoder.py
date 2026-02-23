import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import os
from torchvision.datasets import CocoDetection
from pycocotools import mask as maskUtils
import numpy as np
import random
from torchvision.utils import save_image

# from WAM
class CocoImageIDWrapper(CocoDetection):
    def __init__(self, root, annFile, transform=None, mask_transform=None, random_nb_object=True, max_nb_masks=4, multi_w=False):
        super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        mask = self._load_mask(id)
        if mask is None:
            return None  # Skip this image if no valid mask is available

        img, mask = self.transforms(img, mask)
        return img, mask

    def _load_mask(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  # Return None if there are no annotations

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        # Initialize a list to hold all masks
        masks = []
        if self.random_nb_object and np.random.rand() < 0.5:
            random.shuffle(anns)
            anns = anns[:np.random.randint(1, len(anns)+1)]
        if not(self.multi_w):
            mask = np.zeros((original_height, original_width), dtype=np.float32)
            # one mask for all objects
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m)
            mask = torch.tensor(mask, dtype=torch.float32)
            return mask[None, ...]  # Add channel dimension
        else:
            anns = anns[:self.max_nb_masks]
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            # Stack all masks along a new dimension to create a multi-channel mask tensor
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.bool)
                # Check if the number of masks is less than max_nb_masks
                if masks.shape[0] < self.max_nb_masks:
                    # Calculate the number of additional zero masks needed
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    # Create additional zero masks
                    additional_masks = torch.zeros((additional_masks_count, original_height, original_width), dtype=torch.bool)
                    # Concatenate the original masks with the additional zero masks
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                # Return a tensor of shape (max_nb_masks, height, width) filled with zeros if there are no masks
                masks = torch.zeros((self.max_nb_masks, original_height, original_width), dtype=torch.bool)
            return masks
        
def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    images, masks = zip(*batch)
    images = torch.stack(images)
    
    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    
    # Pad each mask tensor to have 'max_masks' masks and add the inverse mask
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        union_mask = torch.max(mask, dim=0).values  # Assuming mask is of shape [num_masks, H, W]
        
        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(mask, pad=(0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:
            padded_mask = mask
            
        padded_masks.append(padded_mask)
    
    # Stack the padded masks
    masks = torch.stack(padded_masks)
    
    
    return images, masks

# ==========================================
# 1. Shallow Decoder 모델
# ==========================================
class ShallowDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1) 
        )

    def forward(self, x):
        return self.decoder(x)

# ==========================================
# 2. 합성 Logit 생성 함수
# ==========================================
def generate_synthetic_logits(masks, device):
    B, _, H, W = masks.shape
    
    bg_mean, bg_std = 0.455, 0.045
    fg_mean_range = (0.56, 0.61)
    fg_std = 0.037

    blurred_masks = TF.gaussian_blur(masks, kernel_size=[7, 7], sigma=[1.5, 1.5])

    bg_base = torch.full((B, 1, H, W), bg_mean, device=device)
    fg_means = torch.empty(B, 1, 1, 1, device=device).uniform_(*fg_mean_range)
    fg_base = torch.full((B, 1, H, W), 1.0, device=device) * fg_means
    
    synthetic_logits = bg_base * (1 - blurred_masks) + fg_base * blurred_masks

    noise = torch.randn((B, 1, H, W), device=device) * bg_std
    synthetic_logits = synthetic_logits + noise
    
    return torch.clamp(synthetic_logits, 0.0, 1.0)

# ==========================================
# 3. 전체 학습 파이프라인
# ==========================================
def train_framework():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    epochs = 10
    lr = 1e-3
    img_size = 256

    save_dir = "/mnt/nas5/suhyeon/projects/locmark_decoder/0223"
    os.makedirs(save_dir, exist_ok=True)

    train_dir = "/mnt/nas5/suhyeon/datasets/coco-2017/train2017"
    ann_file = "/mnt/nas5/suhyeon/datasets/coco-2017/annotations/instances_train2017.json"
    
    dummy_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    mask_transform = T.Compose([T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)])
    
    dataset = CocoImageIDWrapper(
        root=train_dir,
        annFile=ann_file,
        transform=dummy_transform,
        mask_transform=mask_transform,
        random_nb_object=True, 
        multi_w=False          
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate 
    )
    
    model = ShallowDecoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            if masks is None or masks.numel() == 0:
                continue
            
            images = images.to(device)
            masks = masks.to(device) 
            
            inputs = generate_synthetic_logits(masks, device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            if batch_idx == 0:
                num_samples = min(4, images.size(0)) # 최대 4개 샘플 시각화
                
                inputs_viz = inputs[:num_samples].repeat(1, 3, 1, 1)
                outputs_viz = torch.sigmoid(outputs[:num_samples]).repeat(1, 3, 1, 1) # Sigmoid 적용
                gt_viz = masks[:num_samples].repeat(1, 3, 1, 1)
                
                grid = torch.cat([images[:num_samples], inputs_viz, outputs_viz, gt_viz], dim=0)
                
                save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
                save_image(grid, save_path, nrow=num_samples)
            
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "shallow_decoder_weights.pth")

if __name__ == "__main__":
    train_framework()
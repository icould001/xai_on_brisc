from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import os
import numpy as np
from PIL import Image
from skimage import measure, morphology, filters
from scipy.ndimage import binary_fill_holes

 
class Thresh:
    def __call__(self, img):
        # Convert to numpy grayscale
        img = np.array(img.convert("L"))
        #plt.imshow(Image.fromarray(img), cmap="gray"); plt.axis('off'); plt.title("actual - no preprocess"); plt.show()
        # Auto threshold if not given
        intensity_thresh = filters.threshold_otsu(img)

        binary = img > intensity_thresh
        #plt.imshow(Image.fromarray(binary), cmap="gray"); plt.axis('off'); plt.title("binary"); plt.show()
        
        filled = binary_fill_holes(binary)
        #plt.imshow(Image.fromarray(filled), cmap="gray"); plt.axis('off'); plt.title("filled"); plt.show()
        
        result = np.zeros_like(img)
        result[filled] = img[filled]
        #plt.imshow(Image.fromarray(result), cmap="gray"); plt.axis('off'); plt.title("result"); plt.show()
        return Image.fromarray(result)


class ResizeAndPad512:
    def __call__(self, img):
        # Original size
        w, h = img.size
        
        # Resize if larger than 512 in any dimension, keep aspect ratio
        if w > 512 or h > 512:
            scale = 512 / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            w, h = img.size
        
        # Compute padding to reach 512x512
        pad_left = (512 - w) // 2
        pad_top = (512 - h) // 2
        pad_right = 512 - w - pad_left
        pad_bottom = 512 - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        # Pad with black (0)
        img = TF.pad(img, padding, fill=0, padding_mode='constant')
        
        return img

def get_data(path_to_data = "~/dev/datasets/brisk/brisc2025/classification_task/"):
    path_to_data = os.path.expanduser(path_to_data)

    train_transform = transforms.Compose([
        ResizeAndPad512(),
        Thresh(),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        #transforms.Lambda(lambda tensor: torch.where(tensor < 40/255, torch.tensor(0.0), tensor)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        ResizeAndPad512(),
        Thresh(),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        #transforms.Lambda(lambda tensor: torch.where(tensor < 40/255, torch.tensor(0.0), tensor)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(path_to_data, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(path_to_data, 'test'), transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, test_loader

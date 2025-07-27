import data_proc
import models
import grad_cam
import lime_magic

from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import numpy as np

train_loader, test_loader = data_proc.get_data()

#model, layer_name = models.train(models.get_resnet_model, 25, train_loader, test_loader)

#path_to_dataset = "/kaggle/input/brisc2025"
path_to_dataset = os.path.expanduser("~/dev/datasets/brisk")

classification_path = os.path.join(path_to_dataset,"brisc2025/classification_task/test/")
segmentation_path = os.path.join(path_to_dataset, "brisc2025/segmentation_task/test")

image_name = "brisc2025_test_00750_pi_ax_t1"

# Load an RGB image
if os.path.exists(os.path.join(segmentation_path, 'images', image_name+".jpg")):
    real_image = Image.open(os.path.join(segmentation_path, 'images', image_name+".jpg"))
else:
    if os.path.exists(os.path.join(classification_path, "no_tumor", image_name+".jpg")):
        real_image = Image.open(os.path.join(classification_path, "no_tumor", image_name+".jpg"))
    else:
        print("didn't find image")
    
if os.path.exists(os.path.join(segmentation_path, 'masks', image_name+".png")):
    mask = Image.open(os.path.join(segmentation_path, 'masks', image_name+".png"))
else:
    mask = Image.new('RGB', (512, 512), color=(0,0,0))
    print("didn't find the mask")

plt.imshow(real_image, cmap="gray"); plt.axis('off'); plt.title("actual - no preprocess"); plt.show()

real_transform = transforms.Compose([
        data_proc.ResizeAndPad512(),
        data_proc.Thresh(),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        data_proc.ResizeAndPad512(),
        data_proc.Thresh(),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

image = test_transform(real_image)
real_image = real_transform(real_image)
mask = real_transform(mask)



#model, layer_name = models.train(models.get_densenet_model, 25, train_loader, test_loader)

model, layer_name = models.load_trained_model(models.get_resnet_model, len(train_loader.dataset.classes), "./best_model_resnet50_auto_threshed.pt")

#grad_cam(model, layer_name, image, real_image, mask)

temp1, mask1, temp2, mask2 = lime_magic.lime_magic(model, image.numpy(), real_image)


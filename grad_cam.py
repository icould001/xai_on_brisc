import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchcam.methods import GradCAM, GradCAMpp, LayerCAM, SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import numpy as np

def grad_cam(model, layer_name, image, real_image, mask):
    # Grad-CAM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam_extractor = GradCAMpp(model, layer_name)

    # Preprocess your data and feed it to the model
    out = model.to(device)(image.unsqueeze(0).to(device))
    print(out)
    pred_class = out.squeeze(0).argmax().item()
    print(pred_class)
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(pred_class, out)

    # Overlayed on the image
    for name, cam in zip(cam_extractor.target_names, activation_map):
      result = overlay_mask(to_pil_image(real_image, mode='RGB'), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
      plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()

    
    plt.imshow(to_pil_image(real_image, mode='RGB'))
    plt.imshow(to_pil_image(mask), cmap='jet', alpha=0.5)  # 0.0 = fully transparent, 1.0 = opaque
    plt.axis('off')
    plt.title("actual masked")
    plt.show()
    plt.imshow(to_pil_image(real_image, mode='RGB')); plt.axis('off'); plt.title("actual"); plt.show()

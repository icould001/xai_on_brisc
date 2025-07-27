import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

def lime_magic(model, image, real_image):
    # LIME needs a prediction function
    def predict_fn(images):
        model.eval()
        device = torch.device("cuda")
        batch = [torch.from_numpy(np.transpose(i, (2, 0, 1))) for i in images]
        batch = torch.stack(batch, dim=0)
        
        logits = model.to(device)(batch.to(device))
        probs = F.softmax(logits, dim=1)
        r = probs.detach().cpu().numpy()
        
        return r


    image = np.transpose(image, (1, 2, 0))
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Show LIME result
    plt.imshow(mask)
    plt.title("mask")
    plt.axis('off')
    plt.show()
    plt.imshow(temp)
    plt.title("temp")
    plt.axis('off')
    plt.show()
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("b1")
    plt.axis('off')
    plt.show()

    temp2, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    img_boundry2 = mark_boundaries(image, mask2)
    plt.imshow(mask2)
    plt.title("mask")
    plt.axis('off')
    plt.show()
    plt.imshow(temp2)
    plt.title("temp")
    plt.axis('off')
    plt.show()
    plt.imshow(img_boundry2)
    plt.title("b2")
    plt.axis('off')
    plt.show()
    print(mask.shape)
    print(mask.dtype)
    result = overlay_mask(to_pil_image(real_image, mode='RGB'), to_pil_image(mask.astype(np.float32), mode="F"), alpha=0.5)
    plt.imshow(result); plt.axis('off'); plt.title("name"); plt.show()
    return temp, mask, temp2, mask2

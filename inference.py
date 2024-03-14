import torch
from torchvision import transforms
import numpy as np
import cv2
import os

from lightning_model import DCSwin

from loader import get_dataloader

from tqdm import tqdm

# Define the path to the validation dataset
val_data_path = "/raid/home/sandej17/techteamet_segmentation/val_data"

# Define the path to the saved model weights
weights_path = "/raid/home/sandej17/techteamet_segmentation/checkpoints/model_weights.pth"

# Define the path to the predictions folder
predictions_path = "/raid/home/sandej17/segmentation_framework/predictions"

# Create the predictions folder if it doesn't exist
if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)
else:
    # Delete all the files in the predictions folder
    for file in os.listdir(predictions_path):
        os.remove(os.path.join(predictions_path, file))

# Define the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lm = DCSwin.load_from_checkpoint("checkpoints/last-v6.ckpt", num_classes=2, learning_rate=1e-3, train_loader=None, val_loader=None, model_size="tiny")

validation_loader, num_classes = get_dataloader("first_road", "val", 1, 1.0)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

lm.to(device)

lm.model.eval()

ious = {
    0: [],
    1: [],
    2: []
}

for _, (image, mask, idx_id) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
    image = image.to(device)
    mask = mask.to(device)
    output = lm.model(image)
    output = torch.argmax(output, dim=1)
    output = output.cpu().numpy()
    mask = mask.cpu().numpy()
    image = image[0].permute(1, 2, 0).cpu().numpy()
    
    # BEGIN: iou_calculation
    for c in range(num_classes):
        intersection = np.logical_and(output == c, mask == c).sum()
        union = np.logical_or(output == c, mask == c).sum()
        iou = intersection / union if union > 0 else 0
        ious[c].append(iou)
    # END: iou_calculation
    
    output[output == 1] = 100
    mask[mask == 1] = 100
    output[output == 2] = 200
    mask[mask == 2] = 200
    
    image = (image * 255).astype(np.uint8)
    
    mask = mask.astype(np.uint8)
    output = output.astype(np.uint8)
    
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    output = cv2.cvtColor(output[0], cv2.COLOR_GRAY2RGB)
    mask = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2RGB)
    
    combined = np.concatenate((np.array(image), np.array(output), np.array(mask)), axis=1)
    combined = cv2.line(combined, (combined.shape[1]//3, 0), (combined.shape[1]//3, combined.shape[0]), (255, 255, 255), thickness=2)
    combined = cv2.line(combined, (2*combined.shape[1]//3, 0), (2*combined.shape[1]//3, combined.shape[0]), (255, 255, 255), thickness=2)
    
    prediction_path = os.path.join(predictions_path, list(idx_id)[0])
    cv2.imwrite(prediction_path, combined)


for c in range(num_classes):
    print(f"Class {c} IoU: {np.mean(ious[c])}")
    

exit("")
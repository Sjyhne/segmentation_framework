import torch
from torchvision import transforms
import numpy as np
import cv2
import os

from lightning_model import DCSwin

from loader import get_dataloader

# Define the path to the validation dataset
val_data_path = "/raid/home/sandej17/techteamet_segmentation/val_data"

# Define the path to the saved model weights
weights_path = "/raid/home/sandej17/techteamet_segmentation/checkpoints/model_weights.pth"

# Define the path to the predictions folder
predictions_path = "/raid/home/sandej17/techteamet_segmentation/predictions"

# Create the predictions folder if it doesn't exist
if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)
else:
    # Delete all the files in the predictions folder
    for file in os.listdir(predictions_path):
        os.remove(os.path.join(predictions_path, file))

# Define the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lm = DCSwin.load_from_checkpoint("checkpoints/tiny_epoch=99-val_iou=0.64.ckpt", num_classes=3, learning_rate=1e-3, train_loader=None, val_loader=None, model_size="tiny")

validation_loader, num_classes = get_dataloader("combined", "val", 1, 0.1)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

lm.to(device)

lm.model.eval()

ious = {
    0: [],
    1: [],
    2: []
}

for image, mask, idx_id in validation_loader:
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

import cv2

# Define the transformation to apply to the validation images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the validation dataset
val_dataset = MyDataset(val_data_path, transform=transform)

# Loop over the validation dataset and perform inference on each image
for i in range(len(val_dataset)):
    # Get the image and its filename
    image, filename = val_dataset[i]
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = lm.model(image)
        output = torch.argmax(output, dim=1)
        output = output.cpu().numpy()

    # Get the mask
    mask = val_dataset.get_mask(filename)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Convert the prediction and image to PIL images
    prediction = transforms.ToPILImage()(output[0])
    image = transforms.ToPILImage()(image.cpu()[0])

    # Combine the prediction, image, and mask side-by-side
    combined = np.concatenate((np.array(image), np.array(prediction), np.array(mask)), axis=1)

    # Save the combined image
    prediction_path = os.path.join(predictions_path, filename)
    cv2.imwrite(prediction_path, combined)

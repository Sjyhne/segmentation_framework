from typing import List
import requests
import PIL
from PIL import Image
from io import BytesIO

import numpy as np
from tqdm import tqdm
import time

from pathlib import Path

def quantize_color(color, level):
    """Quantize the RGB values to the nearest multiple of the given level."""
    return np.round(color / level) * level


label_url = "https://openwms.statkart.no/skwms1/wms.fkb4?VERSION=1.3.0&service=WMS&request=GetMap&Format=image/png&GetFeatureInfo=text/plain&CRS=EPSG:25832&Layers=veg,bru,bygning&BBox=480492.00,6470934.00,481004.00,6471372.00&width=1000&height=1000%27;"


def get_label_url(layers: List[str], bbox: List[float], image_size: List[float]) -> str:
    """Returns a url for a map with the given layers and bounding box.

    Args:
        layers (list(str)): List of layers to include in the map.
        bbox (list(float)): List of coordinates for the bounding box. [minx, miny, maxx, maxy]

    Returns:
        str: Url for the map.
    """
    bbox_str = ",".join([str(x) for x in bbox])
    layers_str = ",".join(layers)
    url = f"https://openwms.statkart.no/skwms1/wms.fkb4?VERSION=1.3.0&service=WMS&request=GetMap&Format=image/png&GetFeatureInfo=text/plain&CRS=EPSG:25832&Layers={layers_str}&BBox={bbox_str}&width={image_size[1]}&height={image_size[0]}%27;"

    return url

def get_image_url(bbox: List[float], image_size: List[float]) -> str:
    """Returns a url for a map with the given layers and bounding box.

    Args:
        bbox (list(float)): List of coordinates for the bounding box. [minx, miny, maxx, maxy]

    Returns:
        str: Url for the map.
    """
    bbox_str = ",".join([str(x) for x in bbox])
    url = f"https://wms.geonorge.no/skwms1/wms.nib?service=WMS&request=GetMap&Format=image/png&GetFeatureInfo=text/plain&CRS=EPSG:25832&Layers=ortofoto&BBox={bbox_str}&width={image_size[1]}&height={image_size[0]}%27;"
    
    return url

def write_error(file, message):
    """Writes an error message to the given file.

    Args:
        file (str): The file to write to.
        message (str): The error message to write.
    """
    try:
        with open(file, "r") as f:
            lines = f.readlines()
            if message + "\n" in lines:
                return
    except FileNotFoundError:
        pass
    
    with open(file, "a") as f:
        f.write(message + "\n")

def download_image(url: str, label: bool) -> Image.Image:
    """Downloads an image from the given URL and returns it as an Image object.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image: The downloaded image as an Image object.
    """
    response = requests.get(url)
    try:
        image = Image.open(BytesIO(response.content))
    except PIL.UnidentifiedImageError as e:
        time.sleep(0.2)
        try:
            image = Image.open(BytesIO(response.content))
        except PIL.UnidentifiedImageError as e:
            return e
    
    
    if label:
        label = image.convert("L")
        label = np.array(label)
        
        label[label == 200] = 2 # Road

        road_percentage = np.sum(label == 2) / (label.shape[0] * label.shape[1])

        if road_percentage > 0:
            label[label == 250] = 0 # Background
            label[label == 150] = 1 # Building
            
            label = Image.fromarray(label)
            return label

        return Warning("No road in image")
    
    return image




if __name__ == "__main__":

    starting_point = [385774, 6428468]
    ending_point = [443974, 6472668]
    preferred_image_size = [500, 500]
    resolution = 0.2
    bbox_size = [preferred_image_size[0]*resolution, preferred_image_size[1]*resolution]
    
    # Get the number of images needed to cover the area
    num_images_x = int((ending_point[0] - starting_point[0]) / bbox_size[0])
    num_images_y = int((ending_point[1] - starting_point[1]) / bbox_size[1])
    num_images = num_images_x * num_images_y
    
    root_folder = Path("data")
    data_folder = root_folder.joinpath(f"{starting_point[0]}_{starting_point[1]}_{ending_point[0]}_{ending_point[1]}_{resolution}_{preferred_image_size[0]}_{preferred_image_size[1]}")
    image_folder = data_folder.joinpath("images")
    label_folder = data_folder.joinpath("labels")

    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)

    image_files = set([x for x in image_folder.glob("*.png")])
    label_files = set([x for x in label_folder.glob("*.png")])
    
    bboxes = []
    for x in range(num_images_x):
        for y in range(num_images_y):
            x0 = starting_point[0] + (x * bbox_size[0])
            y0 = starting_point[1] + (y * bbox_size[1])
            x1 = starting_point[0] + ((x + 1) * bbox_size[0])
            y1 = starting_point[1] + ((y + 1) * bbox_size[1])
            
            bboxes.append([x0, y0, x1, y1])
    
    # Get the bounding boxes for each image
    for _, bbox in tqdm(enumerate(bboxes), total=len(bboxes)):
        x0, y0, x1, y1 = bbox
        
        filename = f"{x0}_{y0}_{x1}_{y1}.png"
        
        label_exists = False
        image_exists = False
        
        if filename in label_files:
            print(f"found file {filename} in label folder")
            label_exists = True
        else:
            label_url = get_label_url(["veg", "bru", "bygning"], [x0, y0, x1, y1], preferred_image_size)
            label = download_image(label_url, label=True)
            if isinstance(label, PIL.UnidentifiedImageError) or isinstance(label, Warning):
                write_error(data_folder.joinpath("errors.txt"), f"Error downloading label for {filename}: {label}")
                continue
            
            label.save(label_folder.joinpath(filename))
        
        if filename in image_files:
            print(f"found file {filename} in image folder")
            image_exists = True
        else:
            image_url = get_image_url([x0, y0, x1, y1], preferred_image_size)
            image = download_image(image_url, label=False)
            image.save(image_folder.joinpath(filename))
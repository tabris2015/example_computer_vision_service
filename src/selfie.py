from enum import Enum
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel
from src.config import get_settings

SETTINGS = get_settings()

class ProcessingType(str, Enum):
    remove_background = "REMOVE_BACKGROUND"
    blur_background = "BLUR_BACKGROUND"

class BackgroundColor(BaseModel):
    red: int
    green: int
    blue: int

def get_filter_size(image_size):
    # Compute kernel size from image dimensions and multiplier
    ksize_x = int(image_size[1] * SETTINGS.blur_filter_factor)
    ksize_y = int(image_size[0] * SETTINGS.blur_filter_factor)

    # Ensure kernel sizes are positive and odd
    if ksize_x % 2 == 0:
        ksize_x += 1
    if ksize_y % 2 == 0:
        ksize_y += 1
    
    # Kernel size should not be less than 1
    ksize_x = max(1, ksize_x)
    ksize_y = max(1, ksize_y)
    
    return (ksize_x, ksize_y)


class SelfieProcessor:
    def __init__(self):
        self.model = YOLO(SETTINGS.yolo_version)

    def predict_mask(self, image_array, threshold=0.5):
        results = self.model(image_array, conf=threshold)[0]    # take first result from list
        labels = results.boxes.cls.tolist()
        base_mask = np.zeros(image_array.shape)
        plot_mask = results.plot(
            img=base_mask, 
            labels=False, 
            boxes=False, 
            probs=False
            ).astype(np.uint8)
        
        ret, selfie_mask = cv2.threshold(
            cv2.cvtColor(plot_mask, cv2.COLOR_BGR2GRAY), 
            1, 
            255, 
            cv2.THRESH_BINARY
        )

        return selfie_mask
    
    def remove_background(self, image_array, background_color: tuple[int]):
        selfie_mask = self.predict_mask(image_array)
        new_image = np.full(image_array.shape, background_color, dtype=np.uint8)
        new_image[selfie_mask != 0] = image_array[selfie_mask != 0]

        return new_image
    
    def blur_background(self, image_array):
        width, height, channels = image_array.shape
        filter_size = get_filter_size(image_array.shape)
        selfie_mask = self.predict_mask(image_array)
        new_image = cv2.GaussianBlur(image_array, filter_size, 0)
        new_image[selfie_mask != 0] = image_array[selfie_mask != 0]

        return new_image
    
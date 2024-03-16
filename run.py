from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import dotenv
import numpy as np
import cv2
import time
import json
from paddleocr import PaddleOCR

model = YOLO('ai_models/best-vehicle-lp.pt')
names = model.names

img = 'sample_images/test2.jpg'

result = model.predict(img, save=True)[0]
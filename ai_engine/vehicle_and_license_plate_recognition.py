from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os
import dotenv
import numpy as np
import cv2
import time
import json
from paddleocr import PaddleOCR

dotenv.load_dotenv()

def is_inside(xyxy1, xyxy2):
    flag = True
    if xyxy1[0] < xyxy2[0]:
        flag = False
    if xyxy1[1] < xyxy2[1]:
        flag = False
    if xyxy1[2] > xyxy2[2]:
        flag = False
    if xyxy1[3] > xyxy2[3]:
        flag = False

    return flag

class VehicleLicensePlateRecognition:
    def __init__(self):
        self.model = YOLO('ai_models/best.pt')
        self.ocr = PaddleOCR(lang='en', use_gpu=False)
        self.names = self.model.names

    def ocr_image(self, img, coordinates):
        img = cv2.imread(img)
        x,y,w,h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
        img = img[y:h,x:w]
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

        result = self.ocr.ocr(gray)[0]

        ret = ""
        if result is not None:
            for res in result:
                if ret == "":
                    ret += str(res[1][0])
                else:
                    ret += '-' + str(res[1][0])
        
        print(ret)
        return ret

    def predict(self, img):
        result = self.model.predict(img, device=0)[0]
        boxes = result.boxes

        cls_lp_bbox = []
        cls_v_bbox = []
        for i in range(len(boxes)):
            if boxes.cls[i] == 2:
                if ((boxes.xyxy[i][2] - boxes.xyxy[i][0]) >= (boxes.xyxy[i][3] - boxes.xyxy[i][1])):
                    cls_lp_bbox.append((boxes.xyxy[i][3], boxes.cls[i], boxes.xyxy[i]))
            else:
                cls_v_bbox.append((boxes.xyxy[i][3], boxes.cls[i], boxes.xyxy[i]))
            # cls_bbox.append((boxes.xyxy[i][3], boxes.cls[i], boxes.xyxy[i]))
            # cls_bbox.append((boxes.conf[i], boxes.cls[i], boxes.xyxy[i]))
        # cls_bbox = sorted(cls_bbox, key=lambda tup: tup[0], reverse=True)
        
        temp = []
        for i in range(len(cls_v_bbox)):
            for j in range(len(cls_lp_bbox)):
                if is_inside(cls_lp_bbox[j][2], cls_v_bbox[i][2]):
                    temp.append((min(cls_lp_bbox[j][0], cls_v_bbox[i][0]),
                                cls_v_bbox[i][1],
                                cls_lp_bbox[i][2]
                                ))
                    break
        
        ret = []
        for i in range(len(temp)):
            ret.append((temp[i][0], self.names[int(temp[i][1])], self.ocr_image(img, temp[i][2])))
        ret = sorted(ret, key=lambda tup: tup[0], reverse=True)

        # v = None
        # lp = None
        # for i in range(len(cls_bbox)):
        #     if v == None and cls_bbox[i][1] != 2:
        #         v = cls_bbox[i]
        #     if lp == None and cls_bbox[i][1] == 2:
        #         lp = cls_bbox[i]

        # if v is not None and lp is not None:
        #     if not is_inside(lp[2], v[2]):
        #         if self.orientation == 'LEFT':
        #             if lp[2][2] < v[2][2]:
        #                 v = None
        #             elif lp[2][2] > v[2][2]:
        #                 lp = None
        #         elif self.orientation == 'RIGHT':
        #             if lp[2][2] < v[2][2]:
        #                 lp = None
        #             elif lp[2][2] > v[2][2]:
        #                 v = None

        # vehicle = None
        # license_plate = None
        # if v is not None:
        #     vehicle = self.names[int(v[1])]
        # if lp is not None:
        #     license_plate = self.ocr_image(img, lp[2])

        print(ret)
        return ret
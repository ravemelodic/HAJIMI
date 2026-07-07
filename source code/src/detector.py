#Module A
import torch
from ultralytics import YOLO
from PIL import Image
from src.utils import crop_with_padding


class CatDetector:

    CAT_CLASS_ID = 15  # COCO 里 cat 是第15类

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, bbox_padding=0.15):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.bbox_padding = bbox_padding

    def detect(self, image):
        results = self.model(image, verbose=False, conf=self.conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id != self.CAT_CLASS_ID:
                    continue

                conf = float(boxes.conf[i].item())
                if conf < self.conf_threshold:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                bbox = [int(c) for c in bbox]
                crop = crop_with_padding(image, bbox, padding=self.bbox_padding)

                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "crop": crop,
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

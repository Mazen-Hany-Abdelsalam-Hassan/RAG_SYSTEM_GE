import torch
import torchvision
from src.config import  DEVICE,MODEL_DIR,GROQ_API_KEY
from src.utils import Suppress_Intersect
import numpy as np
from doclayout_yolo import  YOLOv10

class DetectionModel:
    def __init__(self):
        self.model = YOLOv10(MODEL_DIR)

    def recognize_image(self, image, conf_threshold, iou_threshold):
        det_res = self.model.predict(
            image,
            imgsz=1024,
            conf=conf_threshold,
            device=DEVICE)[0]
        boxes = det_res.__dict__['boxes'].xyxy.to('cpu')
        classes = det_res.__dict__['boxes'].cls.to('cpu')
        scores = det_res.__dict__['boxes'].conf.to('cpu')
        indices = torchvision.ops.nms(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),
                                      iou_threshold=iou_threshold)
        boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
        if len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, 0)
            scores = np.expand_dims(scores, 0)
            classes = np.expand_dims(classes, 0)
        keep_index = Suppress_Intersect(boxes)
        return boxes[keep_index], scores[keep_index], classes[keep_index]


class 

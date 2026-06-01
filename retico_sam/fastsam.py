"""
Segment Anything Module
=======================

This module provides ability to segment anything within an image and 
detect all different objects within the image. 
"""

from collections import deque
import cv2
import numpy as np
import threading 
import time 
import torch
from ultralytics import FastSAM 

import retico_core
import sys

#prefix = '../../'
#sys.path.append(prefix+'retico_vision')



from retico_vision.vision import ImageIU, DetectedObjectsIU

class FastSAMModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "SAM Object Detection Module"
    
    @staticmethod
    def description():
        return "An object detection moduel using SAM."
    
    @staticmethod
    def input_ius():
        return [ImageIU]
    
    @staticmethod
    def output_iu():
        return DetectedObjectsIU
    
    MODEL_OPTIONS = {
        "h": "vit_h",
        "l": "vit_l",
        "b": "vit_b",
    }

    def __init__(self, model=None, path_to_chkpnt=None, use_bbox=False, use_seg=False, **kwargs):
        """
        Initialize the SAM Object Detection Module
        Args:
            model (str): the name of the SAM model
                will correspond to the model checkpoint
        """
        super().__init__(**kwargs)

        if model and model.lower() in self.MODEL_OPTIONS:
            model = self.MODEL_OPTIONS[model.lower()]
            print(f"Using {model}. Make sure you have the corresponding checkpoint being passed in.")
        else: 
            print("Unknown model option. Defaulting to h (VIT-H) SAM model.")
            print("Other options include 'l' for vit_l and 'b' for vit_b.")
            model = "vit_h"

        if (use_bbox==False and use_seg==False):
            print("You may choose to use only bounding box or segmentation, please set ONLY one to true.")
            exit()

        if (path_to_chkpnt==None):
            print("Path to checkpoint matching model type must be passed in.")
            exit()

        cuda_available = torch.cuda.is_available()
      
        self.model = FastSAM(path_to_chkpnt)
        if (cuda_available):
            device = "cuda"
            self.model.to(device=device)
        self.queue = deque(maxlen=1)
        self.use_bbox = use_bbox
        self.use_seg = use_seg

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _detector_thread(self):
        while self._detector_thread_active:
            time.sleep(2)
            if len(self.queue) == 0:
                time.sleep(0.5) # original(0.5) ~ change this for more time between segmentation of each image
                continue
            
            input_iu = self.queue.popleft()
            image = input_iu.payload 

            sam_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('test_image.png', sam_image)

            cuda_available = torch.cuda.is_available()
            masks_generated = self.model.predict(sam_image, device='cuda' if cuda_available else 'cpu', retina_masks=True, conf=0.4, iou=0.9 )

            # print(masks_generated[0].keys())

            if self.use_bbox:
                valid_boxes = []  
                for box_num in range(len(masks_generated)):
                    valid_boxes = masks_generated[0].boxes.xywh.tolist() #mask bounding box in XYWH format 
            if self.use_seg:
                valid_segs = []
                for seg_num in range(len(masks_generated)):
                    valid_segs = masks_generated[0].masks.data.tolist() #segmentation masks in binary format 

            if self.use_bbox:
                if len(valid_boxes) == 0: continue
            elif self.use_seg: 
                if len(valid_segs) == 0: continue

            output_iu = self.create_iu(input_iu)
            if self.use_bbox:
                output_iu.set_detected_objects(image, valid_boxes, "bb")
            elif self.use_seg:
                output_iu.set_detected_objects(image, valid_segs, "seg")
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False

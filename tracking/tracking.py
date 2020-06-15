#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from tracking.yolo import YOLO

from tracking.deep_sort import preprocessing
from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tracking.tools import generate_detections as gdet
import imutils.video

warnings.filterwarnings('ignore')

class Tracking:
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'tracking/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    def __init__(self):
        self.yolo = YOLO()
        self.tracker = Tracker(Tracking.metric)
        # fps = 0.0
        # fps_imutils = imutils.video.FPS().start()

    def track(self, frame):
        # t1 = time.time()
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxs = self.yolo.detect_image(image)[0]
        confidence = self.yolo.detect_image(image)[1]

        features = Tracking.encoder(frame,boxs)

        self.detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in self.detections])
        scores = np.array([d.confidence for d in self.detections])
        indices = preprocessing.non_max_suppression(boxes, Tracking.nms_max_overlap, scores)
        self.detections = [self.detections[i] for i in indices]
        
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(self.detections)
        
        # for track in self.tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue 
        #     bbox = track.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        #     cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        # for det in self.detections:
        #     bbox = det.to_tlbr()
        #     score = "%.2f" % round(det.confidence * 100, 2)
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        #     cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
            
        # cv2.imshow('', frame)

        # fps_imutils.update()

        # fps = (fps + (1./(time.time()-t1))) / 2
        # print("FPS = %f"%(fps))

        # fps_imutils.stop()
        # print('imutils FPS: {}'.format(fps_imutils.fps()))
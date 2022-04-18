from tensorflow.keras.utils import image_dataset_from_directory
from object_detection import get_bounding_box, plot_results, plot_bounding_box, iou, get_json_fname, get_iou_metrics, accuracy, plot_bounding_box_i
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import tensorflow_addons as tfa
from tf2_yolov4.model import YOLOv4
from tf2_yolov4.anchors import YOLOV4_ANCHORS
import tensorflow as tf

print(tf.__version__)


HEIGHT, WIDTH = (640, 960)


model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=100,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)

model.load_weights("./yolov4.h5")

root_dir = Path('ELPephant/images')

tfds = image_dataset_from_directory(str(root_dir), labels=None,
                                    image_size=(HEIGHT, WIDTH),
                                    batch_size=32,
                                    shuffle=False).map(tf.keras.layers.Rescaling(1/255.0))

boxes, scores, classes, valid_detections = model.predict(tfds.take(3))
print(boxes.shape, valid_detections)

# elp_detected_indices = np.where(classes == 20)  # COCO dataset class for elephant - 20
# elp_detected_thresh_indices = elp_detected_indices[0][scores[elp_detected_indices] > 0.5]

inference_data = {'boxes': boxes,
                  'scores': scores,
                  'classes': classes,
                  'valid_detections': valid_detections}

with open('inference_simple_resize.pickle', 'wb') as f:
    pickle.dump(inference_data, f)

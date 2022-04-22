import tensorflow as tf

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import tensorflow_addons as tfa
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HEIGHT, WIDTH = (640, 960)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def get_bounding_box(fname, normalize=False):
    '''
    returns [x_min, y_min, x_max, y_max], (H, W)
    '''
    with open(fname) as f:
        data = f.read()
    json_data = json.loads(data)
    H, W = json_data['imageHeight'], json_data['imageWidth']

    shapes = json_data.get('shapes', [])
    for shape in shapes:
        if shape.get('shape_type', '') == 'rectangle':
            bb_coords = np.array(shape['points']).flatten()

            if normalize:
                xmin, ymin, xmax, ymax = bb_coords
                return np.array([xmin/W, ymin/H, xmax/W, ymax/H]), (H, W)
            else:
                return bb_coords, (H, W)


# %config InlineBackend.figure_format = 'retina'

def plot_results(pil_img, boxes, scores, classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=COLORS[int(cl) % 6], linewidth=3))
            text = f'{CLASSES[cl]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def plot_bounding_box(img, boxes, labels=None, normalized=False):
    if labels is None:
        labels = ['box-'+str(i) for i in range(1, len(boxes)+1)]
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    ax = plt.gca()

    H, W = img.shape[:2]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if normalized:
            xmin, ymin, xmax, ymax = np.array(box) * [W, H, W, H]
        else:
            xmin, ymin, xmax, ymax = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=COLORS[i % len(COLORS)], linewidth=3))
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))


def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_union_min, y_union_min = min(x1_min, x2_min), min(y1_min, y2_min)
    x_union_max, y_union_max = max(x1_max, x2_max), max(y1_max, y2_max)
    union_area = (x_union_max - x_union_min) * (y_union_max - y_union_min)

    x_intersection_min, y_intersection_min = max(x1_min, x2_min), max(y1_min, y2_min)
    x_intersection_max, y_intersection_max = min(x1_max, x2_max), min(y1_max, y2_max)
    intersection_area = (x_intersection_max - x_intersection_min) * (y_intersection_max - y_intersection_min)

    return intersection_area/union_area


def get_json_fname(fname, root_dir):
    return root_dir / fname.parent.name / (fname.stem + '.json')


def get_iou_metrics(elp_detected_thresh_indices, img_fnames, classes, boxes, json_root_dir):
    '''
    doesn't take into account multiple elephants per image

    computes giou and iou of pred and ground truth bounding boxes
    '''
    N = len(img_fnames)
    metrics = np.zeros((N, 2))
    for i in elp_detected_thresh_indices:
        fname = img_fnames[i]
        json_fname = get_json_fname(fname, json_root_dir)
        if json_fname.exists():
            y_true, (H, W) = get_bounding_box(json_fname, normalize=True)

            elp_box_index = np.where(classes[i] == 20)[0][0]  # first elephant bounding box index
            y_pred = boxes[i, elp_box_index]

            metrics[i, :] = np.array([1-tfa.losses.giou_loss(y_true, y_pred).numpy(), iou(y_true, y_pred)])
        else:
            print(json_fname, 'does not exist')
    return metrics


def accuracy(metrics, N, iou_thresh=0.4):
    return np.sum(metrics[:, 0] > iou_thresh)/N


def plot_bounding_box_i(i, img_fnames, classes, boxes, json_root_dir):
    fname = img_fnames[i]
    img = plt.imread(fname)
    json_fname = get_json_fname(fname, json_root_dir)
    print(json_fname)
    y_true, (H, W) = get_bounding_box(json_fname, normalize=True)

    elp_box_index = np.where(classes[i] == 20)[0][0]  # first elephant bounding box index
    y_pred = boxes[i, elp_box_index]

    plot_bounding_box(img, [y_true, y_pred], ['true', 'pred'], normalized=True)

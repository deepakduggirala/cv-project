from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pathlib import Path
import argparse


def is_valid_bbox(bbox, area_thresh=150*150, aspect_thresh=1.5):
    x, y, w, h = bbox
    return w*h >= area_thresh and max(w, h)/min(w, h) <= aspect_thresh


def fetch_and_save(img, data_dir, catIds, area_thresh=150*150, aspect_thresh=1.67):
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    valid_anns = [ann for ann in anns if is_valid_bbox(ann['bbox'], area_thresh, aspect_thresh)]

    if len(valid_anns) > 0:
        I = io.imread(img['coco_url'])

        for i, ann in enumerate(valid_anns):
            x, y, w, h = bbox = ann['bbox']
            crop = I[int(y):int(y+h), int(x):int(x+w)]
            out_file = Path(data_dir) / f"{img['id']}_{i}.png"
            io.imsave(out_file, crop)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train',
                        help="train or val")
    args = parser.parse_args()

dataDir = '/Users/deepakduggirala/Documents'
dataType = args.mode+'2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

# catIds = coco.getCatIds(catNms=['elephant'])
# imgIds = coco.getImgIds(catIds=catIds)
# imgs = coco.loadImgs(ids=imgIds)
# data_dir = f'/Users/deepakduggirala/Documents/coco-dataset/{args.mode}/elephants'


data_dir = f'/Users/deepakduggirala/Documents/coco-dataset/{args.mode}/others'
np.random.seed(0)
for catId in [20, 21]:
    imgIds = coco.getImgIds(catIds=catId)
    print(len(imgIds), catId)
    imgs = np.array(coco.loadImgs(ids=imgIds))
    if args.mode == 'train':
        idxs = np.random.randint(low=0, high=len(imgs), size=700)
    else:
        idxs = np.random.randint(low=0, high=len(imgs), size=30)

    for i, img in enumerate(imgs[idxs]):
        print(f'fetching {i:04d} of {len(idxs):04d}')
        fetch_and_save(img, data_dir, catId)

import sys
import os

import cv2
import json
import numpy as np
import h5py
import math
import scipy
import scipy.spatial

from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_COCO(ann_path,out_length):
    with open(ann_path, 'r') as f:
        dataset = json.load(f)
    images = {image.get('id', None): {
        'file_name': image.get('file_name', ''),
        'height': image.get('height', ''),
        'width': image.get('width', ''),
    } for image in dataset.get('images', [])}
    annotations = dataset.get('annotations', [])
    annos = {}

    for index, annotation in enumerate(annotations):
        annotation_image_id = annotation.get('image_id')

        height = images[annotation_image_id].get('height')
        width = images[annotation_image_id].get('width')
        scale_factor = out_length/max(height,width)

        if annotation_image_id not in annos:
            annos[annotation_image_id] = {}
        objects = annos[annotation_image_id].get('objects', [])
        segmentations = annotation.get('segmentation')
        for segmentation in segmentations:
            xs = segmentation[::2]
            ys = segmentation[1::2]
            points = [[x, y] for x, y in zip(xs, ys)]
            objects.append((np.array(points).astype(float)*scale_factor).astype(int).tolist())
        annos[annotation_image_id]['objects'] = objects

    return images,annos

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * max(radius[0],radius[1]) + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = cv2.resize(gaussian,dsize=(radius[0]*2+1,radius[1]*2+1),interpolation=cv2.INTER_LINEAR)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
    top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius[1] - top:radius[1] + bottom, radius[0] - left:radius[0] + right]
    '''
    plt.subplot(121)
    plt.imshow(gaussian)
    plt.subplot(122)
    plt.imshow(masked_gaussian)
    plt.show(block=True)
    '''
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

if __name__ == '__main__':
    base_dir = sys.path[0]
    out_length = 2048

    ann_path = os.path.join(base_dir,'annotations_COCO.json')
    images,annos = parse_COCO(ann_path,out_length)

    for index, (image_id, values) in enumerate(annos.items()):
        img_name = images[image_id].get('file_name')
        height = images[image_id].get('height')
        width = images[image_id].get('width')
        scale_factor = out_length/max(height,width)
        berry_targets = values.get('objects')
        #print(index,img_name,height,width,len(berry_targets))

        img = cv2.imdecode(np.fromfile(os.path.join(base_dir,'images',img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(int(width*scale_factor), int(height*scale_factor)))

        final_probmap = np.zeros((img.shape[0],img.shape[1]))
        for berry_target in tqdm(berry_targets):
            x1,y1 = np.min(np.array(berry_target)[:,0]),np.min(np.array(berry_target)[:,1])
            x2,y2 = np.max(np.array(berry_target)[:,0]),np.max(np.array(berry_target)[:,1])
            if x2 == int(img.shape[1]):
                x2-=1
            if y2 == int(img.shape[0]):
                y2-=1
            h, w = y2 - y1, x2 - x1
            radius = (math.ceil(w/2),math.ceil(h/2))
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            final_probmap = draw_gaussian(final_probmap, ct_int, radius)
            '''
            plt.imshow(final_probmap)
            plt.show(block=True)
            '''
            
        with h5py.File(os.path.join(base_dir,'probmaps_bbox',img_name).replace('.jpg','.h5'), 'w') as hf:
            hf['probmap'] = final_probmap
            hf['count'] = len(berry_targets)
        cv2.imencode('.jpg', final_probmap*255)[1].tofile(os.path.join(base_dir,'probmaps_bbox',img_name))
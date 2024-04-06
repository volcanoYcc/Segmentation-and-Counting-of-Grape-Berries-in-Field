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

def gaussian_filter_prob(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048

    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    for i in tqdm(range(pts.shape[0])):
        pt = pts[i]
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        filter = scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[pt[1]][pt[0]]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    return density

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

        temp_pointmap = np.zeros((img.shape[0],img.shape[1]))
        for berry_target in berry_targets:
            x1,y1 = np.min(np.array(berry_target)[:,0]),np.min(np.array(berry_target)[:,1])
            x2,y2 = np.max(np.array(berry_target)[:,0]),np.max(np.array(berry_target)[:,1])
            if x2 == int(img.shape[1]):
                x2-=1
            if y2 == int(img.shape[0]):
                y2-=1
        
            temp_pointmap[int((y2+y1)/2),int((x2+x1)/2)]=1
        final_probmap = gaussian_filter_prob(temp_pointmap)

        with h5py.File(os.path.join(base_dir,'probmaps_point',img_name).replace('.jpg','.h5'), 'w') as hf:
            hf['probmap'] = final_probmap
            hf['count'] = len(berry_targets)
        cv2.imencode('.jpg', final_probmap*255)[1].tofile(os.path.join(base_dir,'probmaps_point',img_name))
        
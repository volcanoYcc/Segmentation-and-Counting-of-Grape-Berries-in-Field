import sys
import os

import cv2
import json
import numpy as np
import h5py
import math
import torch
import torch.nn as nn

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

def gaussian(r,ind):
    x = np.ogrid[-r:r + 1]
    diameter = 2 * r
    sigma = diameter / 6
    h = np.exp((-x*x/(2 * sigma * sigma)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    if sigma==0:
        print(h[int(r+ind)])
    return h[int(r+ind)]

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
            #计算颗粒区域内每一个点到边框的最短距离
            berry_area_target_temp1 = np.zeros((img.shape[0],img.shape[1]))
            cv2.fillPoly(berry_area_target_temp1, np.array([berry_target]), 1)
            t = np.zeros((y2-y1+1,x2-x1+1))
            boundary_dist = np.zeros((y2-y1+1,x2-x1+1))
            t[:,:] = berry_area_target_temp1[y1:y2+1,x1:x2+1]
            dist = 1
            while True:
                t = torch.from_numpy(t).unsqueeze(0).cuda()
                avg_pooled_parts_probmap = nn.functional.avg_pool2d(t, 3, stride=1, padding=1).cpu().numpy()[0]
                boundary_dist = np.where((avg_pooled_parts_probmap>0.0)&(avg_pooled_parts_probmap<1.0)&(t.cpu().numpy()[0]==1), dist, boundary_dist)
                t = np.where(avg_pooled_parts_probmap==1, 1.0, 0)
                if np.sum(t) == 0:
                    break
                dist+=1
            #对于每一个颗粒区域内的点，计算其到所有其它点的距离
            boundary_list = []
            wh_list = []
            im_wh_list = []
            boundary_list.append(np.array(berry_target).astype(float))
            wh_list.append((x1,y1,x2,y2))
            im_wh_list.append((img.shape[0],img.shape[1]))
            while True:
                boundary_list.append(boundary_list[-1]/2)
                x1,y1,x2,y2 = wh_list[-1][0]/2,wh_list[-1][1]/2,wh_list[-1][2]/2,wh_list[-1][3]/2
                wh_list.append((int(x1),int(y1),int(x2),int(y2)))
                im_wh_list.append((im_wh_list[-1][0]/2,im_wh_list[-1][1]/2))
                if x2-x1<=10 or y2-y1<=10:
                    break
            for i in range(len(boundary_list)):
                boundary_list[i] = boundary_list[i].astype(int).tolist()
                im_wh_list[i] = (int(im_wh_list[i][0]),int(im_wh_list[i][1]))
            boundary_list.reverse()
            wh_list.reverse()
            im_wh_list.reverse()
            for i in range(len(boundary_list)):
                x1,y1,x2,y2 = wh_list[i]
                h = y2-y1
                w = x2-x1
                berry_area_target_temp2 = np.zeros(im_wh_list[i])
                cv2.fillPoly(berry_area_target_temp2, np.array([boundary_list[i]]), 1)
                tt = np.zeros((h+1,w+1))
                tt[:,:] = berry_area_target_temp2[y1:y2+1,x1:x2+1]
                if i==0:
                    candidate_peaks = np.array(np.nonzero(np.where(tt==1, 1.0, 0))).T
                peak_dist_sums = []
                for k in range(len(candidate_peaks)):
                    candidate_peak = candidate_peaks[k]
                    if i!=0:
                        if candidate_peak[0] > tt.shape[0]-1 or candidate_peak[1] > tt.shape[1]-1 or candidate_peak[0] < 0 or candidate_peak[1] < 0:
                            peak_dist_sums.append(999999999)
                            continue
                        if tt[candidate_peak[0]][candidate_peak[1]] == 0.0:
                            peak_dist_sums.append(999999999)
                            continue
                    peak_dist = np.zeros((h+1,w+1))
                    peak_area = np.zeros((h+1,w+1))
                    peak_dist[candidate_peak[0]][candidate_peak[1]] = 1.0
                    peak_area[candidate_peak[0]][candidate_peak[1]] = 1.0
                    dist = 2
                    while True:
                        peak_area_temp = torch.from_numpy(peak_area).unsqueeze(0).cuda()
                        peak_area_temp = nn.functional.avg_pool2d(peak_area_temp, 3, stride=1, padding=1).cpu().numpy()[0]
                        peak_dist = np.where((peak_area_temp>0)&(peak_area==0)&(tt==1), dist, peak_dist)
                        peak_area = np.where((peak_area_temp>0)&(tt==1), 1.0, 0)
                        if np.sum(tt) == np.sum(peak_area):
                            break
                        dist+=1
                    peak_dist_sums.append(np.sum(peak_dist))
                min_dist = min(peak_dist_sums)
                min_idx = peak_dist_sums.index(min_dist)
                candidate_peak = candidate_peaks[min_idx]
                p,q = candidate_peak
                candidate_peaks = ((2*p-1,2*q-1),(2*p,2*q-1),(2*p+1,2*q-1),(2*p-1,2*q),(2*p,2*q),(2*p+1,2*q),(2*p-1,2*q+1),(2*p,2*q+1),(2*p+1,2*q+1))
            peak_dist = np.zeros((y2-y1+1,x2-x1+1))
            peak_area = np.zeros((y2-y1+1,x2-x1+1))
            peak_dist[candidate_peak[0]][candidate_peak[1]] = 1.0
            peak_area[candidate_peak[0]][candidate_peak[1]] = 1.0
            dist = 2
            while True:
                peak_area_temp = torch.from_numpy(peak_area).unsqueeze(0).cuda()
                peak_area_temp = nn.functional.avg_pool2d(peak_area_temp, 3, stride=1, padding=1).cpu().numpy()[0]
                peak_dist = np.where((peak_area_temp>0)&(peak_area==0)&(tt==1), dist, peak_dist)
                peak_area = np.where((peak_area_temp>0)&(tt==1), 1.0, 0)
                if np.sum(tt) == np.sum(peak_area):
                    break
                dist+=1

            #根据前两步结果绘制非规则形状的二维高斯核
            probmap = np.zeros((y2-y1+1,x2-x1+1))
            for i in range(len(probmap)):
                for j in range(len(probmap[0])):
                    if tt[i][j]!=0:
                        probmap[i][j] = gaussian(peak_dist[i][j]+boundary_dist[i][j]-1,peak_dist[i][j]-1)
                        #probmap[i][j] = gaussian(math.ceil(peak_dist[i][j]/2)+boundary_dist[i][j]-1,math.ceil(peak_dist[i][j]/2)-1)

            '''
            plt.subplot(221)
            plt.imshow(boundary_dist)
            plt.subplot(222)
            plt.imshow(peak_dist)
            plt.subplot(223)
            plt.imshow(probmap)
            plt.show(block=True)
            '''
            
            #print(np.max(probmap))
            berry_prob_target = np.zeros((img.shape[0],img.shape[1]))
            berry_prob_target[y1:y2+1,x1:x2+1] = probmap

            np.maximum(final_probmap,berry_prob_target,out=final_probmap)
            #plt.imshow(final_probmap)
            #plt.show(block=True)
        with h5py.File(os.path.join(base_dir,'probmaps_mask',img_name).replace('.jpg','.h5'), 'w') as hf:
            hf['probmap'] = final_probmap
            hf['count'] = len(berry_targets)
        cv2.imencode('.jpg', final_probmap*255)[1].tofile(os.path.join(base_dir,'probmaps_mask',img_name))
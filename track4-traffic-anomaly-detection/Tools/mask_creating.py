#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:21:20 2019

@author: peterluong
"""

import json
import sys
import os
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir + '/..')
import Config
from vid_utils import natural_keys


def find_first_pos(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                return i,j
    return -1,-1

def expand(mask, region, check):

    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    l = 0
    h, w = mask.shape
    ret = 1
    while l < len(region):
        u = region[l]
        for i in range(0, 4):
            v = (u[0] + dx[i], u[1] + dy[i])
            if 0 <= v[0] < h and 0 <= v[1] < w:
                if check[v[0]][v[1]] == 0 and mask[v[0]][v[1]]:
                    check[v[0]][v[1]] = 1
                    region.append(v)
                    ret += 1
        l += 1
    return ret

def region_extract(mask, threshold_s = 2000):
    check = np.zeros_like(mask, dtype=np.int)
    h, w = mask.shape
    for i in range(0, h):
        for j in range(0, w):
            if check[i][j] == 0 and mask[i][j]:
                u = (i, j)
                check[u[0]][u[1]] = 1
                region = [(u[0], u[1])]
                s = expand(mask, region, check)
                if s < threshold_s:
                    for u in region: mask[u[0]][u[1]] = 0

    return mask

def extractMask(vid: Path) -> None:
    capture = cv2.VideoCapture(str(vid))
    scenes = json.load(open(Config.data_path + '/unchanged_scene_periods.json'))

    scenes_id = 0
    for each in scenes[vid.stem]:
        scenes_id += 1
        print(vid.stem, scenes_id)
        start = each[0]
        end = each[1]
        print(start, end)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        success, frame = capture.read()

        temp = np.zeros_like(frame)

        capture.set(cv2.CAP_PROP_POS_FRAMES, (start + end) // 2)
        _, mid_frame = capture.read()

        frame_id = start + 30
        while frame_id < end:
            prev = frame

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = capture.read()
            frame_id += 30

            if success:
                sub = sc.expit(np.abs(frame - prev) - 125)
                temp = temp + sub
            else:
                break
        temp = np.sum(temp, 2)
        mask = (temp > 0.2).astype(np.uint8)

        np.save('masks_refine_non_expand/mask_{0}_{1}.npy'.format(vid.stem, scenes_id), mask)

        for count in range(2):
            mask2 = np.zeros_like(mask)
            for i in range(1, mask.shape[0] - 1):
                for j in range(1, mask.shape[1] - 1):
                    mask2[i,j] = max([mask[i-1,j], mask[i,j], mask[i+1,j],
                            mask[i-1,j-1], mask[i,j-1], mask[i+1,j-1],
                            mask[i-1,j+1], mask[i,j+1], mask[i+1,j+1]])
            mask = mask2

        mask = region_extract(mask.copy(), threshold_s=2000)

        for count in range(15):
            mask2 = np.zeros_like(mask)
            for i in range(1, mask.shape[0] - 1):
                for j in range(1, mask.shape[1] - 1):
                    mask2[i,j] = max([mask[i-1,j], mask[i,j], mask[i+1,j],
                            mask[i-1,j-1], mask[i,j-1], mask[i+1,j-1],
                            mask[i-1,j+1], mask[i,j+1], mask[i+1,j+1]])
            mask = mask2

        mask = cv2.blur(mask, (5,5))
        np.save('masks_refine_v3/mask_{0}_{1}.npy'.format(vid.stem, scenes_id), mask)
        imageio.imwrite('masks/{0}_%{1}.jpg'.format(vid.stem, scenes_id),
                        mask.reshape(410,800,1).astype(np.uint8) * mid_frame)

def verifyMask(video_id, scene_id, expand):
    if not expand:
        mask = np.load('./masks_refine_non_expand/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        cv2.imwrite('./mask_ne.png', mask * 255)
    else:
        mask = np.load('./masks_refine_v3/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        cv2.imwrite('./mask.png', mask * 255)
    plt.imshow(mask, cmap='gray')
    plt.show()

def expandMask(video_id, scene_id):
    mask_path = Config.data_path + '/masks_refine_v3/' + 'mask_' + str(video_id) + '_' + str(scene_id) + '.npy'
    mask = np.load(mask_path)
    for count in range(4):
        mask2 = np.zeros_like(mask)
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                mask2[i, j] = max([mask[i - 1, j], mask[i, j], mask[i + 1, j],
                                   mask[i - 1, j - 1], mask[i, j - 1], mask[i + 1, j - 1],
                                   mask[i - 1, j + 1], mask[i, j + 1], mask[i + 1, j + 1]])
        mask = mask2

    np.save(mask_path, mask)


if __name__ == '__main__':

    # Extract the mask
    # [6,11,12,17,20,22,24,26,27,28,32,34,35,44,50,51,55,59,64,66,71,77,79,82,85,90,96,97]:
    videos = Path(Config.dataset_path).glob('*.mp4')
    videos = sorted(list(videos), key=natural_keys)
    for vid in videos:
        extractMask(vid)
        #visualize extracted masks
        verifyMask(video_id=vid.stem, scene_id=1, expand=True)

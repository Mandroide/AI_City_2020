import json
import sys
import os

from typing import Dict, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir + '/..')
import Config

class RoadMask:
    def __init__(self, mask_path, scene_path, im_path):
        self.mask_path = mask_path
        self.scene_path = scene_path
        self.im_path = im_path
        with open(scene_path, 'r') as f:
            self.stableList: Dict[str, List[List[int]]] = json.load(f)
        self.refineMasks()

    def getMask(self, video_id, scene_id):
        mask = np.load(self.mask_path + '/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        return mask

    @staticmethod
    def __refineMask(im, mask):
        mask = mask > 0.3
        mask = (mask * 255).astype(int)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 0)
        im = cv2.bilateralFilter(im, 9, 75, 75)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(im, cmap='gray')
        ax2.imshow(mask, cmap='gray')
        ax3.imshow(cv2.Canny(im, 0, 0), cmap='gray')
        plt.show()


    def refineMasks(self):
        for video_id, stableIntervals in self.stableList.items():
            for scene_id in range(1, len(stableIntervals) + 1):
                l, _ = stableIntervals[scene_id - 1]
                l = int(l / Config.fps) + 1
                mask = self.getMask(video_id, scene_id)
                print(self.im_path + '/' + str(video_id) + '/average' + str(l) + '.jpg')
                im = cv2.cvtColor(cv2.imread(self.im_path + '/' + str(video_id) + '/average' + str(l+5) + '.jpg'), cv2.COLOR_BGR2RGB)
                self.__refineMask(im, mask)
                break

        print(self.stableList)

if __name__ == '__main__':
    list = RoadMask(Config.data_path + '/masks', Config.data_path + '/unchanged_scene_periods.json', Config.avg_im_path)
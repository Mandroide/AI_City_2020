import json
import os

import cv2
from ... import Config
from ...vid_utils import natural_keys
from pathlib import Path

dataset_cuts = {}
dirname = Path(Config.stop_cuts_dir)
cuts_files = sorted(list(dirname.glob('*.json')), key=natural_keys)
for cuts_file in cuts_files:
    with open(cuts_file, 'r') as f:
        cuts = json.load(f)
    basename = ''.join(cuts_file.name.split('.')[:-1])

    filename = Path(Config.dataset_path).with_name(basename)
    vid = cv2.VideoCapture(filename)
    num_frms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()

    cur_frm = 0

    dataset_cuts[basename] = []

    i = 0
    while i < len(cuts):
        cut = cuts[i] / 30
        start_cut = cut

        duration = 0
        while i + 1 < len(cuts):
            end_cut = cuts[i+1] / 30
            if end_cut > cut + 1:
                break
            duration += 1
            i += 1
            cut = end_cut
        print(basename, start_cut, duration)

        if duration > 0:
            dataset_cuts[basename].append((start_cut-1, start_cut + duration))
        i += 1

    print(basename, len(cuts), dataset_cuts[basename])

with open(os.path.join(Config.data_path, 'stop_scene_periods.json'), 'w') as f:
    json.dump(dataset_cuts, f)

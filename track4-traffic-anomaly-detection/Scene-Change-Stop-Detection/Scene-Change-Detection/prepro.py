import json
import os
from pathlib import Path

import cv2
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir + '/../..')
import Config
from vid_utils import natural_keys

dataset_cuts = {}
dirname = Path(Config.cuts_dir)
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
    for cut in cuts.values():
        if cur_frm < cut - 5*30:
            dataset_cuts[basename].append((cur_frm, cut - 30))
        cur_frm = cut + 30
    if cur_frm < num_frms - 5*30:
        dataset_cuts[basename].append((cur_frm, num_frms))
    print(basename, len(cuts), dataset_cuts[basename])

with open(os.path.join(Config.data_path, 'unchanged_scene_periods.json'), 'w') as f:
    json.dump(dataset_cuts, f)

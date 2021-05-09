import argparse
import glob
import json
import os
import sys

import cv2
from tqdm import tqdm
from pathlib import Path
from ... import Config
from ..vid_utils import LBP


def getCuts(file_name: Path, cap: cv2.VideoCapture) -> None:
    begin_id = 0
    thresh = 2000

    cuts = []
    cnt=0
    #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('frame', 600, 350)
    #cv2.namedWindow('background frame', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('background frame', 600, 350)

    for frm_id in tqdm(range(begin_id, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 30)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
        _, frame = cap.read()


        try:
            # Our operations on the frame come here
            lbp = LBP(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            lbph = cv2.calcHist([lbp],[0],None,[256],[0,256])

            if frm_id == 0:
                prev_lbph = lbph
                continue

            diff=0
            for i in range(256):
                diff += abs(lbph[i]-prev_lbph[i])

            #print frm_id, diff[0]
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            if diff[0] <= thresh:
                cnt+=1
                print("CUT " + str(cnt) + " Detected at frame " + str(frm_id))
                cuts.append(frm_id)

            prev_lbph = lbph

        except UnboundLocalError:
            pass

    print('Found %d scene changes.' % cnt)
    cuts_file = os.path.join(os.path.dirname(file_name), 'stop_cuts', os.path.basename(file_name) + '.json')
    with open(cuts_file, 'w') as f:
        json.dump(cuts, f)

    cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects all scene changing periods in videos.')
    parser.add_argument('vi_or_dir',
                        help='Videos or directory containing videos to be processed.',
                        nargs='+',
                        type=str, default=Config.dataset_path)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    videos = args.vi_or_dir

    if len(videos) > 1:
        print([os.path.isdir(video) for video in videos])
        assert all([os.path.isfile(video) for video in videos]), 'Multiple inputs option is only for inputting videos.'
    assert all([os.path.dirname(video) == os.path.dirname(videos[0]) for video in videos]), 'All videos should be placed in the same directory.'

    if len(videos) == 1 and os.path.isdir(videos[0]):
        videos = Path(videos[0]).glob('*.mp4')

    stop_cuts_dir = Path(Config.stop_cuts_dir)
    stop_cuts_dir.mkdir(parents=True, exist_ok=True)

    for video_name in videos:
        cap = cv2.VideoCapture(str(video_name))
        print('Processing file name: %s' % str(video_name))
        getCuts(video_name, cap)

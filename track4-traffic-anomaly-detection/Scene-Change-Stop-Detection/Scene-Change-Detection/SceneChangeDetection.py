import argparse
import json
import os
import sys

import cv2
from tqdm import tqdm
from pathlib import Path
from ... import Config
from ...vid_utils import LBP


def getCuts(file_name: Path, cap: cv2.VideoCapture) -> None:
    begin_id = 0
    background_alpha = 0.01
    alpha = 0.1
    thresh = 70000

    cuts = []
    cnt = 0
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 600, 350)
    cv2.namedWindow('background frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('background frame', 600, 350)
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin_id)

    _, background_frame = cap.read()
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2YUV)
    background_frame[:, :, 0] = cv2.equalizeHist(background_frame[:, :, 0])
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_YUV2BGR)
    mean_frame = background_frame.copy()

    for frm_id in tqdm(range(begin_id + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)

        background_frame = cv2.addWeighted(background_frame, 1 - background_alpha, frame, background_alpha, 0)
        mean_frame = cv2.addWeighted(mean_frame, 1 - alpha, frame, alpha, 0)

        # Our operations on the frame come here
        if frm_id % 30 == 0:
            lbp = LBP(cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY))
            background_lbp = LBP(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY))

            try:
                lbph = cv2.calcHist([lbp], [0], None, [256], [0, 256])
                background_lbph = cv2.calcHist([background_lbp], [0], None, [256], [0, 256])

                diff = 0
                for i in range(256):
                    diff += abs(lbph[i] - background_lbph[i])

                print(frm_id, diff)

                cv2.imshow('frame', mean_frame)
                cv2.imshow('background frame', background_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if diff[0] >= thresh:
                    cnt += 1
                    print("CUT " + str(cnt) + " Detected at frame " + str(frm_id))
                    cuts.append(frm_id)
                    background_frame = frame.copy()

            except UnboundLocalError:
                pass

    print('Found %d scene changes.' % cnt)
    cuts_file = Config.cuts_dir / file_name.with_suffix('.json')
    with open(cuts_file, 'w') as f:
        json.dump(cuts, f)

    cap.release()
    # cv2.destroyAllWindows()


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
        assert any([Path(video).is_dir() for video in videos]), 'Multiple inputs option is only for inputting videos.'
    assert all([Path(video).parent == Path(videos[0]).parent for video in
                videos]), 'All videos should be placed in the same directory.'

    if len(videos) == 1 and os.path.isdir(videos[0]):
        videos = Path(videos[0]).glob('*.mp4')

    cuts_dir = Path(Config.cuts_dir)
    cuts_dir.mkdir(parents=True, exist_ok=True)

    for video_name in videos:
        cap = cv2.VideoCapture(str(video_name))
        print('Processing file name: %s' % str(video_name))
        getCuts(video_name, cap)

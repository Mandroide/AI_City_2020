from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import Config
from AnomalyDetector import AnomalyDetector
from Detectors import DetectorDay, DetectorNight, DayNightDetector
from MaskList import MaskList
from Misc import Image
from StableFrameList import StableFrameList
from vid_utils import natural_keys

#Initialize detector
print('Parse detector result ...')
dayNightDetector = DayNightDetector()
detectorDay = DetectorDay(Config.data_path + '/result_8_3_3_clas.txt', Config.data_path + '/result_8_3_3_nclas.txt')
detectorNight = DetectorNight(Config.data_path + '/extracted-bboxes-dark-videos')
anomalyDetector = AnomalyDetector()
stableList = StableFrameList(Config.data_path + '/unchanged_scene_periods.json')
maskList = MaskList(Config.data_path + '/masks_refine_v3')

videos_path = Path(Config.dataset_path).glob('*.mp4')
videos_path = sorted(list(videos_path), key=natural_keys)
output_dir = Path(Config.output_path)
for video in videos_path:
    video_id = int(str(video.stem))
    print("Processing video ", video.stem)
    detector = detectorDay
    if dayNightDetector.checkNight(video_id):
        detector = detectorNight

    stableIntervals = stableList[video_id]
    print(stableIntervals)
    confs = {}
    print(detector.name)

    #anomaly save file
    video_folder = output_dir / video.stem
    video_folder.mkdir(exist_ok=True, parents=True)
    f = open(video_folder / 'anomaly_events.txt', 'w')

    #output video of the detected anomaly events
    video_input = cv2.VideoCapture(str(video))
    width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_input.release()
    video_output_path = video_folder / 'anomaly_event.avi'
    video_output = cv2.VideoWriter(str(video_output_path), cv2.VideoWriter_fourcc(*'XVID'), Config.fps, (width, height))

    #loop all stable intervals
    for scene_id in range(1, len(stableIntervals) + 1):
        l, r = stableIntervals[scene_id - 1]
        sl = int(l / Config.fps) + 1
        sr = int(r / Config.fps)
        sceneMask = maskList[(video_id, scene_id)]

        #create output folder
        scene_folder = video_folder / str(scene_id)
        scene_folder.mkdir(exist_ok=True, parents=True)

        # output folder: output / video_id / scene_id / stuffs
        # output: average + boxes, gray_boxes before, gray_boxes after mask

        for frame_id in range(sl, sr):
            ave_im = Image.load(Config.avg_im_path + '/' + str(video_id) + '/average' + str(frame_id) + '.jpg')
            boxes = detector.detect(video_id, frame_id)
            for box in boxes:
                box.applyMask(sceneMask)

            box_im = Image.addBoxes(ave_im, boxes)

            if detector.name == 'night':
                Image.save(box_im, str(scene_folder/('night_average' + format(frame_id, '03d') + '.jpg')))

            else:
                Image.save(box_im, str(scene_folder/('day_average' + format(frame_id, '03d') + '.jpg')))

            #detect anomaly event in scene
            anomalyDetector.addBoxes(boxes, frame_id) #input detected boxes => list of anomaly event
            detectedAnomalyEvents, conf = anomalyDetector.examineEvents(video_id, scene_id, frame_id, frame_id == sr - 1, f)

            event_im = anomalyDetector.drawEvents(box_im)

            Image.save(event_im, scene_folder/('events' + format(frame_id, '03d') + '.jpg'))
            confs[frame_id] = conf

            video_output.write(event_im)

    f.close()
    #release resources
    video_output.release()
    cv2.destroyAllWindows()

    #output anomaly graph text before, anomaly_graph_after, anomaly_graph before, anomaly_graph after, result metric
    print(confs)
    f, ax = plt.subplots()
    ax.plot(list(confs.keys()), list(confs.values()), lw=4)
    ax.set_xlabel('Time')
    ax.set_xlim(left=0)
    ax.set_ylabel('Confidence')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    f.savefig(video_folder/(str(video_id) + '_anomaly.pdf'), bbox_inches='tight')
    plt.close(f)
    f = open(video_folder/(str(video_id) + '_anomaly.txt'), 'w')
    for frame_id, conf in confs.items():
        f.write(str(frame_id) + ' ' + str(conf) + '\n')
    f.close()

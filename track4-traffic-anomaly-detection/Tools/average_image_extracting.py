from pathlib import Path
from .. import Config
import imageio
import numpy as np

alpha = 0.01

output_path = Path(Config.avg_im_path)
dataset_path = Path(Config.dataset_path).glob('*.mp4')
for video_file in dataset_path:
    reader = imageio.get_reader(video_file, 'ffmpeg')
    meta_data = reader.get_meta_data()
    length = int(meta_data['duration']*meta_data['fps'])

    frame = reader.get_data(0)
    
    average = frame

    for i in range(1, length):
        prev = frame
        frame = reader.get_data(i)
        if i % 3600 == 0:
            print(i) #print process

        ## calculating average image
        average = (1 - alpha)*average + alpha*frame
        if i % 30 == 0:
            second = i // 30
            path = output_path / video_file.stem
            path.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(path / "average{}.jpg".format(second), average.astype(np.uint8))

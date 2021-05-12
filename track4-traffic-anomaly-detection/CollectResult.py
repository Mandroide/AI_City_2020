import Config
from pathlib import Path
from vid_utils import natural_keys


g = open(Config.output_path + '/result_all.txt', 'w')
output_dirs = sorted([x for x in Path(Config.output_path).iterdir() if x.is_dir()], key=natural_keys)
for output_dir in output_dirs:
    f = open(output_dir/'anomaly_events.txt', 'r')
    c = f.read()
    g.write(c)
    f.close()
g.close()

import json
import Config
from typing import Dict, List

class StableFrameList:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.stableList: Dict[str, List[List[int]]] = json.load(f)

        self.preprocessing()

    def preprocessing(self):
        #join interval
        pass

    def __getitem__(self, key):
        return self.stableList[str(key)]

if __name__ == '__main__':
    list = StableFrameList(Config.data_path + '/unchanged_scene_periods.json')
    for k, v in list.stableList.items():
        print(k, len(v))

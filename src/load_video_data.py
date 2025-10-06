import os
import json
import pandas as pd

def get_data_paths(dataset):
    datasets = {}
    datasets['msrvtt-ret'] = '/export/share/projects/mcai/datasets/msrvtt/'
    datasets['msvd-ret'] = '/export/share/projects/mcai/datasets/msvd/'
    datasets['didemo-ret'] = '/export/share/projects/mcai/datasets/didemo/'

    return datasets[dataset]

def load_video_ret(taskname):
    datapath = get_data_paths(taskname)
    file = os.path.join(datapath, taskname + '.json')

    with open(file, 'r') as f:
        captions_file = json.load(f)

    with open(os.path.join(datapath, 'test_list.txt'), 'r') as f:
        test_vids = [x.strip() for x in f.readlines()]
    
    test_videos = []
    test_captions = []
    for vid in test_vids:
        if('msrvtt' in taskname):
            test_videos.append(os.path.join(datapath, 'videos', vid + '.mp4'))
        elif('msvd' in taskname):
            test_videos.append(os.path.join(datapath, 'videos', vid + '.avi'))
        elif('didemo' in taskname):
            test_videos.append(os.path.join(datapath, 'videos', vid))
        else:
            test_videos.append(vid)

        test_captions.append(captions_file[vid][0])
    
    return test_videos, test_captions

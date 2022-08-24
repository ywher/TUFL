import os,sys
import json
import numpy as np
from PIL import Image
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)

n_classes = 13
lb_map = {7:0,8:1,11:2,19:3,20:4,21:5,23:6,24:7,25:8,26:9,28:10,32:11,33:12}

def convert_labels(label):
        label_copy = 255 * np.ones(label.shape, dtype=np.int64)
        for k, v in lb_map.items():
            label_copy[label == k] = v
        return label_copy

result = {str(i):[] for i in range(n_classes)}
dataset_root = os.path.join(root_folder, 'data/cityscapes') 
img_folder = os.path.join(dataset_root, 'leftImg8bit', 'train')
lb_folder = os.path.join(dataset_root, 'gtFine', 'train')

sub_folders = os.listdir(lb_folder) #cities
for city in sub_folders:
    print('city', city, 'begin')
    city_path = os.path.join(lb_folder, city)
    items = os.listdir(city_path)
    label_names = [item for item in items if 'labelIds.png' in item]
    for label_name in label_names:
        lb_path = os.path.join(city_path, label_name)
        lb = Image.open(lb_path)
        lb = np.array(lb).astype(np.int64)
        lb = convert_labels(lb)
        unique_ids = np.unique(lb)
        for lb_id in unique_ids:
            if lb_id < n_classes: #exclude 255
                result[str(lb_id)].append(city+'/'+label_name)
    print(len(label_names), 'done')

with open('cityscapes_ids2path.json', 'w') as f:
    json.dump(result, f, indent=4)


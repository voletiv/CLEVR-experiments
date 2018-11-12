import json
import numpy as np
import os

from tqdm import tqdm

with open('/data/lisa/data/clevr-vikram/CLEVR_v1.0/scenes/CLEVR_train_scenes.json', 'r') as f:
    d = json.load(f)

scenes = d['scenes']

len_of_objs = []
for scene in scenes:
    len_of_objs.append(len(scene['objects']))

l = np.array(len_of_objs)
np.histogram(l, bins=list(range(np.min(len_of_objs), np.max(len_of_objs)+2)))

list_of_num_of_objs = np.unique(l)

for i in list_of_num_of_objs:
    os.makedirs(os.path.join('/data/lisa/data/clevr-vikram/clevr_num_of_objs/train', '%02d'% i))

for i in tqdm(range(len(l))):
    os.rename(os.path.join("/data/lisa/data/clevr-vikram/CLEVR_v1.0/images/train", "CLEVR_train_{0:06d}.png".format(i)),
        os.path.join("/data/lisa/data/clevr-vikram/clevr_num_of_objs/train", '%02d'% l[i], "CLEVR_train_{0:06d}.png".format(i)))



with open('/data/lisa/data/clevr-vikram/CLEVR_v1.0/scenes/CLEVR_val_scenes.json', 'r') as f:
    d = json.load(f)

scenes = d['scenes']

len_of_objs = []
for scene in scenes:
    len_of_objs.append(len(scene['objects']))

l = np.array(len_of_objs)
np.histogram(l, bins=list(range(np.min(len_of_objs), np.max(len_of_objs)+2)))

list_of_num_of_objs = np.unique(l)

for i in list_of_num_of_objs:
    os.makedirs(os.path.join('/data/lisa/data/clevr-vikram/clevr_num_of_objs/val', '%02d'% i))

for i in tqdm(range(len(l))):
    os.rename(os.path.join("/data/lisa/data/clevr-vikram/CLEVR_v1.0/images/val", "CLEVR_val_{0:06d}.png".format(i)),
        os.path.join("/data/lisa/data/clevr-vikram/clevr_num_of_objs/val", '%02d'% l[i], "CLEVR_val_{0:06d}.png".format(i)))

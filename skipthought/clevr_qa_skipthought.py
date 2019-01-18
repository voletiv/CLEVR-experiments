def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")

import gc
import glob
import h5py
import json
import numpy as np
import os

from tqdm import tqdm

#########################################
# IMAGES + QUESTIONS + ANSWERS
#########################################

dset = 'train'

clevr_dir = '/home/user1/Datasets/clevr-vikram/CLEVR_v1.0'
# CC
# clevr_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/CLEVR_v1.0'

# Images
with open(os.path.join(clevr_dir, 'scenes/CLEVR_' + dset + '_scenes.json'), 'r') as f:
    d = json.load(f)

scenes = d['scenes']

# QA
with open(os.path.join(clevr_dir, 'questions/CLEVR_'+ dset + '_questions.json'), 'r') as f:
    d = json.load(f)

qas = d['questions']

# Data
clevr_data = {}
clevr_data['image_index'] = []
clevr_data['image_filenames'] = []
clevr_data['num_of_objs'] = []
clevr_data['questions'] = []
clevr_data['answers'] = []
for qa in qas:
    clevr_data['image_index'].append(qa['image_index'])
    clevr_data['image_filenames'].append(qa['image_filename'])
    clevr_data['num_of_objs'].append(len(scenes[qa['image_index']]['objects']))
    clevr_data['questions'].append(qa['question'])
    clevr_data['answers'].append(qa['answer'])

qs = clevr_data['questions']
ans = clevr_data['answers']

# Skipthought vectors for QA
import skipthoughts; mem_check()
model = skipthoughts.load_model(); mem_check()
encoder = skipthoughts.Encoder(model)
skipthought_encoder = encoder
mem_check()
qs_enc = np.empty((0,4800))
ans_enc = np.empty((0,4800))
counter = 0
batch_size = 128
# do_nothing = True
do_nothing = False
for i in tqdm(range(len(qs)//batch_size+1)):
    # Reset qs_enc
    if i % 5 == 0:
        counter += 1
        if not do_nothing:
            del qs_enc
            del ans_enc
            print("gc collect")
            gc.collect()
            qs_enc = np.empty((0,4800))
            ans_enc = np.empty((0,4800))
    # # continue
    # if counter < 385:
    #     continue
    # # Do stuff
    # do_nothing = False
    # qs_enc
    qs_batch = qs[i*batch_size:(i+1)*batch_size]
    qs_batch_enc = skipthought_encoder.encode(qs_batch)
    qs_enc = np.vstack((qs_enc, qs_batch_enc))
    np.save('qs_{0}_{1:04d}.npy'.format(dset, counter), qs_enc)
    # ans_enc
    ans_batch = ans[i*batch_size:(i+1)*batch_size]
    ans_batch_enc = skipthought_encoder.encode(ans_batch)
    ans_enc = np.vstack((ans_enc, ans_batch_enc))
    np.save('ans_{0}_{1:04d}.npy'.format(dset, counter), ans_enc)
    mem_check()

"""
# SAVE FULL in h5 - OOM
# Load full qa
npy_files_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/skipthought'
qs_enc_files = sorted(glob.glob(os.path.join(npy_files_dir, 'qs*' + dset + '*.npy')))
ans_enc_files = sorted(glob.glob(os.path.join(npy_files_dir, 'ans*' + dset + '*.npy')))

qs_enc = np.empty((0,4800))
ans_enc = np.empty((0,4800))

for file in tqdm(qs_enc_files):
    qs_enc = np.vstack((qs_enc, np.load(file)))

for file in tqdm(ans_enc_files):
    ans_enc = np.vstack((ans_enc, np.load(file)))

# Save as h5
with h5py.File("arr.h5", "w") as hf:
    hf.create_dataset("qs", data=qs_enc)
    hf.create_dataset("ans", data=ans_enc)

# Save as npy
np.savez('a.npz', qs=qs_enc, ans=ans_enc)

# Load

for img_idx, q_enc, a_enc in zip()
    clevr_iqa[q['image_index']]['questions'].append()
    clevr_iqa[q['image_index']]['answers'].append()
"""

# SAVE h5 IN PARTS
npy_files_dir = '.'
dset = 'train'

qs_enc_files = sorted(glob.glob(os.path.join(npy_files_dir, 'qs*' + dset + '*.npy')))
ans_enc_files = sorted(glob.glob(os.path.join(npy_files_dir, 'ans*' + dset + '*.npy')))

vals_per_batch = 50000

qs_enc_batch = np.empty((0,4800))
ans_enc_batch = np.empty((0,4800))
counter = 0
# save_h5 = False
save_h5 = True
for qs_file, ans_file in tqdm(zip(qs_enc_files, ans_enc_files), total=len(qs_enc_files)):
    qs_enc_file = np.load(qs_file)
    ans_enc_file = np.load(ans_file)
    # if 'qs_train_0590.npy' in qs_file:
    #     save_h5 = True
    if len(qs_enc_batch) + len(qs_enc_file) >= vals_per_batch:
        print("Batch:", len(qs_enc_batch), "; curr file:", len(qs_enc_file))
        prev_len = len(qs_enc_batch)
        qs_enc_batch = np.vstack((qs_enc_batch, qs_enc_file[:(vals_per_batch-prev_len)]))
        ans_enc_batch = np.vstack((ans_enc_batch, ans_enc_file[:(vals_per_batch-prev_len)]))
        if save_h5:
            print("Saving", "clevr_skipthought_qa_" + dset + "_{0:02d}.h5".format(counter))
            with h5py.File("clevr_skipthought_qa_" + dset + "_{0:02d}.h5".format(counter), "w") as hf:
                hf.create_dataset("qs", data=qs_enc_batch)
                hf.create_dataset("ans", data=ans_enc_batch)
        # Next
        counter += 1
        qs_enc_batch = qs_enc_file[(vals_per_batch-prev_len):]
        ans_enc_batch = ans_enc_file[(vals_per_batch-prev_len):]
        print("New qs_enc_batch len exp", len(qs_enc_file)-(vals_per_batch-prev_len), "; real:", len(qs_enc_batch))
    else:
        qs_enc_batch = np.vstack((qs_enc_batch, qs_enc_file))
        ans_enc_batch = np.vstack((ans_enc_batch, ans_enc_file))


# Images
with open(os.path.join(clevr_dir, 'scenes/CLEVR_' + dset + '_scenes.json'), 'r') as f:
    d = json.load(f)

scenes = d['scenes']

for scene in scenes:
    clevr_iqa[scene['image_index']] = {}
    clevr_iqa[scene['image_index']]['image_filename'] = scene['image_filename']
    clevr_iqa[scene['image_index']]['class'] = len(scene['objects'])
    clevr_iqa[scene['image_index']]['questions'] = []
    clevr_iqa[scene['image_index']]['answers'] = []

def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")

import skipthoughts; mem_check()
model = skipthoughts.load_model(); mem_check()
encoder = skipthoughts.Encoder(model)
skipthought_encoder = encoder
mem_check()

import gc
import json
import numpy as np
import os

from tqdm import tqdm

dset = 'train'

clevr_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/CLEVR_v1.0'

# QA
with open(os.path.join(clevr_dir, 'questions/CLEVR_'+ dset + '_questions.json'), 'r') as f:
    d = json.load(f)

mem_check()

qas = d['questions']

qa_img_idx = []
qs = []
ans = []
for qa in qas:
    qa_img_idx.append(qa['image_index'])
    qs.append(qa['question'])
    ans.append(qa['answer'])

mem_check()

del d
del qas
gc.collect()

mem_check()

# 1

# Skipthought vectors for QA
qs_enc = np.empty((0,4800))
ans_enc = np.empty((0,4800))
counter = 0
batch_size = 128
dont_do_anything = True
for i in tqdm(range(3700)):
    # Reset qs_enc
    if i % 5 == 0:
        counter += 1
        if not dont_do_anything:
            del qs_enc
            del ans_enc
            print("gc collect")
            gc.collect()
            qs_enc = np.empty((0,4800))
            ans_enc = np.empty((0,4800))
    # continue
    if i < 2946:
        continue
    # Do stuff
    dont_do_anything = False
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

# 2

# Skipthought vectors for QA
qs_enc = np.empty((0,4800))
ans_enc = np.empty((0,4800))
counter = 0
batch_size = 128
do_nothing = True
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
    # continue
    if i < 4705:
        continue
    # Do stuff
    do_nothing = False
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

counter = 0
a = 0
for i in range(1000):
    if a + 640 >= vals_per_batch:
        p = a
        print(counter, i, a)
        a += vals_per_batch - p
        a = 640 - (vals_per_batch - p)
        counter += 1
        print(a, counter)
    else:
        a += 640


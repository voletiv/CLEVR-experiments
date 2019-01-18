import argparse
import cv2
import glob
import h5py
import json
import numpy as np
import os
import pickle
import shutil
import time
import tqdm


def _big_enough(image_p, min_size):
    img = Image.open(image_p)
    img_min_dim = min(img.size)
    if img_min_dim < min_size:
        print('Skipping {} ({})...'.format(image_p, img.size))
        return False
    return True


def read_clevr_data(clevr_dir, dset):
    # dset = 'train'
    # clevr_dir = '/home/user1/Datasets/clevr-vikram/CLEVR_v1.0'
    # # CC
    # # clevr_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/CLEVR_v1.0'
    # Images
    print("Reading", os.path.join(clevr_dir, 'scenes/CLEVR_' + dset + '_scenes.json'))
    with open(os.path.join(clevr_dir, 'scenes/CLEVR_' + dset + '_scenes.json'), 'r') as f:
        d = json.load(f)
    scenes = d['scenes']
    # QA
    print("Reading", os.path.join(clevr_dir, 'questions/CLEVR_' + dset + '_questions.json'))
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
    for qa in tqdm.tqdm(qas, total=len(qas)):
        clevr_data['image_index'].append(qa['image_index'])
        clevr_data['image_filenames'].append(qa['image_filename'])
        clevr_data['num_of_objs'].append(len(scenes[qa['image_index']]['objects']))
        clevr_data['questions'].append(qa['question'])
        clevr_data['answers'].append(qa['answer'])
    return clevr_data


def make_hdf5_files(out_dir, dset, clevr_dir, clevr_skipthought_npy_dir, imsize=128, shuffle=False, num_per_shard=1000,
                    max_shards=None, min_size=None, name_fmt='shard_{:05d}.hdf5', force=False):
    # dset = 'train'
    # clevr_dir = '/home/user1/Datasets/clevr-vikram/CLEVR_v1.0'
    # # CC
    # # clevr_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/CLEVR_v1.0'
    # # clevr_skipthought_npy_dir = '/home/voletivi/projects/rpp-bengioy/voletivi/data/sagan/clevr-vikram/skipthought'
    """
    Notes:
        - total output file size may be much bigger, as JPGs get decompressed and stored as uint8
    :param out_dir:
    :param images_glob:
    :param shuffle:
    :param num_per_shard:
    :param max_shards:
    :param min_size:
    :param name_fmt:
    :param force:
    :return:
    """
    if os.path.isdir(out_dir):
        if not force:
            raise ValueError('{} already exists.'.format(out_dir))
        print('Removing {}...'.format(out_dir))
        time.sleep(1)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'log'), 'w') as f:
        info_str = '\n'.join('{}={}'.format(k, v) for k, v in [
            ('out_dir', out_dir),
            ('dset', dset),
            ('clevr_dir', clevr_dir),
            ('clevr_skipthought_npy_dir', clevr_skipthought_npy_dir),
            ('imsize', imsize),
            ('shuffle', shuffle),
            ('num_per_shard', num_per_shard),
            ('max_shards', max_shards),
            ('min_size', min_size),
            ('name_fmt', name_fmt),
            ('force', force)])
        print(info_str)
        f.write(info_str + '\n')

    # Read clevr data
    clevr_data = read_clevr_data(clevr_dir, dset)

    # Skipthought file format
    skipthought_npy_counter = 0

    num_shards_total = len(clevr_data['image_index'])//num_per_shard
    name_fmt += '_of_{:05d}'.format(num_shards_total)

    writer = None
    shard_ps = []
    image_filename = ''
    # for count, image_p in enumerate(image_ps):

    counts = np.arange(len(clevr_data['image_index']))
    if shuffle:
        print('Shuffling...')
        np.random.shuffle(counts)

    for count in tqdm.tqdm(counts):

        # If shard is filled with num_per_shard entries
        if count % num_per_shard == 0:
            if writer:
                writer.close()
            shard_number = count // num_per_shard
            if max_shards is not None and shard_number == max_shards:
                print('Created {} shards...'.format(max_shards))
                return
            shard_p = os.path.join(out_dir, 'CLEVR_' + dset + '_' + name_fmt.format(shard_number))
            assert not os.path.exists(shard_p), 'Record already exists! {}'.format(shard_p)
            print('Creating {} [{}/{}]...'.format(shard_p, shard_number, num_shards_total))
            shard_ps.append(shard_p)
            writer = h5py.File(shard_p, 'w')
            writer.create_dataset('dset', data=dset)
            writer.create_group('images')
            writer.create_group('questions')
            writer.create_group('answers')
            writer.create_group('q_skipthoughts')
            writer.create_group('a_skipthoughts')

        # Index in shard
        index = str(count % num_per_shard)  # expected by HDF5DataLoader, TODO: document

        # Image
        if clevr_data['image_filenames'][count] != image_filename:
            # image = Image.open(os.path.join(clevr_dir, 'images', dset, clevr_data['image_filenames'][count])).convert('RGB')
            # http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
            image_filename = clevr_data['image_filenames'][count]
            image = cv2.imread(os.path.join(clevr_dir, 'images', dset, image_filename))
            # image = np.array(image, np.uint8).transpose((2, 0, 1))
            image = cv2.resize(image, (imsize, imsize), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            assert image.shape[0] == 3
        # Add image
        writer['images'].create_dataset(index, data=image)

        # Q
        writer['questions'].create_dataset(index, data=clevr_data['questions'][count])

        # A
        writer['answers'].create_dataset(index, data=clevr_data['answers'][count])

        # SKIPTHOUGHT
        # Read new npy if counter resets
        if count % 640 == 0:
            skipthought_npy_counter += 1
            skipthought_q_npy = np.load(os.path.join(clevr_skipthought_npy_dir, 'qs_' + dset + '_{0:04d}.npy'.format(skipthought_npy_counter)))
            skipthought_a_npy = np.load(os.path.join(clevr_skipthought_npy_dir, 'ans_' + dset + '_{0:04d}.npy'.format(skipthought_npy_counter)))
        # Read the correct entry in npy
        writer['q_skipthoughts'].create_dataset(index, data=skipthought_q_npy[(count % 640)])
        writer['a_skipthoughts'].create_dataset(index, data=skipthought_a_npy[(count % 640)])

    if writer:
        writer.close()
        assert len(shard_ps)
        # writing this to num_per_shard.pkl
        p_to_num_per_shard = {os.path.basename(shard_p): num_per_shard for shard_p in shard_ps}
        last_shard_p = shard_ps[-1]
        with h5py.File(last_shard_p, 'r') as f:
            p_to_num_per_shard[os.path.basename(last_shard_p)] = len(f.keys())
        print("Writing", os.path.join(out_dir, 'num_per_shard.pkl'))
        with open(os.path.join(out_dir, 'num_per_shard.pkl'), 'wb') as f:
            pickle.dump(p_to_num_per_shard, f)
    else:
        print('Nothing written, processed {} files...'.format(count))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir', type=str,
                   help='Where to store .hdf5 files. Additionally, the following files are stored: 1) log, which saves '
                        'the parameters used to create the .hdf5 files; and 2) num_per_shard.pkl, which stores a '
                        'dictionary mapping file names to number of entries in that file (see maker._get_num_in_shard).')
    p.add_argument('dset', choices=['train', 'val', 'test'])
    p.add_argument('clevr_dir')
    p.add_argument('clevr_skipthought_npy_dir')
    p.add_argument('--imsize', type=int, default=128,
                   help='Number of entries per record. Default: 1000')
    p.add_argument('--shuffle', action='store_true',
                   help='Shuffle images before putting them into records. Default: Not set')
    p.add_argument('--num_per_shard', type=int, default=1000,
                   help='Number of entries per record. Default: 1000')
    p.add_argument('--max_shards', type=int,
                   help='Maximum number of shards. Default: None')
    p.add_argument('--min_size', type=int,
                   help='Only use images with either height or width >= MIN_SIZE. Default: None')
    p.add_argument('--name_fmt', default='shard_{:05d}.hdf5',
                   help='Format string for shards, must contain one placeholder for number. Default: shard_{:05d}.hdf5')
    p.add_argument('--force', action='store_true',
                   help='If given, continue creation even if `out_dir` exists already. NOTE: In this case, '
                        '`out_dir` is removed first!')
    flags = p.parse_args()
    make_hdf5_files(**flags.__dict__)


if __name__ == '__main__':
    main()

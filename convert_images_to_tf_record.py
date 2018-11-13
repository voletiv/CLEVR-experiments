import os
import tensorflow as tf
import time

from glob import glob
from random import shuffle
from scipy.misc import imread
from skimage.transform import resize
from tqdm import tqdm

from utils import get_image, colorize

# Based on - https://github.com/openai/improved-gan/blob/master/imagenet/convert_imagenet_to_records.py

# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 128

DATASET_NAME = "CLEVR"
DATASET_TYPE = "train"
DATASET_DIR = "/data/lisa/data/clevr-vikram/clevr_num_of_objs/" + DATASET_TYPE
IMAGES_TYPE = "png"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_image_and_resize(image_file, size=IMSIZE):
    return resize(imread(image_file)[:, :, :3], (size, size), preserve_range=True).astype('uint8')


def main(argv):
    # dirs = glob("/home/ian/imagenet/ILSVRC2012_img_train_t1_t2/n*")
    # assert len(dirs) == 1000, len(dirs)
    # dirs = [d.split('/')[-1] for d in dirs]
    # dirs = sorted(dirs)
    dirs = [os.path.basename(d) for d in sorted(glob(os.path.join(DATASET_DIR, "*")))]
    str_to_int = dict(zip(dirs, range(len(dirs))))

    # pattern = "/home/ian/imagenet/ILSVRC2012_img_train_t1_t2/n*/*JPEG"
    pattern = os.path.join(DATASET_DIR, "*/*." + IMAGES_TYPE)
    files = glob(pattern)
    assert len(files) > 0
    # assert len(files) > 1000000, len(files)
    shuffle(files)

    # outfile = '/media/NAS_SHARED/imagenet/imagenet_train_labeled_' + str(IMSIZE) + '.tfrecords'
    outfile = DATASET_NAME + '_' + DATASET_TYPE + '_' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for f in tqdm(files):
        # print i
        # image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        # image = colorize(image)
        # assert image.shape == (IMSIZE, IMSIZE, 3)
        # image += 1.
        # image *= (255. / 2.)
        # image = image.astype('uint8')
        #print image.min(), image.max()
        #from pylearn2.utils.image import save
        #save('foo.png', (image + 1.) / 2.)
        image = read_image_and_resize(f)
        image_raw = image.tostring()
        class_str = f.split('/')[-2]
        label = str_to_int[class_str]
        # if i % 1 == 0:
        #     print i, '\t',label
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    tf.app.run()

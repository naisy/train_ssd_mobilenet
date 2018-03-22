# coding: utf-8
import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import yaml

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

# input data
LABEL_MAP_FILE     = cfg['label_map_file']
PASCALVOC_DIR      = cfg['pascalvoc_dir']
IMAGESETS_MAIN_DIR = cfg['imagesets_main_dir']
ANNOTATIONS_DIR    = cfg['annotations_dir']
JPEGIMAGES_DIR     = cfg['jpegimages_dir']
TRAINVAL_FILE      = cfg['trainval_txt']


# output data
TFRECORD_DIR       = cfg['tfrecord_dir']
TRAIN_RECORD_FILE  = cfg['train_record_file']
VAL_RECORD_FILE    = cfg['val_record_file']

IMAGESETS_MAIN_DIR = os.path.join(PASCALVOC_DIR,IMAGESETS_MAIN_DIR)
ANNOTATIONS_DIR    = os.path.join(PASCALVOC_DIR,ANNOTATIONS_DIR)
JPEGIMAGES_DIR     = os.path.join(PASCALVOC_DIR,JPEGIMAGES_DIR)
TRANVAL_FILE       = os.path.join(IMAGESETS_MAIN_DIR,TRAINVAL_FILE)

TRAIN_RECORD_FILE  = os.path.join(TFRECORD_DIR,TRAIN_RECORD_FILE)
VAL_RECORD_FILE    = os.path.join(TFRECORD_DIR,VAL_RECORD_FILE)

trainval_list = dataset_util.read_examples_list(TRAINVAL_FILE)

def mkdir(path):
    '''
    ディレクトリを作成する
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    print('Error. Already exists: {}'.format(path))
    return False

def dict_to_tf_example(data,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
            running dataset_util.recursive_parse_xml_to_dict)
        label_map_dict: A map from string label names to integers ids.
        ignore_difficult_instances: Whether to skip difficult instances in the
            dataset  (default: False).

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    jpeg_file = os.path.join(JPEGIMAGES_DIR, data['filename'])
    with tf.gfile.GFile(jpeg_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        class_name = obj["name"]
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(output_record_file,
                     label_map_dict,
                     jpegfilenames):
    '''Creates a TFRecord file from examples.

    Args:
        output_record_file: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        jpegfilenames: Examples to parse and save to tf record.
    '''
    writer = tf.python_io.TFRecordWriter(output_record_file)
    for idx, example in enumerate(jpegfilenames):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(jpegfilenames))
        xmlfile = os.path.join(ANNOTATIONS_DIR, example + '.xml')

        if not os.path.exists(xmlfile):
            logging.warning('Could not find %s, ignoring example.', xmlfile)
            continue
        with tf.gfile.GFile(xmlfile, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main():

    if not mkdir(TFRECORD_DIR): return
    
    label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_FILE)

    logging.info('Reading from dataset.')

    # shuffle
    random.seed(42)
    random.shuffle(trainval_list)
    num_trainval = len(trainval_list)
    num_train = int(0.7 * num_trainval)
    train = trainval_list[:num_train]
    val = trainval_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train), len(val))

    # create train.record
    create_tf_record(TRAIN_RECORD_FILE, label_map_dict, train)
    # create val.record
    create_tf_record(VAL_RECORD_FILE, label_map_dict, val)

if __name__ == '__main__':
    main()

# coding: utf-8
# pip install lxml pyyaml
import logging
import os

from lxml import etree
import tensorflow as tf
import numpy as np
import yaml

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def mkdir(path):
    '''
    ディレクトリを作成する
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    print('Error. Already exists: {}'.format(path))
    return False

def checkfile(path):
    '''
    ファイル・ディレクトリの存在を確認する
    '''
    if not os.path.exists(path):
        print('Error. Not exists: {}'.format(path))
        return False
    return True
    
def filewrite(filepath,s):
    f = open(filepath,'a')
    f.writelines(s)
    f.close()
    return

def fileread(filepath):
    f = open(filepath,'r')
    lines = []
    for line in f:
        lines += [line]
    f.close()
    return lines

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

LABEL_MAP_FILE     = cfg['label_map_file']
PASCALVOC_DIR      = cfg['pascalvoc_dir']
IMAGESETS_MAIN_DIR = cfg['imagesets_main_dir']
ANNOTATIONS_DIR    = cfg['annotations_dir']
JPEGIMAGES_DIR     = cfg['jpegimages_dir']
TRAINVAL_FILE      = cfg['trainval_txt']

IMAGESETS_MAIN_DIR = os.path.join(PASCALVOC_DIR,IMAGESETS_MAIN_DIR)
ANNOTATIONS_DIR    = os.path.join(PASCALVOC_DIR,ANNOTATIONS_DIR)
JPEGIMAGES_DIR     = os.path.join(PASCALVOC_DIR,JPEGIMAGES_DIR)
TRANVAL_FILE       = os.path.join(IMAGESETS_MAIN_DIR,TRAINVAL_FILE)

def main():

    if not checkfile(LABEL_MAP_FILE)  or \
       not checkfile(PASCALVOC_DIR)   or \
       not checkfile(ANNOTATIONS_DIR) or \
       not checkfile(JPEGIMAGES_DIR): return
    if not mkdir(IMAGESETS_MAIN_DIR): return

    # classname一覧をlabel_map.pbtxtから取得する
    label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_FILE)
    print(label_map_dict)
    classname_list=[]
    for classname in label_map_dict:
        classname_list += [classname]
    print(classname_list)
    classname_set = set(classname_list)


    # Annotationsのxmlファイル一覧を取得する
    file_names = sorted(os.listdir(ANNOTATIONS_DIR))
    for file_name in file_names:
        if not file_name.endswith(".xml"):
            continue
        with tf.gfile.GFile(os.path.join(ANNOTATIONS_DIR,file_name), 'r') as fid:
            xml_str = fid.read()
            #print(xml_str)
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)["annotation"]
            #print(xml)
            #print(data)
            # .jpgを削除してfilenameを取得する
            filename = data["filename"][:-4]
            found_classname_list = []
            for obj in data["object"]:
                found_classname_list += [obj["name"]]
                #print("{} {}".format(filename,obj["name"]))
            found_classname_set = set(found_classname_list)
            # Set Difference
            not_found_classname_list = list(classname_set - found_classname_set)
            #print(found_classname_list)
            #print("---")
            #print(not_found_classname_list)

            # ファイルの中でclassname全てを確認し、存在するなら 1、存在しないなら-1のファイルを作成する classname_trainval.txt
            if not checkfile(os.path.join(JPEGIMAGES_DIR,filename+".jpg")): return

            for classname in found_classname_list:
                s = filename+" 1\n"
                filewrite(os.path.join(IMAGESETS_MAIN_DIR,classname+"_trainval.txt"),s)
            for classname in not_found_classname_list:
                s = filename+" -1\n"
                filewrite(os.path.join(IMAGESETS_MAIN_DIR,classname+"_trainval.txt"),s)

    '''
    Don't need this.
    # trainval.txtをシャッフルしてtrain.txtとval.txtに分割する
    for classname in classname_list:
        filepath = os.path.join(IMAGESETS_MAIN_DIR,classname+"_trainval.txt")
        lines = fileread(filepath)
        np.random.shuffle(lines)
        num_datas = len(lines)
        num_train_datas = int(num_datas*0.5)
        train_datas = lines[:num_train_datas]
        val_datas = lines[num_train_datas:]
        filewrite(os.path.join(IMAGESETS_MAIN_DIR,classname+"_train.txt"),train_datas)
        filewrite(os.path.join(IMAGESETS_MAIN_DIR,classname+"_val.txt"),val_datas)
    '''
    # One class tranval.txt is enough. Because we only use the jpeg file list.
    for classname in classname_list:
        filepath = os.path.join(IMAGESETS_MAIN_DIR,classname+"_trainval.txt")
        lines = fileread(filepath)
        filewrite(os.path.join(TRAINVAL_FILE),lines)
        break



if __name__ == '__main__':
    main()

<a name='top'>

# Train ssd_mobilenet of the Tensorflow Object Detection API with your own data.
<hr>

## My training machine<br>
* AWS p3.2xlarge (Tesla V100)
  * Ubuntu
  * docker
  * nvidia-docker
    * nvidia/cuda
    * Python 3.6.3
    * OpenCV 3.3.1
    * Tensorflow r1.4.1 (build from source)
    * Tensorflow Object Detection API (branch r1.5)

<hr>

<a name='0'>

## Table of contents<br>
* Create PascalVOC data from your JPG images.(#1)
* Setting Tensorflow Object Detection API.(#2)
* Create TF Record data from PascalVOC data.(#3)
* Training.(#4)
* Freeze Graph.(#5)

<hr>

<a name='1'>

## Create PascalVOC data from your JPG images.(#1-1)
* Install [LabelImg](https://github.com/tzutalin/labelImg) on your Ubuntu desktop PC.
  * For Windows PC you can use VMware Player.
* Make labels by hand.(#1-2)
* Move files into dirs.(#1-3)
* Make label_map.pbtxt.(#1-4)
* Upload to traingin machine.(#1-5)

<a name='1-1'>

#### Install [LabelImg](https://github.com/tzutalin/labelImg) on your Ubuntu desktop PC.
Install LabelImg.
```
mkdir ~/github
sudo apt-get install -y pyqt4-dev-tools
sudo apt-get install -y python-pip
sudo pip install --upgrade pip
sudo pip install lxml

cd ~/github
git clone https://github.com/tzutalin/labelImg

cd ~/github/labelImg
make qt4py2
```

<a name='1-2'>

#### Make labels by hand.
Make all image's label with LabelImg.
```
cd ~/github/labelImg
./labelImg.py
```

![labelImg.png](./document/labelImg.png)

<a name='1-3'>

#### Move files into dirs.
Divide directory of jpg file and xml file.
```
mkdir ~/roadsign_data/PascalVOC/JPEGImages
mkdir ~/roadsign_data/PascalVOC/Annotations

# in your data dir
mv *.jpg ~/roadsign_data/PascalVOC/JPEGImages
mv *.xml ~/roadsign_data/PascalVOC/Annotations
```

<a name='1-4'>

#### Make label_map.pbtxt.
Make your label_map file like this.<br>
file:[./roadsign_data/roadsign_label_map.pbtxt](./roadsign_data/roadsign_label_map.pbtxt)<br>
```
item {
  id: 1
  name: 'stop'
}

item {
  id: 2
  name: 'speed_10'
}

item {
  id: 3
  name: 'speed_20'
}

item {
  id: 4
  name: 'speed_30'
}
```

<a name='1-5'>

#### Upload to traingin machine.
Copy the data to training machine.<br>
Example:<br>
```
scp -r ~/roadsign_data training_machine:~/github/train_ssd_mobilenet/
```

[<PAGE TOP>](#top)　[<TOC>](#0)
<hr>

<a name='2'>

## Setting Tensorflow Object Detection API.
* git clone Tensorflow Object Detection API.(#2-1)
* Edit exporter.py for Tensorflow r1.4.1.(#2-2)
* Build protocol buffer.(#2-3)
* Download checkpoint of ssd_mobilenet.(#2-4)
* Make your pipeline config file.(#2-5)

<a name='2-1'>

#### git clone Tensorflow Object Detection API.
Branch r1.5.
```
cd ~/github
git clone https://github.com/tensorflow/models
cd models/
git fetch
git checkout r1.5
```

<a name='2-2'>

#### Edit exporter.py for Tensorflow r1.4.1.
If you want to run on r1.4.1, you need to fix this problem.<br>
ValueError: Protocol message RewriterConfig has no "layout_optimizer" field.<br>
[https://github.com/tensorflow/tensorflow/issues/16268](https://github.com/tensorflow/tensorflow/issues/16268)<br>

Edit ~/github/models/research/object_detection/exporter.py L:71-72<br>
```
        rewrite_options = rewriter_config_pb2.RewriterConfig()
```

<a name='2-3'>

#### Build protocol buffer.
```
sudo apt-get install -y protobuf-compiler
cd ~/github/models/research
protoc object_detection/protos/*.proto --python_out=.
```

<a name='2-4'>

#### Download checkpoint of ssd_mobilenet.
Download checkpoint from here.
[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
```
cd ~/github/train_ssd_mobilenet/
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar xvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
```

<a name='2-5'>

#### Make your pipeline config file.
Copy sample config.<br>
```
cp ~/github/models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config ~/github/train_ssd_mobilenet/ssd_mobilenet_v1_roadsign.config
```

Edit your pipeline config like this.<br>
pipeline config: [ssd_mobilenet_v1_roadsign.config](ssd_mobilenet_v1_roadsign.config)
```
diff -u ~/github/models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config ~/train_ssd_mobilenet/ssd_mobilenet_v1_roadsign.config

--- /home/ubuntu/github/models/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config	2017-12-20 11:46:42.832787513 +0900
+++ /home/ubuntu/github/train_ssd_mobilenet/ssd_mobilenet_v1_roadsign.config	2018-03-19 11:22:10.521440000 +0900
@@ -6,7 +6,7 @@
 
 model {
   ssd {
-    num_classes: 90
+    num_classes: 4
     box_coder {
       faster_rcnn_box_coder {
         y_scale: 10.0
@@ -155,7 +155,7 @@
       epsilon: 1.0
     }
   }
-  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
+  fine_tune_checkpoint: "ssd_mobilenet_v1_coco_2017_11_17/model.ckpt"
   from_detection_checkpoint: true
   # Note: The below line limits the training process to 200K steps, which we
   # empirically found to be sufficient enough to train the pets dataset. This
@@ -174,9 +174,9 @@
 
 train_input_reader: {
   tf_record_input_reader {
-    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
+    input_path: "roadsign_data/tfrecords/train.record"
   }
-  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
+  label_map_path: "roadsign_data/roadsign_label_map.pbtxt"
 }
 
 eval_config: {
@@ -188,9 +188,9 @@
 
 eval_input_reader: {
   tf_record_input_reader {
-    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
+    input_path: "roadsign_data/tfrecords/val.record"
   }
-  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
+  label_map_path: "roadsign_data/roadsign_label_map.pbtxt"
   shuffle: false
   num_readers: 1
   num_epochs: 1
```

[<PAGE TOP>](#top)　[<TOC>](#0)
<hr>

<a name='3'>

## Create TF Record data from PascalVOC data.
```
sudo pip install lxml pyyaml

cd ~/github/train_ssd_mobilenet
# Please check config.yml

python build1_trainval.py
python build2_tf_record.py
```

[<PAGE TOP>](#top)　[<TOC>](#0)
<hr>

<a name='4'>

## Training.
```
cd ~/github/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd ~/github/train_ssd_mobilenet
# training/continue from checkpoint
python ~/github/models/research/object_detection/train.py --logtostderr --train_dir=./train --pipeline_config_path=./ssd_mobilenet_v1_roadsign.config
```

[<PAGE TOP>](#top)　[<TOC>](#0)
<hr>

<a name='5'>

## Freeze Graph.
```
# If you have output dir, please remove it first.
rm -rf ./output/

# Please change to your checkpoint file.: ./train/model.ckpt-11410
python ~/github/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=./ssd_mobilenet_v1_roadsign.config --trained_checkpoint_prefix ./train/model.ckpt-11410 --output_directory ./output \
       --config_override " \
            model{ \
              ssd { \
                post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"

# if you want to strip more, then execute next.
python ./freeze_graph.py

ls -l ./output/
```

[<PAGE TOP>](#top)　[<TOC>](#0)

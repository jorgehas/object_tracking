## TRAINING MAHJONG IMAGES FOR OBJECTS LABEL TRACKING USING SSD ALGORITHM 
# Table of Contents
1. [Architecture](#Architecture)
2. [Preparing the image dataset](#Training)
3. [Training the images with SSD-inceptionv2 model](#Training)
3. [Run detection and recognition on image](#third-example)

## 1- Architecture

![](Diagram.jpg)
 ## 1- Preparing the image dataset

***DATASET***### 1.1 saving image from a video file```python3 save_frame.py```### 1.2 Image renaming#### Use the following command to rename the images contained in the dataset```find . -name '*.jpg' | gawk 'BEGIN{ a=1 }{ printf "mv \"%s\" %04d.jpg\n", $0, a++ }' | bash```### 1.3  Images annotation and labeling   using LabelImg toolhttps://github.com/tzutalin/labelImg***For python 3.5***```
$ sudo apt install python3-lxml pyqt5-dev-tools cmake$ make qt5py3$ python3 labelImg.py```### 2- Training the images with SSD-inceptionv2 model### 2.1 Create tensorflow record```python3 create_pascal_tf_record.py \    --label_map_path=/home/georges/models-master/object_detection/data/mscoco_label_map.pbtxt \    --data_dir=/home/georges/models-master/slim/VOCdevkit/ --year=VOC2007 --set=train \    --output_path=mscoco_train.record```
### 2.2 Training phase
```
python3 object_detection/train.py \    --logtostderr \    --pipeline_config_path=/raid/georges_data/models-master/object_detection/samples/configs/ ssd_inception_v2_coco.config \    --train_dir=/raid/georges_data/models-master/slim/coco_inceptionv2```### 2.3 Exporting the inference graph ```
CHECKPOINT_NUMBER=60000python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/georges/models-master/object_detection/sampleconfigs/ssd_inception_v2_coco.config --trained_checkpoint_prefix   /home/georges/models-master/object_detection/model.ckpt-${CHECKPOINT_NUMBER} --output_directory /home/georges/models-master/object_detection/ output_inference_graph.pb
```

### 3- Run detection and recognition on image
```
$ python3 ssd_coco_detection.py

```
### Result

![](Result.png)




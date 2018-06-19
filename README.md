## TRAINING MAHJONG IMAGES FOR OBJECTS LABEL TRACKING USING SSD ALGORITHM 

## Architecture

![](Diagram.jpg)
 ## 1- Preparing the image dataset### 1.1 Image renaming#### Use the following command to rename the images contained in the dataset```find . -name '*.jpg' | gawk 'BEGIN{ a=1 }{ printf "mv \"%s\" %04d.jpg\n", $0, a++ }' | bashfind . -name '*.jpg' | gawk 'BEGIN{ a=968 }{ printf "mv \"%s\" %05d.jpg\n", $0, a++ }' | bash```### 1.2  Images annotation and labeling   using LabelImg tool	`https://github.com/tzutalin/labelImg`***For python 3.5***```
$ sudo apt install python3-lxml pyqt5-dev-tools cmake$ make qt5py3$ python3 labelImg.py```### 2- Training the images with SSD-VGG model***DATASET******Renaming the images***```
find . -name '*.jpg' | gawk 'BEGIN{ a=0 }{ printf "mv \"%s\" %05d.jpg\n", $0, a++ }' | bash
	--------```### 2.1 saving image from a video filepython3 save_frame.py```find . -name '*.jpg' | gawk 'BEGIN{ a=969 }{ printf "mv \"%s\" %05d.jpg\n", $0, a++ }' | bash```
```sudo rm /usr/bin/pythonsudo ln -s /usr/bin/python2.7 /usr/bin/python```### 2.2 Create tensorflow record```python3 create_pascal_tf_record.py \    --label_map_path=/home/georges/models-master/object_detection/data/mscoco_label_map.pbtxt \    --data_dir=/home/georges/models-master/slim/VOCdevkit/ --year=VOC2007 --set=train \    --output_path=mscoco_train.record```### 2.3 Exporting the inference graph ```
CHECKPOINT_NUMBER=60000python3 export_inference_graph.py     --input_type image_tensor     --pipeline_config_path /home/georges/models-master/object_detection/sampleconfigs/ssd_inception_v2_coco.config     --trained_checkpoint_prefix   /home/georges/models-master/object_detection/model.ckpt-${CHECKPOINT_NUMBER}   --output_directory /home/georges/models-master/object_detection/ output_inference_graph.pb```
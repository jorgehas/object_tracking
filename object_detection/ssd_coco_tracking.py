# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pylab
import numpy as np
import os
import six.moves.urllib as urllib
import sys
from datetime import datetime, timedelta

import tarfile
import tensorflow as tf
import zipfile
import time
import imutils
from imutils.video import FPS
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("../")
from time import clock

from utils import label_map_util
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from utils import visualization_utils as vis_util
import cv2
# import the required  packages
from imutils.video import FileVideoStream
#from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import sys

from tracking.utils.fps2 import FPS2
from matplotlib import tight_bbox

ap = argparse.ArgumentParser()


ap.add_argument("-t", "--type", required=True,
                help="input  from [0..5] for selection of type of tracker from ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'] ")
args = vars(ap.parse_args())

print("[info] tracker selected is ", args["type"])
# What model to download.
MODEL_NAME = '/home/georges/models-master/object_detection/output_inference_graph.pb'
MODEL_FILE = MODEL_NAME
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/georges/models-master/object_detection/output_inference_graph_mobilenet.pb/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', '/home/georges/models-master/object_detection/data/mscoco_label_map.pbtxt')

NUM_CLASSES = 30

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

trackerType = trackerTypes[int(args["type"])]

# initialize  multiple Tracker object with tracking algo
multipleTrackerOpenCV = cv2.MultiTracker(trackerType)
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap('/home/georges/models-master/object_detection/data/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def process_image(img):
    # Run SSD network.
    # rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
    #                                                           feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_np=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
    return classes, scores, boxes
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = '/home/georges/models-master/object_detection/test_images/'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
#
# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)


    # for image_path in TEST_IMAGE_PATHS:
    #   image = Image.open(image_path)
    #   # the array based representation of the image will be used later in order to prepare the
    #   # result image with boxes and labels on it.
    #   image_np = load_image_into_numpy_array(image)
    #   # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    #   image_np_expanded = np.expand_dims(image_np, axis=0)
    #   # Actual detection.
    #
    #   # Visualization of the results of a detection.
    #
    #   plt.figure(figsize=IMAGE_SIZE)
    #   plt.imshow(image_np)
    #   pylab.show()
t = 0

# Define the codec and create VideoWriter object

FPS_VAL = 0
i = 0
sec = clock()
cap = cv2.VideoCapture('/home/georges/models-master/video_test_2.mp4')
# cap.set(3, 640) #width
# cap.set(4, 480) #height
# cap.set(5, 20)  #frame rate
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300);
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360);
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_v2_608.avi',fourcc, cv2.CAP_PROP_FPS, (640,480))
fps = FPS2().start()
bboxes = []
init = False

while(1):
    t = t+1
    # get a frame
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=450)
    # # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.dstack([frame, frame, frame])
    print("ret==",ret)
    # show a frame
    #cv2.imshow("capture", frame)
    #img = mpimg.imread(path + image_names[-2])
    #if t <= 5:
    #    continue
    if ret == False:
        break
    t = 0
    # b, g, r = cv2.split(frame)
    # img1 = cv2.merge([r,g,b])
    #img = frame
    # img =cv2.imread(frame)
    start = time.time()
    # image_np = load_image_into_numpy_array(img1)
    # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    classes, scores, boxes =  process_image(frame)
    box = cv2.selectROI('tracking', frame, showCrossair=False, fromCenter=False)
    print(box)
    bboxes.append(box)
    if not init:
        success = multipleTrackerOpenCV.add(frame, bboxes)
        init = True

    success, bboxes = multipleTrackerOpenCV.update(frame)

    print("[info] no boxes {}".format(len(bboxes)))

    # for box in bboxes:
    #     p1 = (int(box[0]), int(box[1]))
    #     p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(bboxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    # cv2.rectangle(frame, p1, p2, (200, 0, 0))
    fps.update()
    cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.putText(frame,str(fps.fps()), (10, 30),
    #             font, 0.6, (0, 255, 0), 2)
    end = time.time()
    print(fps)
    # r, g, b = cv2.split(img)
    # img2 = cv2.merge([b,g,r])
    # print("img2.shape==",img2.shape)
    cv2.imshow("capture", frame)
    out.write(frame)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    fps.update()
fps.stop()
out.release()
cap.release()
cv2.destroyAllWindows()


       

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data' + '/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


k=3096
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      image_np = cv2.imread("C:/Users/Dylan/Autopilot-TensorFlow-master/driving_dataset_2/" + str(k) + ".jpg")


      # stop sign verify
      # image_np = cv2.imread("C:/Users/Dylan/Downloads/models/object_detection/stop-sign.jpg")

      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

  
      for i,b in enumerate(boxes[0]):
        if classes[0][i] == 1 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 8 or classes[0][i] == 10 or classes[0][i] == 13:
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=5)

          # Altert required for following classes
          if  classes[0][i] == 1 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 8:   
            if scores[0][i] >= 0.5:
              mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
              mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
              apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)*10
              cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*455),int(mid_y*256)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

              if apx_distance <=2:
                if mid_x > 0.3 and mid_x < 0.7:
                  cv2.putText(image_np, 'WARNING !',(300,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
          
          #traffic signal color
          

          # stop sign
          if classes[0][i] == 13:
            if scores[0][i] >=  0.5:
              cv2.putText(image_np, 'STOP !!',(300,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
              

      cv2.imshow('window',image_np)

      k += 1
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

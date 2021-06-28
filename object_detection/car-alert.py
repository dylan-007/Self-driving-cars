import numpy as np
import os
from numpy.lib.type_check import imag
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

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#########################################################################################

def detect_red_and_yellow(img, Threshold=0.19):
    """
    detect red and yellow
    :param img:
    :param Threshold:
    :return:
    """

    desired_dim = (30, 90)  # width, height
    img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red1 = np.array([170, 70, 50])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

    # defining the Range of yellow color
    lower_yellow = np.array([21, 39, 64])
    upper_yellow = np.array([40, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # red pixels' mask
    mask = mask0 + mask1 + mask2

    # Compare the percentage of red values
    rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])
    print(rate)

    if rate >= Threshold:
        return 1
    elif 0.1<rate<Threshold :
        return 2
    else:
        return 3



def read_traffic_lights_object(image, boxes, scores, classes, max_boxes_to_draw=5, min_score_thresh=0.5,traffic_light_label=10):

    im_width, im_height = image.size
    stop_flag = 10
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_light_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            crop_img = image.crop((left, top, right, bottom))

            if detect_red_and_yellow(crop_img) == 1:
                stop_flag = 1
            elif detect_red_and_yellow(crop_img) == 2:
                stop_flag = 2

    return stop_flag

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def main():
    k= 9830
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                image_np = cv2.imread("C:/Users/Dylan/Autopilot-TensorFlow-master/driving_dataset_1/" + str(k) + ".jpg")
                # image_np = cv2.imread("C:/Users/Dylan/Downloads/models/object_detection/stop-sign.jpg")
                # image_np = cv2.imread("C:/Users/Dylan/Downloads/models/object_detection/traffic-signal-red.jpg")
                # image_np = cv2.imread("C:/Users/Dylan/Downloads/models/object_detection/traffic-signal-green.jpg")


                image = tensor_to_image(image_np)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                # Actual detection
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=5)

                for i,b in enumerate(boxes[0]):

                    # Distance Calculation and Alert required for following classes
                    #                 person                  car                  motorcycle            bus                  truck     
                    if  classes[0][i] == 1 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 8:   
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                        
                            if mid_x > 0.25 and mid_x < 0.75:
                                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)*10
                            
                                if apx_distance <=2:
                                    cv2.putText(image_np, '{}'.format(apx_distance) + ' m', (int(mid_x*455),int(mid_y*230)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                                    cv2.putText(image_np, 'WARNING !',(300,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                else:
                                    cv2.putText(image_np, '{}'.format(apx_distance) + ' m', (int(mid_x*455),int(mid_y*230)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            
                    #traffic light 
                    if classes[0][i] == 10:
                        if scores[0][i] >= 0.5:
                            stop_flag = read_traffic_lights_object(image, np.squeeze(boxes), np.squeeze(scores),np.squeeze(classes).astype(np.int32))

                            if stop_flag == 1:
                                print("Red signal detected")
                                cv2.putText(image_np, 'Red Signal..STOP!',(100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            elif stop_flag == 2:
                                print("Green signal detected")
                                cv2.putText(image_np, 'GO!',(100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


                    # stop sign
                    if classes[0][i] == 13:
                        if scores[0][i] >=  0.5:
                            cv2.putText(image_np, 'STOP !!',(300,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        

                cv2.imshow('result',image_np)

                k += 1

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

if __name__ == "__main__":
    main()

import numpy as np
import os
from numpy.lib.type_check import imag
import model
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from subprocess import call
import cv2
import pyttsx3
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


sess_1 = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess_1, "C:/Users/Dylan/Downloads/models/object_detection/save/model.ckpt")


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


sess = tf.Session(graph=detection_graph)

# Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#########################################################################################


def detect_color(img):

    desired_dim = (30,90)
    img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # red mask 
    low_red0 = np.array([0, 70, 50])
    high_red0 = np.array([10, 255, 255])
    mask_red0 = cv2.inRange(img_hsv, low_red0, high_red0)

    low_red1 = np.array([170, 70, 50])
    high_red1 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(img_hsv, low_red1, high_red1)
    mask_red = mask_red0 + mask_red1
    # cv2.imshow("mask-red",cv2.resize(mask_red,(90,270)))
    ratio_1 = (np.count_nonzero(mask_red) /(desired_dim[0] * desired_dim[1]))
    # print(ratio_1*100," % of red")

    # yellow mask
    low_yellow = np.array([21, 39, 64])
    high_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, low_yellow, high_yellow)
    # cv2.imshow("mask-yellow",cv2.resize(mask_yellow,(90,270)))
    ratio_2 = (np.count_nonzero(mask_yellow) /(desired_dim[0] * desired_dim[1]))
    # print(ratio_2*100," % of yellow")

    # green mask 
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    mask_green = cv2.inRange(img_hsv, low_green, high_green)
    # cv2.imshow("mask-green",cv2.resize(mask_green,(90,270)))
    ratio_3 = (np.count_nonzero(mask_green) /(desired_dim[0] * desired_dim[1]))
    # print(ratio_3*100," % of green")

    best_color = max(ratio_1,ratio_2,ratio_3)

    if best_color >0.01:
        if best_color == ratio_1:
            return 1
        elif best_color == ratio_2:
            return 2
        elif best_color == ratio_3:
            return 3
    else:
        return 25


def read_traffic_lights_object(image, boxes, scores, classes, max_boxes_to_draw=5, min_score_thresh=0.5,traffic_light_label=10):

    im_width, im_height = image.size
    stop_flag = 10
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh and classes[i] == traffic_light_label:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

                     
            crop_img = image.crop((left, top, right, bottom))

            if detect_color(crop_img) == 1:
                stop_flag = 1
            elif detect_color(crop_img) == 2:
                stop_flag = 2
            elif detect_color(crop_img) == 3:
                stop_flag = 3

    return stop_flag


def main():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    volume = engine.getProperty("volume")
    engine.setProperty("volume",0.1)
    engine.setProperty("voice",voices[0].id)

    steering_img = cv2.imread('C:/Users/Dylan/Downloads/models/object_detection/steering_wheel_image.jpg',0)
    rowst,colst = steering_img.shape
    smoothed_angle = 0

    k= 3096

    while True:

        image = Image.open(r"C:/Users/Dylan/Autopilot-TensorFlow-master/driving_dataset_2/" + str(k) + ".jpg")
        image_np = cv2.imread("C:/Users/Dylan/Autopilot-TensorFlow-master/driving_dataset_2/" + str(k) + ".jpg")

        # image = Image.open(r"C:/Users/Dylan/Downloads/models/object_detection/5.jpeg")
        # image_np = cv2.imread("C:/Users/Dylan/Downloads/models/object_detection/5.jpeg")


        image_model = cv2.resize(image_np[-150:], (200, 66)) / 255.0
                
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=2)

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
                            engine.say("warning vehicle ahead")
                            engine.runAndWait()
                            distance = str(round(apx_distance))

                        else:
                            cv2.putText(image_np, '{}'.format(apx_distance) + ' m', (int(mid_x*455),int(mid_y*230)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
            #traffic light 
            if classes[0][i] == 10:
                if scores[0][i] >= 0.5:
                    stop_flag = read_traffic_lights_object(image, np.squeeze(boxes), np.squeeze(scores),np.squeeze(classes).astype(np.int32))

                    if stop_flag == 1:
                        # print("Red signal detected")
                        cv2.putText(image_np, 'STOP!',(100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        engine.say("stop red signal detected")     

                    # elif stop_flag == 2:
                    #     # print("Yellow signal detected")
                    #     cv2.putText(image_np, 'GET READY !!',(100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                    
                    elif stop_flag == 3:
                        # print("Green signal detected")
                        cv2.putText(image_np, 'GO !!',(100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        engine.say("go")
                        
                    engine.runAndWait()
                    

            # stop sign
            if classes[0][i] == 13:
                if scores[0][i] >=  0.5:
                    cv2.putText(image_np, 'STOP !!',(300,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    engine.say("stop")
                    engine.runAndWait()


        #steering angle calculation
        degrees = model.y.eval(feed_dict={model.x: [image_model], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((colst/2,rowst/2),-smoothed_angle,1)
        dst = cv2.warpAffine(steering_img,M,(colst,rowst))
        cv2.imshow('result',image_np)
        cv2.imshow("steering wheel", dst)

        k += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import anpr.yolo_model.core.utils as utils
from anpr.yolo_model.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


Fweights = 'C:/ANPR System/frontend/anpr/yolo_model/data/custom'

Fclasses = 'C:/ANPR System/frontend/anpr/yolo_model/data/classes/obj.names'

#Flags
Ftiny = False
Fmodel = 'yolov4'
Fsize = 416
Fframework = 'tf'
Foutput_format = 'XVID'
Fiou = 0.45
Fscore = 0.25
Fdont_show = True

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(Fclasses, Ftiny, Fmodel)
saved_model_loaded = tf.saved_model.load(Fweights, tags=[tag_constants.SERVING])

def format_boxes(boxes, height, width, threshold):
    # out_boxes -> coordinates
    # out_scores -> scores
    # out_classes -> classes
    # num_boxes -> number of predictions
    out_boxes, out_scores, out_classes, num_boxes = list(boxes)
    rois = []

    for itr in range(num_boxes[0]):
        if out_scores[0][itr] > threshold:
            xmin = int(out_boxes[0][itr][1] * width)
            ymin = int(out_boxes[0][itr][0] * height)
            xmax = int(out_boxes[0][itr][3] * width)
            ymax = int(out_boxes[0][itr][2] * height)
            class_id = int(out_classes[0][itr])
            rois.append([xmin, ymin, xmax, ymax, class_id])

    return rois


def detect_box(frame, threshold):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(frame, (Fsize, Fsize))
    image_data = image_data / 255.

    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=Fiou,
        score_threshold=Fscore
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    height, width, _ = frame.shape
    rois = format_boxes(pred_bbox, height, width, threshold)

    return rois

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def detect(framework = 'tf', weights = 'yolov4-416', size = 416, tiny = False, model = 'yolov4', str_input = False ,image = 'img.jpg' , pic_output = False ,output = 'result.png' , iou = 0.45, score = 0.25,classes_file=read_class_names("obj.names")):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    if str_input:
        original_image = cv2.imread(image)
    else:
        original_image =image
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (size, size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if model == 'yolov3' and tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([size, size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([size, size]))
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
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
        iou_threshold=iou,
        score_threshold=score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    if pic_output:
        img=utils.draw_bbox(original_image,pred_bbox)
        cv2.imwrite(output,img)
    return utils.get_boxes(original_image,pred_bbox,classes_file)




if __name__ == '__main__':
    try:
        app.run(detect)
    except SystemExit:
        pass

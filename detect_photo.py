import cv2
import numpy as np
import argparse
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import libraries.tf_lite_detect as detector
import time

import libraries.plate_detector as p_reader
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--p", required=True,
                help="path to input directory of faces + images")
args = vars(ap.parse_args())
# initialize our ANPR class
anpr = p_reader.plate_reader(debug=True)

img = cv2.imread(args["p"])
cv2.imshow("res", img)
cv2.waitKey()

boxes = detector.detect('tflite', 'tflite_vehicle_recognition_ver2.tflite', size=416, tiny=True, model='yolov4',
                        str_input=False, image=img, pic_output=False, output='result.png', iou=0.45, score=0.25,
                        classes_file=detector.read_class_names('obj.names'))

for x in boxes:
    if (x['class'] == "Car") or (x['class'] == "Motorcycle"):
        image= img[x['c0']:x['c2'], x['c1']:x['c3']]
        cv2.imshow("car",image)
        (lpText, lpCnt) = anpr.find_and_ocr(image, psm=7,
                                            clearBorder=True)
        # only continue if the license plate was successfully OCR'd
        if lpText is not None and lpCnt is not None:
            # fit a rotated bounding box to the license plate contour and
            # draw the bounding box on the license plate
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            # compute a normal (unrotated) bounding box for the license
            # plate and then draw the OCR'd license plate text on the
            # image
            (x, y, w, h) = cv2.boundingRect(lpCnt)
            cv2.putText(image, p_reader.cleanup_text(lpText), (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # show the output ANPR image
            print("[INFO] ",lpText)
            print(len(lpText))
            cv2.imshow("Output ANPR", image)
            cv2.waitKey(0)

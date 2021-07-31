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
ap.add_argument("-p", "--picture", required=True,
                help="path to input picture")
ap.add_argument("-o", "--output", required=False,
                help="path to output picture")
ap.add_argument("-m", "--model", default="tflite_model.tflite",
                help="path to tflite trained model [STRING]")
ap.add_argument("-cl", "--classes", default="obj.names",
                help="path to classes file of the model [STRING]")
ap.add_argument("-si", "--size", default=416,
                help="size of the pictures the model was trained on [INT]")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
ap.add_argument("-sig", "--sigma", default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                help="the alphabet allowed in the licence plate [STRING]")
ap.add_argument("-lr", "--lengthrange", default="0",
                help="the length allowed for a licence plate {format: 7-8 or 3-6 etc.} [STRING]")

args = vars(ap.parse_args())
# initialize our ANPR class
anpr = p_reader.plate_reader(debug=False)

size = args["size"]
conf = args["confidence"]
model = args["model"]
classes = detector.read_class_names(args["classes"])
out = args["output"]
if args["lengthrange"] == "0":
    args["lengthrange"] = 0

img = cv2.imread(args["picture"])
cv2.imshow("res", img)
cv2.waitKey()

boxes = detector.detect(framework='tflite', weights=model, size=size, tiny=True, model='yolov4',
                        str_input=False, image=img, pic_output=False, output='result.png', iou=0.45, score=conf,
                        classes_file=classes)
count = 0
for x in boxes:
    if (x['class'] == "Car") or (x['class'] == "Motorcycle"):
        image = img[x['c0']:x['c2'], x['c1']:x['c3']]
        (lpText, lpCnt) = anpr.find_and_ocr(image, args["sigma"], args["lengthrange"], psm=7,
                                            clearBorder=True)
        # only continue if the license plate was successfully OCR'd
        if lpText is not None and lpCnt is not None and lpText != "":
            # fit a rotated bounding box to the license plate contour and
            # draw the bounding box on the license plate
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            # compute a normal (unrotated) bounding box for the license
            # plate and then draw the OCR'd license plate text on the
            # image
            (x, y, w, h) = cv2.boundingRect(lpCnt)
            cv2.putText(image, lpText, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            # show the output ANPR image
            print("[INFO] ", lpText)
            print(len(lpText))
            cv2.imshow("Output ANPR", image)
            cv2.waitKey(0)
            if out is not None:
                cv2.imwrite(out + str(count) + ".jpg", image)
                count = count + 1

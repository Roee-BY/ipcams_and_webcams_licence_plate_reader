import cv2
import imutils
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import libraries.tf_lite_detect as detector
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--p", required=True,
                help="path to input directory of faces + images")
args = vars(ap.parse_args())

img = cv2.imread(args["p"])
cv2.imshow("res", img)
cv2.waitKey()

boxes = detector.detect('tflite', 'tflite_vehicle_recognition_ver2.tflite', size=416, tiny=True, model='yolov4',
                        str_input=False, image=img, pic_output=False, output='result.png', iou=0.45, score=0.25,
                        classes_file=detector.read_class_names('obj.names'))

for x in boxes:
    if (x['class'] == "Car") or (x['class'] == "Motorcycle"):
        image= img[x['c0']:x['c2'], x['c1']:x['c3']]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)#grabs the contours according to opencv ver
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = []
        cv2. imshow('gray',edged)
        cv2.waitKey(0)
        for c in contours:
            peri = cv2.arcLength(c, True)#shape boundery length (true symbolises closed)
            approx = cv2.approxPolyDP(c, 0.018	* peri, True)

            if len(approx) == 4:
                screenCnt.append(approx)

        if screenCnt is []:
            detected = 0
            print ("No contour detected")
        else:
             detected = 1

        if detected == 1:
            for x in screenCnt:
                cv2.drawContours(image, [x], -1, (0, 0, 255), 3)
        cv2. imshow('gray',image)
        cv2.waitKey(0)
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(image,image,mask=mask)
        print("1")

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]
        cv2.imshow("cropped",Cropped)
        cv2.waitKey(0)

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print("programming_fever's License Plate Recognition\n")
        print("Detected license plate Number is:",text)
        img = cv2.resize(img,(500,300))
        Cropped = cv2.resize(Cropped,(400,200))
        cv2.imshow('car',img)
        cv2.imshow('Cropped',Cropped)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

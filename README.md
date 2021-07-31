# IP Security Cameras and WebCameras Licence Plate Recognizer

Using yolov4 object detection and image processing with OpenCV, the project enables detecting and reading licence plates and storing a log plates detected.
The detection is achieved by detecing all the vehicles in the photo using a yolov4 tiny model that was converted to tflite and for each of the detected vehicles using image processing the lince plate is detected and after enhancing the contrast between the letters and the background the letters the licence plate is being processed by tesseract-ocr and we get the licence plate number.


## Usage Examples
the project was made light weight to enable deployment on low power hardware such as a raspberry pi or jetson nano or PCs and some usage example of it are:
1. Automation of a garage door to open upon recognition of your car
2. Keeping track of the vehicles entering to a certain event 

execution example py recognize_video.py -a rtsp://[username]:[password]@[ipaddress]/[stream settings] -v True -cl obj.names -m tflite_vehicle_recognition.tflite 

## Getting Started

The repository includes the following files:
1. pretrained tflite model which can recognize cars motorcycles trucks people ambulances and buses 
2. two folders of code files that are used to operate the process
3. the following 3 scripts:
    1. recognize_video.py - this script allows licence plate recognition in video streams and supports the following options:
        - pay attention!! all options have default values if you are uncertain about the values leave them at default
        - -m/--model - path to tflite trained model [STRING]
        - -a/--address - stream address
             - [rtsp] = rtsp://[username]:[password]@[ipaddress]/[camera specific additions] [STRING]
             - [webcam] = 0 [INT]
        - -cl/--classes - path to classes file of the model [STRING]
        - -l/--log - creating a log of recognized numbers [True\False]
        - -v/--view - viewing result for debugging [True/False]
        - -sl/--sleep" - amount of sleep time between frames in seconds [DOUBLE]
        - -si/--size" - size of the pictures the model was trained on [INT]
        - -c/--confidence" - minimum probability to filter weak detections [DOUBLE]
        - -sig/--sigma - the alphabet allowed in the licence plate [STRING]
        - -lr/--lengthrange - the length allowed for a licence plate {format: 7-8 or 3-6 etc.} [STRING]
    2. recognize_photo.py - this script allows licence plate recognition in photos
        - -m/--model - path to tflite trained model [STRING]
        - -p/--picture - path to input picture [STRING]
        - -o/--output - path to output picture defaults to null (in this case there wont be a output image)[STRING]
        - -cl/--classes - path to classes file of the model [STRING]
        - -si/--size" - size of the pictures the model was trained on [INT]
        - -c/--confidence" - minimum probability to filter weak detections [DOUBLE]
        - -sig/--sigma - the alphabet allowed in the licence plate [STRING]
        - -lr/--lengthrange - the length allowed for a licence plate {format: 7-8 or 3-6 etc.} [STRING]
    3. take_photo.py - this script allows you to check that the video stream you chose works by taking a single picture from the stream, it supports the following options:
        - -a/--address - stream address
             - [rtsp] = rtsp://[username]:[password]@[ipaddress]/[camera specific additions] [STRING]
             - [webcam] = 0 [INT]
        - -o/--out - path for the output photo [STRING]  
### Prerequisites

you will need the following dependencies to use the project:
- tesseract
- scikit-image
- numpy
- pytesseract
- tensorflow (version 2.3 and above)
- opencv
- argparse
### Installing

To install the following dependencies with pip use the following  commands:
- tesseract 
```
on debian(ubuntu/raspbian etc.)

apt-get install tesseract-ocr

or on windows 

https://github.com/UB-Mannheim/tesseract/wiki
```
- scikit-image
```
py -m pip install scikit-image
```
- numpy
```
py -m pip install numpy
```
- pytesseract
```
py -m pip install pytesseract
```
- tensorflow (version 2.3 and above)
```
py -m pip install tensorflow==2.3.0
```
- opencv
```
py -m pip install opencv-python
```
- argparse
```
py -m pip install argparse
```

## Built With

* [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) - The OCR model used
* [imutils](https://github.com/jrosebr1/imutils) - Video stream handling
* [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) - Used to create the vehicle recognition model

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/Roee-BY/ipcams_and_webcams_licence_plate_reader) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Roee Ben Yosef** - *Initial work* - [Roee-BY](https://github.com/Roee-BY)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


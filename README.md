# Automatic-Number-Plate-Recognition-System
This Automatic Number Plate Recognition System was created without using any external libraries such as PyTesseract. 

This project is split into three parts. One is localizing the number plate. The second is the OCR part where it reads all characters from the number plate. Third is the integration with Django as the frontend.

# Localizing the Number Plate
The dataset from https://www.kaggle.com/kedarsai/indian-license-plates-with-labels was used. YOLOv4 was used to train the model and then the the YOLO weights were converted to TensorFlow using https://github.com/hunglc007/tensorflow-yolov4-tflite this repo. The number plates were then extracted. The folder structure for this part is a little messed up but if you follow that repository properly you should be fine. Please make sure to implement that repo if you are following this repo down to a T.

# Optical Character Recognition
The number plates recognized in the previous step were processed using OpenCV and a CNN was used to classify the characters recognized. Various filtering techniques were used to filter out the characters in the number plate. I have uploaded the OCR pipeline separately in case you want to use it again.

# Frontend
The project was finally completed in Django by integrating the above two steps to form an effective processing pipeline and produce the required output. This project is just a basic implementation of the ANPR systems already in place, so there may be errors.

Thank you for taking a glance at this repo. Feel free to use it and make any changes as you like.

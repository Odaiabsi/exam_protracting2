# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:52:00 2020

@author: hp
"""

import cv2
import numpy as np

def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    """
    Get the face detection caffe model of OpenCV's DNN module
    
    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel" or models/opencv_face_detector_uint8.pb" based on quantization.
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt" or "models/opencv_face_detector.pbtxt" based on quantization.
    quantization: bool, optional
        Determines whether to use quantized tf model or unquantized caffe model. The default is False.
    
    Returns
    -------
    model : dnn_Net

    """
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.4:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def draw_faces(img, faces):
    """
    Draw faces on image

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw

    Returns
    -------
    None.

    """
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)

def count_persons(faces):
    """
    Count the number of unique persons detected in the frame.

    Parameters
    ----------
    faces : list
        List of coordinates of the detected faces.

    Returns
    -------
    int
        Number of unique persons detected.
    """
    num_persons = 0
    face_centers = []

    for face in faces:
        x, y, x1, y1 = face
        center_x = (x + x1) // 2
        center_y = (y + y1) // 2
        center = (center_x, center_y)

        # Check if the detected face is close to an existing face
        new_person = True
        for prev_center in face_centers:
            distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            if distance < 50:  # Adjust this threshold as needed
                new_person = False
                break

        if new_person:
            num_persons += 1
            face_centers.append(center)

    return num_persons

        
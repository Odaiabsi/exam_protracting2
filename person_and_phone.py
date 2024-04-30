import cv2

# Load the pre-trained COCO model and configuration files
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco.pbtxt')

# Check if the model loaded successfully
if net.empty():
    print("Failed to load the model.")
else:
    print("Model loaded successfully.")

# Continue with the rest of your code...

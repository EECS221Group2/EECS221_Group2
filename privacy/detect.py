# import necessary libraries
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

#load the models
yolov5_path = './yolov5'
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes as scale_coords
from utils.augmentations import letterbox

# load all the possible objects that can be detected
classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"
]

# load the pre-trained model of yolo_v5
def load_model(weights_path='yolov5s.pt', device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

model = load_model()

# set the path of the picutres needed to detect
input_path = './data/inputs'  
output_path = './data/outputs'

def perform_cleanup_tasks():
    print("Performing cleanup tasks...")
    print("Cleanup completed.")

# check through the whole folder to fetch the images and export the processed images
def process_folder(input_path, model, classes):
    for filename in os.listdir(input_path):
        # check if the file is a picture or not
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_path, filename)
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"Failed to load image {img_path}")
                continue
            # apply the pre-processed images
            processed_image = process_image(original_image)  
            # apply the mosaic to human that detected
            output_image = detect_and_mosaic(model, processed_image, original_image, classes)

            # show the processed images
            cv2.imshow("Mosaic Applied", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # save the processed images to the specific path
            outputpath = os.path.join(output_path, f"mosaic_{filename}")
            cv2.imwrite(outputpath, output_image)

    print("All images have been processed.")

def process_image(img):
    img = letterbox(img, new_shape=640)[0] # modify the image size and add the fillings
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416  
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32 to increase calculation efficiency
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

# plot the box that surround the 'people' image
def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    """Draw one bounding box on image img."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color[::-1]  # BGR to RGB
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])) # extract the axis and convert
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def detect_and_mosaic(model, img, original_image, classes):
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.15, 0.3, classes=[0], agnostic=False)  # make sure that the class of human is '0'

    for i, det in enumerate(pred):  # check through all the detecting results
        if len(det):
            # transform the 'image_size' to the original image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_image.shape).round()

            for *xyxy, conf, cls in det:
                if int(cls) == 0:  # if the component being detected is human
                    apply_mosaic_to_person(original_image, xyxy)  # apply mosaic to that human

    return original_image

# function that applying moasaic to human image
def apply_mosaic_to_person(img, bbox, neighborhood=15):
    x1, y1, x2, y2 = [int(x) for x in bbox]  # convert into integer
    roi = img[y1:y2, x1:x2]  # extract the area contains box labeled as human
    h, w = roi.shape[:2]
    if h > 0 and w > 0: 
        roi_small = cv2.resize(roi, (max(1, w // neighborhood), max(1, h // neighborhood)), interpolation=cv2.INTER_LINEAR) # make the ROI to the small part, losing some details to make the image more like a moasic
        roi_large = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST) # enlarge the image to the original size
        img[y1:y2, x1:x2] = roi_large  # put the enlarged image back into the box

process_folder(input_path, model, classes)

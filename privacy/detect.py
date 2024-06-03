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
def load_model(weights_path='yolov5m.pt', device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    return model

model = load_model()

# set the path of the targets needed to detect
img_input_path = './data/inputs/in_img'  
img_output_path = './data/outputs/out_img'
vid_input_path = './data/inputs/in_vid'
vid_output_path = './data/outputs/out_vid'

def perform_cleanup_tasks():
    print("Performing cleanup tasks...")
    print("Cleanup completed.")

# check the folder to fetch the images and export the processed images
def process_images_folder(img_input_path, img_output_path, model, classes):
    for filename in os.listdir(img_input_path):
        # check the if it's an img by extension
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            print(f"Skipping non-image file: {filename}")
            continue
        
        img_path = os.path.join(img_input_path, filename)
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Failed to load image {img_path}")
            continue

        processed_image = process_image(original_image)
        output_image = detect_and_mosaic(model, processed_image, original_image, classes)
        img_outputpath = os.path.join(img_output_path, f"mosaic_{filename}")
        cv2.imwrite(img_outputpath, output_image)
        print(f"Processed and saved: {img_outputpath}")

    print("All images have been processed.")


# check the folder to fetch the videos and export the processed video
def process_videos_folder(vid_input_path, vid_output_path, model, classes):
    for filename in os.listdir(vid_input_path):
        # Check if the file is a video by its extension
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Skipping non-video file: {filename}")
            continue

        video_path = os.path.join(vid_input_path, filename)
        output_video_path = os.path.join(vid_output_path, f"mosaic_{filename}")
        print(f"Processing video: {filename}")
        process_video(video_path, output_video_path, model, classes)
        print(f"Processed video saved to {output_video_path}")

    print("All videos have been processed.")



def process_image(img):
    img = letterbox(img, new_shape=640)[0]  # Adjust size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # Change data layout to CxHxW
    if torch.cuda.is_available(): # use GPU if available for better performance
        img = torch.from_numpy(img).to('cuda')
    else:
        img = torch.from_numpy(img).to('cpu')
    img = img.float() / 255.0  # Normalize to [0, 1]
    img = img.unsqueeze(0)  # Add batch dimension to match the models' needs
    return img

def process_video(vid_input_path, vid_output_path, model, classes):
    # open the video file
    cap = cv2.VideoCapture(vid_input_path)
    # obtain the basic charateristics of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(vid_output_path, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # deal with each fps of the video
        processed_frame = process_image(frame)  
        output_frame = detect_and_mosaic(model, processed_frame, frame, classes)
        # write the output video
        out.write(output_frame)

    # release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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

def main():
    process_images_folder(img_input_path, img_output_path, model, classes)
    process_videos_folder(vid_input_path, vid_output_path, model, classes)

if __name__ == "__main__":
    main()

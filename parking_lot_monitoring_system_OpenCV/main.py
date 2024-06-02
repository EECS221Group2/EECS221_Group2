import cv2
import numpy as np 
import pickle

# Loading list of parking space coordinates from a pickle file
with open('car_park_pos', 'rb') as f:
    pos_list = pickle.load(f)

# Dimensions of each parking space
width, height = 27, 15

# Function to check the status of each parking space in the given frame
def check_parking_space(img):
    free_spaces = 0

    # Looping through each parking space coordinate
    for pos in pos_list:
        # Cropping the image to get only the parking space area
        img_crop = img[pos[1]:pos[1] + height, pos[0]:pos[0] + width]                       
        count = cv2.countNonZero(img_crop)

        if count > 110:
            color = (0, 0, 255)
            car_color = detect_car_color(frame[pos[1]:pos[1] + height, pos[0]:pos[0] + width])
            cv2.putText(frame, car_color, (pos[0], pos[1] + height - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
        else:
            free_spaces += 1
            color = (0, 255, 0)

        # Drawing a rectangle around the parking space and displaying the count of non-zero pixels inside it
        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 1)

    # Displaying the total number of free parking spaces out of the total number of parking spaces
    cv2.putText(frame, f'{free_spaces} / {len(pos_list)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)

def detect_car_color(img_crop):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

    # Get the center pixel of the cropped image
    center_pixel = hsv[height // 2, width // 2]

    # Extract the hue, saturation, and value components
    hue, saturation, value = center_pixel

    # Thresholds for differentiating white, black, and gray
    sat_threshold = 50  # Low saturation
    val_threshold_high = 190  # High value for white
    val_threshold_low = 40   # Low value for black

    if value > val_threshold_high and saturation < sat_threshold:
        return 'White'
    elif value < val_threshold_low:
        return 'Black'
    elif saturation < sat_threshold and val_threshold_low < value < val_threshold_high:
        return 'Gray'

    # Convert hue to color label
    return hue_to_color(hue)

def hue_to_color(hue):
    # Mapping hues to color names based on the HSV hue angle
    if (hue >= 0 and hue <= 10) or (hue >= 170 and hue <= 180):
        return 'Red'
    elif hue >= 11 and hue <= 25:
        return 'Orange'
    elif hue >= 26 and hue <= 34:
        return 'Yellow'
    elif hue >= 35 and hue <= 85:
        return 'Green'
    elif hue >= 86 and hue <= 125:
        return 'Blue'
    elif hue >= 126 and hue <= 160:
        return 'Violet'
    else:
        return 'Unknown'

cap = cv2.VideoCapture("busy_parking_lot.mp4")

# Getting the dimensions of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Setting up the video writer to write the processed video to a file
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 codec
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    # Reading a frame from the video capture
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Converting the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurring the grayscale frame using a Gaussian filter
    blurred_frame = cv2.GaussianBlur(gray_frame, (3,3), 1)

    # Applying adaptive thresholding to the blurred frame to binarize it
    threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 25, 16)

    # Applying median filtering to the thresholded frame to remove noise
    frame_median = cv2.medianBlur(threshold_frame, 5)

    # Dilating the filtered frame to fill in gaps in the parking space boundaries
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(frame_median, kernel, iterations=1)
    
    check_parking_space(dilated_frame)
    out.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
           break

cap.release()
out.release()
cv2.destroyAllWindows()

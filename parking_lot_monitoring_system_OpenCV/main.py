# Importing necessary libraries
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
        #cv2.putText(frame, str(count), (pos[0], pos[1] + height - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)

    # Displaying the total number of free parking spaces out of the total number of parking spaces
    cv2.putText(frame, f'{free_spaces} / {len(pos_list)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)

def detect_car_color(img_crop):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Calculate the histograms for saturation and brightness (value)
    sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation channel
    val_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Value channel

    # Find the most dominant saturation and value
    dominant_sat = np.argmax(sat_hist)
    dominant_val = np.argmax(val_hist)

    # Thresholds for differentiating white, black, and gray
    sat_threshold = 25  # Low saturation
    val_threshold_high = 200  # High value for white
    val_threshold_low = 50   # Low value for black

    if dominant_val > val_threshold_high and dominant_sat < sat_threshold:
        return 'White'
    elif dominant_val < val_threshold_low:
        return 'Black'
    elif dominant_sat < sat_threshold and val_threshold_low < dominant_val < val_threshold_high:
        return 'Gray'

    # Continue with hue-based color detection for more vivid colors
    # Mask to filter only colors with enough saturation and value to ignore whites/grays
    mask = cv2.inRange(hsv, (0, 60, 60), (180, 255, 255))
    hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    h = hsv_masked[:,:,0]
    h = h[h > 0]  # Exclude zero values which represent masked out areas
    if len(h) == 0:
        return 'Unknown'  # Return 'Unknown' if no significant color is detected
    # Get the most common hue value
    hue_mode = np.bincount(h).argmax()
    # Convert hue to color label
    return hue_to_color(hue_mode)

def hue_to_color(hue):
    # Mapping hues to color names based on the HSV hue angle
    if (hue >= 0 and hue <= 10) or (hue >= 170 and hue <= 180):
        return 'Red'
    elif hue >= 11 and hue <= 18:
        return 'Red-Orange'
    elif hue >= 19 and hue <= 30:
        return 'Orange'
    elif hue >= 31 and hue <= 45:
        return 'Yellow-Orange'
    elif hue >= 46 and hue <= 70:
        return 'Yellow'
    elif hue >= 71 and hue <= 79:
        return 'Yellow-Green'
    elif hue >= 80 and hue <= 140:
        return 'Green'
    elif hue >= 141 and hue <= 169:
        return 'Blue'
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

while 1:
        # Reading a frame from the video capture
        ret, frame = cap.read()

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
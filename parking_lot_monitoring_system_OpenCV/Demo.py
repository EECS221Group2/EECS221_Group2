import cv2
import numpy as np
import pickle
import time
import csv
import os
from collections import Counter

# Loading list of parking space coordinates from a pickle file
with open('car_park_pos', 'rb') as f:
    pos_list = pickle.load(f)

# Dimensions of each parking space
width, height = 27, 15

# Dictionary to store the timestamps for each parking spot
parking_times = {}
# List to store parking records
parking_records = []

# Function to check the status of each parking space in the given frame
def check_parking_space(img):
    free_spaces = 0
    current_time = time.time()

    for pos in pos_list:
        img_crop = img[pos[1]:pos[1] + height, pos[0]:pos[0] + width]
        contours, _ = cv2.findContours(img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        car_detected = False

        if pos not in parking_times:
            parking_times[pos] = {'enter': None, 'exit': None, 'color': None}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area to be considered as a car
                car_detected = True

                # Mask for the contour
                mask = np.zeros(img_crop.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

                # Get the dominant color inside the contour
                dominant_color = get_dominant_color(frame[pos[1]:pos[1] + height, pos[0]:pos[0] + width], mask)
                car_color = detect_car_color(dominant_color)
                parking_times[pos]['color'] = car_color
                cv2.putText(frame, car_color, (pos[0], pos[1] + height - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
                break

        if not car_detected:
            free_spaces += 1
            color = (0, 255, 0)
            cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 1)

            if parking_times[pos]['enter'] is not None and parking_times[pos]['exit'] is None:
                parking_times[pos]['exit'] = current_time
                parked_duration1 = current_time - parking_times[pos]['enter']
                if parked_duration1 > 2:
                    parked_duration = parked_duration1
                else:
                    continue
                parking_records.append({
                    'position': pos,
                    'entry_time': round(parking_times[pos]['enter'], 2),
                    'exit_time': round(parking_times[pos]['exit'], 2),
                    'duration': round(parked_duration, 2),
                    'color': parking_times[pos]['color']
                })
                print(f"Vehicle left spot {pos}. Parked for {parked_duration:.2f} seconds. Color: {parking_times[pos]['color']}")
                parking_times[pos]['enter'] = None
                parking_times[pos]['exit'] = None
                parking_times[pos]['color'] = None

    cv2.putText(frame, f'{free_spaces} / {len(pos_list)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

def get_dominant_color(img_crop, mask):
    # Masked image
    masked_img = cv2.bitwise_and(img_crop, img_crop, mask=mask)

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    # Flatten the HSV image and mask, then filter out background pixels
    hsv_pixels = hsv_img[mask == 255]
    
    # Count the frequency of each color
    hsv_pixels = [tuple(p) for p in hsv_pixels]
    most_common_pixel = Counter(hsv_pixels).most_common(1)[0][0]

    return most_common_pixel

def detect_car_color(hsv_color):
    hue, saturation, value = hsv_color

    # Updated thresholds for better detection of white, black, and gray
    sat_threshold_white = 30
    val_threshold_white = 200
    sat_threshold_black = 50
    val_threshold_black = 50
    sat_threshold_gray = 50
    val_threshold_gray_high = 200
    val_threshold_gray_low = 50

    if value > val_threshold_white and saturation < sat_threshold_white:
        return 'White'
    elif value < val_threshold_black and saturation < sat_threshold_black:
        return 'Black'
    elif sat_threshold_gray < saturation < val_threshold_gray_high and val_threshold_gray_low < value < val_threshold_gray_high:
        return 'Gray'

    return hue_to_color(hue)

def hue_to_color(hue):
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

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 1)
    threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 16)
    frame_median = cv2.medianBlur(threshold_frame, 5)
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

# Saving the parking records to a CSV file
with open('parking_records.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['position', 'entry_time', 'exit_time', 'duration', 'color'])
    writer.writeheader()
    for record in parking_records:
        writer.writerow(record)

# Print the current working directory
print("Current working directory:", os.getcwd())

import openai
import cv2
import numpy as np
from PIL import Image
import base64
import io

# Set the API key
openai.api_key = 'sk-proj-2SIZzo4oMllAvkLRScoXT3BlbkFJZq4lfMUJEHeNLnGnCYvA'  # This will be kept empty in GitHub due to the restrictions of the API key

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_response(user_input, parking_records, image_base64=None, car_color=None):
    messages = [
        {
            "role": "system",
            "content": "This is a conversation with an AI about the user's car in the parking lot."  # This helps the AI model understand the type of interaction expected.
        },
        {
            "role": "user",
            "content": user_input  # User's input about their car 
        },
        {
            "role": "user",
            "content": f"Here is the information of the parking lot: {parking_records}"  # Provide the parking lot information as part of the user's message
        }
    ]

    if image_base64:
        messages.append({
            "role": "assistant",
            "content": f"Here is an image of the {car_color} car you're looking for: ![Car Image](data:image/png;base64,{image_base64})"
        })

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message['content']  # Return the content of the response

# Load the analysis results
with open("parking_records.csv", "r") as f:
    parking_records = f.read()

# Encode the images to base64
black_car_image_base64 = encode_image_to_base64("/Users/stevenyeh18/Desktop/parking_lot_monitoring_system/privacy/data/inputs/in_img/0.png") #path to the image
grey_car_image_base64 = encode_image_to_base64("/Users/stevenyeh18/Desktop/parking_lot_monitoring_system/privacy/data/inputs/in_img/1.png")  #path to the image

# Main interaction loop
while True:
    user_input = input("Describe the car you want to find (Ex: parking time, color...): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break  # Allow the user to exit the conversation

    # Check if the query is about a black or grey car
    if 'black car' in user_input.lower():
        response = get_response(user_input, parking_records, black_car_image_base64, "black")
    elif 'grey car' in user_input.lower():
        response = get_response(user_input, parking_records, grey_car_image_base64, "grey")
    else:
        response = get_response(user_input, parking_records)

    print("System:", response)

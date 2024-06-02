import openai
import cv2
import numpy as np 

# Set the API key
openai.api_key = ''#This will be kept enpty due to the restrictions of the API key

def get_response(user_input, parking_records):
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
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
        ],
        max_tokens=300
    )
    return response.choices[0].message.content  # Return the content of the response


# Load the analysis results
with open("parking_records.csv", "r") as f:
    parking_records = f.read()

# Main interaction loop
while True:
    user_input = input("Describe the car you want to find(Ex:parking time, color...): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break  # Allow the user to exit the conversation
    print("System:", get_response(user_input, parking_records))

import openai

# Set the API key
openai.api_key = 'sk-proj-6nHUCBnvJAPuCQFjylcHT3BlbkFJkKv5qoquSWPmOvIckWde'

# Constant URL for the image about which the user will ask questions
# We will replace this image URL with parkiing lot camera image in the future
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

def get_response(user_input):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "This is a conversation with an AI about the user's car in the parking lot." #This helps the AI model understand the type of interaction expected.
            },
            {
                "role": "user",
                "content": user_input  # User's input about their car 
            },
            {
                "role": "user",
                "content": {
                    "type": "image_url",  #tells the model the input shoould be considered in the context of the provided image
                    "image_url": IMAGE_URL
                }
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message['content']  # Return the content of the response

# Main interaction loop
while True:
    user_input = input("Describe Your Car: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break  # Allow the user to exit the conversation
    print("GPT-4-Turbo:", get_response(user_input))

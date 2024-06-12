# Privacy-aware IoT Systems in Computer Vision - EECS221 Group 2

This project is part of the Class EECS221 Internet of Things, Group 2. We are focused on developing privacy-aware systems using computer vision technologies in IoT environments. For more detailed information, please visit [Our Website](https://sites.google.com/view/eecs221group2/home).

| Title   | Content |
|---------|---------|
| Class:  | EECS221 |
| Group:  | GROUP2  |

## Overview

[Parking_Iot_monitoring_system_OpenCV](parking_lot_monitoring_system_OpenCV) contains the code for our parking lot monitoring system. We use machine learning models to identify cars by their color and parking duration. [Privacy](privacy)focuses on protecting users' information from being leaked to unauthorized parties.

## Instructions

The project is implemented in `Python`. The [detect.py](privacy/detect.py) script in [privacy](privacy) requires PyTorch to run. The directory includes:

- **[Data](privacy/data)**: Stores [input data](privacy/data/inputs/in_img), [output data](privacy/data/outputs/out_img) as well as the [training images](privacy/data/persons).
- **[Mosaic_faces_and_plates.py](privacy/mosaic_faces_and_plates.py)**: Specifically for detecting and mosaicing faces and license plates.
- **[Detect.py](privacy/detect.py)**: Detects human forms and applies mosaics for privacy.

The [LLMandPrivacy.py](LLMandPrivacy.py) requires `OpenAI API key`, and `base64 library`, and the [QueryLLM.py](QueryLLM.py) requires `OpenAI GPT-4o` access.<br>
Dependencies include `OpenCV` and `PyTorch`.

## [LLMandPrivacy.py](LLMandPrivacy.py) Script

This Python script integrates OpenAI's GPT-4 Turbo and utilizes OpenCV for image processing within our parking lot monitoring system:

- **API Integration**: Configures the OpenAI API to generate AI-driven responses based on user inputs about their cars.
- **Image Encoding**: Converts images to a base64 format for secure embedding within application responses.
- **Response Generation**: Constructs a conversational flow using AI to provide visual and text-based information on user-queried vehicles.
- **Data Handling**: Reads and processes parking records and images from specified paths, ensuring efficient and secure data management.
- **Interactive Query Loop**: Engages users in a dynamic interaction to retrieve specific vehicle information with options to exit the conversation.

### How It Works

The user interacts with the system by describing the car they are looking for. The script processes the input to fetch and display relevant car details and images, enhancing the user experience through an AI-driven conversational interface.

## [QueryLLM.py](QueryLLM.py) Script

This script enables interactive AI conversations using OpenAI's GPT-3.5 Turbo to answer questions about vehicles in a parking lot context:

- **API Configuration**: Initializes the API key for OpenAI service integration.
- **Interactive Query Functionality**: Utilizes a structured conversation to generate responses based on user inputs and a placeholder image URL.
- **Structured Conversation Setup**: Ensures that the AI understands the interaction context via predefined roles and potential visual inputs.
- **User Interaction Loop**: Allows continuous user input to describe their vehicle, facilitating tailored AI responses with an exit option.

### How It Works

Users provide descriptions of their cars, and the AI generates responses aimed at assisting with vehicle identification or information, planning for future integration with live parking lot images.

## [mosaic_faces_and_plates.py](privacy/mosaic_faces_and_plates.py)

This script enhances privacy by applying mosaic effects to sensitive regions like faces and license plates in images before sending them to the user:

- **Privacy Protection**: Utilizes Haar cascade classifiers to detect areas requiring anonymization and applies a mosaic effect.
- **Image Processing**: Handles image conversion and applies pixelation techniques to obscure sensitive details effectively.
- **Handling Multiple Images**: Processes and displays multiple images, adjusting them to fit screen resolutions and deletes them post-display to ensure data privacy.
- **File Management**: Manages image files carefully, ensuring no sensitive data remains stored after processing.

### How It Works

The script processes specified images to detect license plates, applies a mosaic blur, displays the results, and then deletes any temporary files to ensure privacy. This functionality is crucial for applications where personal data protection is a priority.

## Features

- **Vehicle Identification**: Automatically recognizes vehicle features such as brand and color.
- **Privacy Protection**: Applies mosaics to sensitive information in stored images, such as human forms and license plates.
- **Interactive AI**: Allows users to describe their vehicle, matching this description with database entries to retrieve the correct vehicle image.

## Technologies Used

- **Artificial Intelligence and Machine Learning**: For vehicle detection and information matching.
- **Computer Vision**: For processing images from parking lot cameras.
- **APIs and Database Management**: To handle data storage and retrieval operations efficiently.

## Contribution

Developers are encouraged to contribute by submitting pull requests or creating issues for bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


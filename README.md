# Privacy-aware IoT Systems in Computer Vision - EECS221 Group 2

This is Class EECS221 Internet of Things, group 2. Our main topic is about developing privacy-aware systems using computer vision technologies in IoT environments. For more detailed information, please visit [Our Website](https://sites.google.com/view/eecs221group2/home).

| Title   | Content |
|---------|---------|
| Class:  | EECS221 |
| Group:  | GROUP2  |

## Overview

The `parking_lot_monitoring_system_OpenCV` folder contains the code for our parking lot monitoring system. Here, we use machine learning models to identify cars by their color and parking duration. The `privacy` folder is dedicated to protecting users' information from being leaked to unauthorized parties.

## Instructions

The project is based on `Python`. The `detect.py` in the `privacy` folder requires PyTorch to run. Inside the `privacy` folder, you will find three main components:

- `data`: This directory is used for both input and output data.
- `detect.py`: This script detects human forms and applies mosaics to ensure privacy.
- `mosaic_faces_and_plates.py`: This script is specifically for detecting and mosaicing faces and license plates.

Note that both `OpenCV` and `PyTorch` are required to run the codes.

## Features

- **Vehicle Identification**: Automatically recognizes vehicle features such as brand and color.
- **Privacy Protection**: Applies mosaics to sensitive information in stored images, such as human forms and license plates.
- **Interactive AI**: Users can interact with a generative chat AI to describe their vehicle, and the system matches this description with database entries to display the correct vehicle image.

## Technologies Used

- **Artificial Intelligence and Machine Learning**: Utilized for vehicle detection and information matching.
- **Computer Vision**: Used for processing images from the parking lot cameras.
- **APIs and Database Management**: Handle data storage and retrieval operations efficiently.

## Contribution

Developers are welcome to contribute to this project by submitting pull requests or creating issues for bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


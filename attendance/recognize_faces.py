import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import face_recognition
import numpy as np
import base64
from PIL import Image
import io
import os
import cv2
from django.conf import settings  # Import settings to access MEDIA_ROOT

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')  # Path to the known faces directory
TOLERANCE = 0.5

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load known faces once and store them in a global variable
known_face_encodings = []
known_face_names = []
employee_numbers = []  # List to store employee IDs

def load_known_faces(KNOWN_FACES_DIR):
    global known_face_encodings, known_face_names, employee_numbers
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Directory '{KNOWN_FACES_DIR}' created for storing known faces.")
    
    # Proceed with loading faces if the directory exists
    for name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_folder):
            # Split the folder name to extract employee ID and full name
            try:
                employee_number, full_name = name.split(" - ", 1)  # Split into two parts
                print(f"Processing Employee ID: {employee_number}, Full Name: {full_name}")

                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(full_name)  # Use the full name
                        employee_numbers.append(employee_number)  # Store employee ID
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            except ValueError:
                print(f"Skipping folder '{name}': it does not match expected format.")

def recognize_faces_from_image(image_data):
    try:
        # List to store results for each face
        results = []

        # Decode the base64 image if not already in bytes
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        # Open the image
        image = Image.open(io.BytesIO(image_data))

        # Convert image to RGB
        image = image.convert('RGB')
        image_array = np.array(image)

        # Find faces in the captured image
        unknown_face_encodings = face_recognition.face_encodings(image_array)

        for face_encoding in unknown_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
            name = "Unknown"
            employee_number = "N/A"  # Default if not recognized

            # Check if we have a match
            if True in matches:
                # Find the indexes of all matched faces
                matched_indexes = [i for (i, b) in enumerate(matches) if b]

                # Use the first matched face (could also average scores)
                first_match_index = matched_indexes[0]
                name = known_face_names[first_match_index]
                employee_number = employee_numbers[first_match_index]  # Get corresponding employee ID

            results.append({"name": name, "employee_number": employee_number})  # Append result

        return results  # Return results

    except Exception as e:
        print(f"Error during recognition: {e}")
        return []

# Load known faces when the module is loaded
load_known_faces(KNOWN_FACES_DIR)

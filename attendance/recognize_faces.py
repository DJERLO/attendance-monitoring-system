from torchvision import transforms
import face_recognition
import numpy as np
import base64
from PIL import Image
import io
import os
from django.conf import settings  # Import settings to access MEDIA_ROOT
from django.core.cache import cache

from attendance.models import FaceImage

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

def load_known_faces_dataset(KNOWN_FACES_DIR):
    global known_face_encodings, known_face_names, employee_numbers

    # Try to get cached face encodings
    cached_faces = cache.get("known_faces")

    if cached_faces:
        known_face_encodings, known_face_names, employee_numbers = cached_faces
        print("Loaded faces from cache!")
        return

    print("Loading faces from directory...")

    known_face_encodings, known_face_names, employee_numbers = [], [], []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Directory '{KNOWN_FACES_DIR}' created for storing known faces.")

    for name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_folder):
            try:
                employee_number, full_name = name.split(" - ", 1)

                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(full_name)
                        employee_numbers.append(employee_number)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

            except ValueError:
                print(f"Skipping folder '{name}': it does not match expected format.")

    # Cache the known faces for 30 minutes
    cache.set("known_faces", (known_face_encodings, known_face_names, employee_numbers), timeout=1800)
    print("Faces cached successfully!")

# This function is called to load known faces from the database
def load_known_faces():
    global known_face_encodings, known_face_names, employee_numbers

    # Try to get cached face encodings
    cached_faces = cache.get("known_faces")

    if cached_faces:
        known_face_encodings, known_face_names, employee_numbers = cached_faces
        print("Loaded faces from cache!")
        return

    print("Cache expired or not found. Loading faces from the database...")

    known_face_encodings, known_face_names, employee_numbers = [], [], []

    # Fetch all the face images from the database
    face_images = FaceImage.objects.all()

    for face_image in face_images:
        # Retrieve the encoding for each face image
        encoding = face_image.get_encoding()
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(face_image.employee.full_name())  # Assuming Employee has a 'name' field
            employee_numbers.append(face_image.employee.employee_number)  # Assuming Employee has 'employee_number'

    # Cache the known faces for 30 minutes
    cache.set("known_faces", (known_face_encodings, known_face_names, employee_numbers), timeout=60*30)
    print("Faces cached successfully!")

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
        load_known_faces()  # Reload known faces
        return []
    
load_known_faces()

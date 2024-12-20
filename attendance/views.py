import datetime
import os
import base64
import logging
import json
import io
import cv2
import numpy as np
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from django.conf import settings
from django.utils import timezone
from asgiref.sync import sync_to_async
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from attendance.forms import EmployeeRegistrationForm, UserRegistrationForm
from attendance.models import CheckIn, Employee, FaceImage
from .model_evaluation import detect_face_spoof
from PIL import Image
import torch
from torchvision import models

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')  # Path to the known faces directory
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')

from .recognize_faces import load_known_faces, recognize_faces_from_image  # Ensure these functions are implemented
#Load Faces before check-in connection established
@sync_to_async
def async_load_known_faces(KNOWN_FACES_DIR):
    """Asynchronous wrapper for face recognition."""
    return load_known_faces(KNOWN_FACES_DIR)

@sync_to_async
def async_recognize_faces(img_data):
    """Asynchronous wrapper for face recognition."""
    return recognize_faces_from_image(img_data)

@sync_to_async
def async_detect_fake_face(img_data):
    """Asynchronous wrapper for fake face detection."""
    return detect_face_spoof(img_data)

@sync_to_async
def get_employee_by_id(employee_number):
    """Fetch employee from the database by employee_id."""
    return get_object_or_404(Employee, employee_number=employee_number)

@sync_to_async
def check_in(employee_number):
    today = timezone.now().date() # Get Todat's Date
    employee = get_object_or_404(Employee, employee_number=employee_number) #Get Employee Id Number and find that person
    return CheckIn.objects.filter(employee=employee, timestamp__date=today).exists() #Check if the person already check-in

@sync_to_async
def mark_attendance(employee_number):
    employee = get_object_or_404(Employee, employee_number=employee_number)
    return CheckIn.objects.create(employee=employee)

def check_attendance(request):
    """Renders the attendance page."""
    load_known_faces(KNOWN_FACES_DIR)
    return render(request, 'attendance/check.html')

async def attendance(request):
    """Handles the attendance face recognition."""
    logger.info("Received request: %s", request.body)
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            img_data = data.get('image')

            # Extract base64 string from data URL
            if img_data.startswith('data:image/jpeg;base64,'):
                img_data = img_data.replace('data:image/jpeg;base64,', '')
                img_data = base64.b64decode(img_data)
            else:
                return JsonResponse({"error": "Invalid image format"}, status=400)

            class_idx, confidence, message = await async_detect_fake_face(img_data)

            if message == 'Real':
                verify = await async_recognize_faces(img_data)  # Again, use original data
                logger.info("Verification result: %s", verify)

                # Assuming verify contains a list of dictionaries with name and employee_id
                if verify and isinstance(verify, list) and len(verify) > 0:
                    employee = None
                    employee_data = verify[0]
                    employee_number = employee_data.get('employee_number')
                    
                    if employee_number:  # Check if employee_id is not None
                        # Fetch employee profile using employee_id asynchronously
                        employee = await get_employee_by_id(employee_number)
                    
                    if employee:  # Check if employee is found
                        # Check if the employee has already checked in today
                        checkin_exists = await check_in(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        if not checkin_exists:
                            
                            try:
                                submit = await mark_attendance(employee_number)
                                print(submit)
                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked in today.",
                                    }]
                                }, status=200)
                            except Exception as e:
                                logger.error("Error recording attendance: %s", e)
                                return JsonResponse({
                                    "result": [{
                                        "message": "Error recording attendance. Please try again later."
                                    }]
                                }, status=500)

                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} has already checked in today.",
                                }]
                            }, status=400)

                    else:
                        logger.warning("Employee with ID %s not found.", employee_number)
                        return JsonResponse({"result": [{"message": "Employee not found."}]}, status=404)
            
            if message == 'Fake':
                return JsonResponse({"result":[{"message": "Possible Spoofing Detected"}]}, status=200)
            

        except json.JSONDecodeError as e:
            logger.error("JSON Decode Error: %s", e)
            return JsonResponse({"error": ["Invalid JSON"]}, status=400)
        except Exception as e:
            logger.error("Error: %s", e)
            return JsonResponse({"error": [str(e)]}, status=500)

    return JsonResponse({"result":[{"status": "No Content"}]}, status=200)

# Register New Employee
def user_registration(request):
    user = None  
    #Check if user is authenticated
    if request.user.is_authenticated:
        user = request.user  # Get the user
        
        # If user already registered employee, redirect to the dashboard
        if Employee.objects.filter(user=user).exists():
            messages.info(request, "You are already registered as an employee.")
            return redirect('dashboard')
        
        #If user hasn't registed yet for employee, redirect to the registration process
        if request.method == 'POST':
            user_form = UserRegistrationForm(request.POST, instance=user)
            employee_form = EmployeeRegistrationForm(request.POST, request.FILES)

            if user_form.is_valid() and employee_form.is_valid():
                # Create User instance
                user = user_form.save()
                # Create Employee instance and link to User
                employee = employee_form.save(commit=False)
                employee.user = user  # Link employee to the user
                employee.first_name = user.first_name  # Set first name from user
                employee.last_name = user.last_name    # Set last name from user
                employee.email = user.email            # Set email from user
                employee.save()  # Save employee instance

                messages.success(request, "Employee registered successfully! Please proceed to facial registration.")
                return redirect('facial-registration', employee_number=employee.employee_number)  # Redirect to a face registration page to capture their face for face recognition

        else:
            # Pre-fill the forms with user's information
            user_form = UserRegistrationForm(initial={
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'username': user.username,  # If you want to pre-fill the username as well
            })
            employee_form = EmployeeRegistrationForm()
        
    else:
        #If user choose to register without linking Google or any Third party
        if request.method == 'POST':
            user_form = UserRegistrationForm(request.POST)
            employee_form = EmployeeRegistrationForm(request.POST, request.FILES)

            if user_form.is_valid() and employee_form.is_valid():
                # Create User instance
                user = user_form.save()
                # Create Employee instance and link to User
                employee = employee_form.save(commit=False)
                employee.user = user  # Link employee to the user
                employee.first_name = user.first_name  # Set first name from user
                employee.last_name = user.last_name    # Set last name from user
                employee.email = user.email            # Set email from user
                employee.save()  # Save employee instance

                messages.success(request, "Employee registered successfully! Please proceed to facial registration.")
                return redirect('facial-registration', employee_number=employee.employee_number)  # Redirect to a face registration page to capture their face for face recognition

        else:
            user_form = UserRegistrationForm()
            employee_form = EmployeeRegistrationForm()

    return render(request, 'attendance/employee-registration.html', {
        'user_form': user_form,
        'employee_form': employee_form,
    })

# Upload Face Images
def user_face_registration(request, employee_number):
    if request.method == 'POST':
        data = json.loads(request.body)  # Load JSON payload
        image_data = data.get('image')  # Get the base64 image data
        
        try:
            person = Employee.objects.get(employee_number=employee_number)
        except Employee.DoesNotExist:
            return JsonResponse({"message": "Employee not found."}, status=404)

        print("Employee found:", person)

        if image_data:
            # If the image data is in a list, use the first item
            if isinstance(image_data, list):
                image_data = image_data[0]

            # Check if the data is in the expected format
            if image_data.startswith("data:image/") and ';base64,' in image_data:
                _, encoded = image_data.split(';base64,', 1)
            else:
                encoded = image_data  # Directly use the raw base64 data

            # Decode the base64 string
            try:
                image_bytes = base64.b64decode(encoded)
            except Exception as e:
                return JsonResponse({"message": "Error decoding image: " + str(e)}, status=400)

            # Create the employee's folder if it doesn't exist
            employee_folder = os.path.join(settings.MEDIA_ROOT, 'known_faces', f"{person.employee_number} - {person.first_name} {person.last_name}")
            os.makedirs(employee_folder, exist_ok=True)

            # Check the number of existing face images
            existing_face_images = FaceImage.objects.filter(employee=person)

            if existing_face_images.count() >= 5:
                # Optional: Remove the oldest image if the limit is reached
                oldest_face_image = existing_face_images.order_by('uploaded_at').first()
                if oldest_face_image:
                    # Delete the image file from the file system
                    if os.path.exists(oldest_face_image.image.path):
                        os.remove(oldest_face_image.image.path)
                    # Remove the image entry from the database
                    oldest_face_image.delete()

            # Create a unique filename for each image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Use timestamp for uniqueness
            image_filename = f"{person.employee_number}_face_{timestamp}.jpg"
            image_path = os.path.join(employee_folder, image_filename)

            # Save the image as a file
            with open(image_path, 'wb') as destination:
                destination.write(image_bytes)

            # Optionally, create the face image entry linked to the employee
            FaceImage.objects.create(employee=person, image=image_path)

            return JsonResponse({"message": "Image uploaded successfully."}, status=200)

        return JsonResponse({"message": "No image data provided."}, status=400)

    return render(request, 'attendance/face-registration.html', {'employee_number': employee_number})


#For Training Model Online
@csrf_exempt  # Use this for testing, consider a better approach for production
def online_training(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data['image'].split(',')[1]  # Get the base64 image data
            # Decode the base64 image
            image_data = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB

            # Convert the image to a numpy array for OpenCV
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV uses BGR format)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Load Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)

            results = []

            for (x, y, w, h) in faces:
                # Extract the face from the image
                face_image = image_np[y:y+h, x:x+w]

                # Encode face as Base64 for spoof detection
                _, buffer = cv2.imencode('.jpg', face_image)
                face_data = base64.b64encode(buffer).decode('utf-8')

                # Call the model evaluation function on each detected face
                class_idx, confidence, message = detect_face_spoof(face_data)

                results.append({
                    'class_idx': class_idx,  # Convert NumPy integer to Python integer
                    'confidence': confidence,  # Ensure confidence is a float
                    'message': message,
                    'coordinates': {
                        'x': int(x),    # Convert NumPy integers to Python integers
                        'y': int(y),
                        'w': int(w),
                        'h': int(h)
                    }  
                })

            return JsonResponse({'results': results})  # Return results for all detected faces
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'training/training.html')


# Login User
@login_required
def dashboard(request):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=request.user)
        # Continue to the dashboard since employee exists
        return render(request, 'user/dashboard.html',{ 
                    'user': user, 
                    'employee': employee})
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
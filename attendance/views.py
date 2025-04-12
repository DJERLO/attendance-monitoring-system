import datetime
import os
import base64
import logging
import json
import io
import cv2
import face_recognition
import numpy as np
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.contrib.auth import logout, login, authenticate
from django.http import FileResponse, HttpResponse, JsonResponse
from django.db.models import Count
from django.template.loader import get_template
from django.urls import reverse
from django.conf import settings
from django.utils import timezone
from datetime import timedelta, datetime as dt
from django.db.models import Q, Sum
from pytz import timezone as pytz_timezone
from asgiref.sync import sync_to_async
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import Group
from django.contrib.auth.decorators import login_required
import requests
from attendance.forms import EmployeeEmergencyContactForm, EmployeeRegistrationForm, LeaveRequestForm, UserRegistrationForm
from attendance.models import Announcement, Event, Notification, ShiftRecord, Employee, FaceImage, WorkHours
from xhtml2pdf import pisa
from .model_evaluation import detect_face_spoof
from PIL import Image
import torch
from torchvision import models
from channels.layers import get_channel_layer
from django.http import HttpResponse
from upstash_redis import Redis
from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.views.decorators.cache import cache_page
from django.utils.timezone import now
from django.core.paginator import Paginator
from django.http import JsonResponse

redis = Redis.from_env()

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')  # Path to the known faces directory
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')

def counter(request):
    count = redis.incr('counter')
    return HttpResponse(f'Page visited {count} times.')

# Check if the user is in the certain group (For Role-Based Access Control)
def is_Admin(user):
    if user.groups.filter(name='ADMIN').exists():
        return True
    return PermissionError
def is_hr(user):
    if user.groups.filter(name='HR ADMIN').exists():
        return True
    return PermissionError
def is_faculty(user):
    if user.groups.filter(name__icontains='Teaching Staff').exists():
        return True
    return PermissionError

#index.html
def index(request):
    return redirect('account_login')

def logout_view(request):
    logout(request)  # This properly logs the user out
    return redirect('account_logout')  # Redirect to allauth's logout page

from .recognize_faces import load_known_faces, recognize_faces_from_image  # Ensure these functions are implemented
#Load Faces before check-in connection established
@sync_to_async
def async_load_known_faces(KNOWN_FACES_DIR):
    """Asynchronous wrapper for face recognition."""
    return load_known_faces(KNOWN_FACES_DIR)

# Face Recognition
@sync_to_async
def async_recognize_faces(img_data):
    """Asynchronous wrapper for face recognition."""
    return recognize_faces_from_image(img_data)

# Anti-Spoofing Detection
@sync_to_async
def async_detect_fake_face(img_data):
    """Asynchronous wrapper for fake face detection."""
    return detect_face_spoof(img_data)

# Get Current Employee Instance by his employee_number 
@sync_to_async
def get_employee_by_id(employee_number):
    """Fetch employee from the database by employee_id."""
    return get_object_or_404(Employee, employee_number=employee_number)

# Checking In
@sync_to_async
def check_in(employee_number):
    """Checking-In The Employee"""
    today = timezone.now()  # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_in__date=today).exists()  # Check if already checked in

# Clocking-In Employee's Time
async def clock_in(employee_number):
    """Clocking-In Employee's Time-In"""
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee

    # Await the result of check-in function
    already_checked_in = await check_in(employee_number)  # Await the async call
    if not already_checked_in:
        # Create a new shift record in a sync-to-async context
        shift_record = await sync_to_async(ShiftRecord.objects.create)(
            employee=employee, clock_in = timezone.now()
        )
        return shift_record
    return None  # Or raise an exception if desired

# Checking-Out
@sync_to_async
def check_out(employee_number):
    """Checking-Out The Employee"""
    today = timezone.now() # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_out__date=today).exists()  # Check if already checked out

async def clock_out(employee_number):
    """Clocking-Out Employee's Time-Out"""
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee
    today = timezone.now()  # Get today's date

    # Find the shift record for AM and await
    shift_record = await sync_to_async(lambda: ShiftRecord.objects.filter(employee=employee, clock_in__date=today).first())()
    
    if shift_record:
        # Update clock-out time in a sync-to-async context
        shift_record.clock_out= timezone.now()
        await sync_to_async(shift_record.save)()
        return shift_record
    return None  # Or raise an exception if no record is found

# Check-In View Attendance
@cache_page(60 * 5)
def check_in_attendance(request):
    """Renders the attendance page."""
    load_known_faces(KNOWN_FACES_DIR) # Loads all the employee's faces in our dataset
    # Set your timezone to Asia/Manila
    manila_tz = pytz_timezone('Asia/Manila')
    current_time = timezone.now().astimezone(manila_tz)  # Convert to Manila timezone
    current_hour = current_time.hour
    is_am = 6 <= current_hour < 12  # From 6:00 AM to 11:59 AM

    # Initialize the context dictionary
    context = {}

    # Fetch attendance based on the time of day
    if is_am:
        context['time'] = 'AM'
    else:
        context['time'] = 'PM'

    # Format the timestamp for display (e.g., 'YYYY-MM-DD HH:MM:SS')
    context['timestamp'] = current_time.strftime('%m/%d/%Y %I:%M:%S %p')  # Format timestamp

    return render(request, 'attendance/check_in.html', context)

async def checking_in(request):
    """Handles the attendance check-in face recognition."""
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

            #This function checking for Face-Spoofing attacks
            class_idx, confidence, message = await async_detect_fake_face(img_data)

            #Check if the img_data is Real or Fake
            if message == 'Real':
                #Start the Facial Recognition Process
                verify = await async_recognize_faces(img_data)  
                logger.info("Verification result: %s", verify)

                #Verify may contains employee's name and employee_id
                if verify and isinstance(verify, list) and len(verify) > 0:
                    employee = None
                    employee_data = verify[0]
                    employee_number = employee_data.get('employee_number')
                    
                    if employee_number:  # Check if employee_id is not None
                        # Fetch employee profile using employee_id asynchronously
                        employee =  await get_employee_by_id(employee_number)
                    
                    if employee:  # Check if employee is found
                        # Check if the employee has already checked in today
                        checkin_exists =  await check_in(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        #If employee hasn't check_in yet
                        if not checkin_exists:
                            try:
                                #Clock-in that employee
                                submit =  await clock_in(employee_number)
                                
                                # Send WebSocket notification
                                # **SEND TO WEBSOCKET**
                                channel_layer = get_channel_layer()
                                await (channel_layer.group_send)(
                                    "attendance_group",
                                    {
                                        "type": "send_message",
                                        "message": f"✅ {employee_data.get('name')} has checked in for today's morning shift."
                                    }
                                )

                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked in today's morning shift.",
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
                                    "message": f"{employee_data.get('name')} has already checked in for today's morning shift.",
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

# Check-Out View Attendance
@cache_page(60 * 5)
def check_out_attendance(request):
    """Renders the attendance page."""
    load_known_faces(KNOWN_FACES_DIR) # Loads all the employee's faces in our dataset
    # Set your timezone to Asia/Manila
    manila_tz = pytz_timezone('Asia/Manila')
    current_time = timezone.now().astimezone(manila_tz)  # Convert to Manila timezone
    current_hour = current_time.hour
    is_am = 6 <= current_hour < 12  # From 6:00 AM to 11:59 AM

    # Initialize the context dictionary
    context = {}

    # Fetch attendance based on the time of day
    if is_am:
        context['time'] = 'AM'
    else:
        context['time'] = 'PM'

    # Format the timestamp for display (e.g., 'YYYY-MM-DD HH:MM:SS')
    context['timestamp'] = current_time.strftime('%m/%d/%Y %I:%M:%S %p')  # Format timestamp

    return render(request, 'attendance/check_out.html', context)

async def checking_out(request):
    """Handles the attendance check-out face recognition."""
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

            #This function checking for Face-Spoofing attacks
            class_idx, confidence, message = await async_detect_fake_face(img_data)

            #Check if the img_data is Real or Fake
            if message == 'Real':
                #Start the Facial Recognition Process
                verify = await async_recognize_faces(img_data)  
                logger.info("Verification result: %s", verify)

                #Verify may contains employee's name and employee_id
                if verify and isinstance(verify, list) and len(verify) > 0:
                    employee = None
                    employee_data = verify[0]
                    employee_number = employee_data.get('employee_number')
                    
                    if employee_number:  # Check if employee_id is not None
                        # Fetch employee profile using employee_id asynchronously
                        employee =  await get_employee_by_id(employee_number)
                    
                    if employee:  # Check if employee is found
                        # Check if the employee has already checked in today
                        checkin_exists =  await check_in(employee_number)
                        checkout_exists =  await check_out(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        # If employee has checked in and not checked out, allow clocking out
                        if checkin_exists and not checkout_exists:
                            try:
                                # Clock-Out that employee
                                submit = await clock_out(employee_number)
                                
                                # WebScoket
                                channel_layer = get_channel_layer()
                                await (channel_layer.group_send)(
                                    "attendance_group",
                                    {
                                        "type": "send_message",
                                        "message": f"✅ {employee_data.get('name')} has checked-out!"
                                    }
                                )

                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked out today!",
                                    }]
                                }, status=200)
                            except Exception as e:
                                logger.error("Error recording attendance: %s", e)
                                return JsonResponse({
                                    "result": [{
                                        "message": "Error recording attendance. Please try again later."
                                    }]
                                }, status=500)
                        
                        # If Employee already Check-In and Check-Out in the morning shift
                        elif checkin_exists and checkout_exists:
                            return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has already checked out for today!",
                                    }]
                                }, status=409)
                        
                        # If Employee hasn't Check-In Yet for morning shift.
                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} hasn't checked in yet for today!",
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
from allauth.socialaccount.models import SocialAccount
@login_required
def user_registration(request):
    user = None  
    #Check if user is authenticated
    if request.user.is_authenticated:
        user = request.user  # Get the user
        user_has_social_account = SocialAccount.objects.filter(user=request.user).exists()  
        # If user already registered employee, redirect to the dashboard
        if Employee.objects.filter(user=user).exists():
            messages.info(request, "You are already registered as an employee.")
            return redirect('dashboard')
        
        #If user hasn't registed yet for employee, redirect to the registration process
        if request.method == 'POST':
            # Handle social password setting logic
            if user_has_social_account:
                set_password = request.POST.get("set_password", None)
                if set_password:
                    # Hash and set password for fallback login
                    user.set_password(set_password)
                    user.save()
                    messages.success(request, "Password successfully set.")
                    
                    # Re-authenticate to keep the user logged in
                    user = authenticate(request, username=user.username, password=set_password)
                    if user is not None:
                        login(request, user)

                else:
                    messages.error(request, "Please set a password to complete registration.")
                    return redirect('employee-registration')

            user_form = UserRegistrationForm(request.POST, instance=user)
            employee_form = EmployeeRegistrationForm(request.POST, request.FILES)
            emergency_contact_form = EmployeeEmergencyContactForm(request.POST)

            if user_form.is_valid() and employee_form.is_valid() and emergency_contact_form.is_valid() :
                # Create User instance
                user = user_form.save()
                # Create Employee instance and link to User
                employee = employee_form.save(commit=False)
                employee.user = user  # Link employee to the user
                employee.first_name = user.first_name  # Set first name from user
                employee.last_name = user.last_name    # Set last name from user
                employee.email = user.email            # Set email from user
                employee.save()  # Save employee instance

                #For Emergency Contact of Employee
                emergency_contact = emergency_contact_form.save(commit=False)
                emergency_contact.employee = employee  # link to employee
                emergency_contact.save()

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
            emergency_contact_form= EmployeeEmergencyContactForm()
    else:
        #If user choose to register without linking Google or any Third party
        if request.method == 'POST':
            user_form = UserRegistrationForm(request.POST)
            employee_form = EmployeeRegistrationForm(request.POST, request.FILES)
            emergency_contact_form = EmployeeEmergencyContactForm(request.POST)

            if user_form.is_valid() and employee_form.is_valid() and emergency_contact_form.is_valid():
                # Create User instance
                user = user_form.save()
                # Create Employee instance and link to User
                employee = employee_form.save(commit=False)
                employee.user = user  # Link employee to the user
                employee.first_name = user.first_name  # Set first name from user
                employee.last_name = user.last_name    # Set last name from user
                employee.email = user.email            # Set email from user
                employee.save()  # Save employee instance

                #For Emergency Contact of Employee
                emergency_contact = emergency_contact_form.save(commit=False)
                emergency_contact.employee = employee  # link to employee
                emergency_contact.save()

                messages.success(request, "Employee registered successfully! Please proceed to facial registration.")
                return redirect('facial-registration', employee_number=employee.employee_number)  # Redirect to a face registration page to capture their face for face recognition

        else:
            user_form = UserRegistrationForm()
            employee_form = EmployeeRegistrationForm()
            emergency_contact_form = EmployeeEmergencyContactForm()

    return render(request, 'attendance/employee-registration.html', {
        "user_has_social_account": user_has_social_account,
        'user_form': user_form,
        'employee_form': employee_form,
        'emergency_contact_form': emergency_contact_form,
    })

# Upload Face Images
@login_required
def user_face_registration(request, employee_number):
    if request.method == 'POST':
        data = json.loads(request.body)  # Load JSON payload
        image_data = data.get('image')  # Get the base64 image data
        
        try:
            person = Employee.objects.get(employee_number=employee_number)
            user = person.user
        except Employee.DoesNotExist:
            return JsonResponse({"message": "Employee not found."}, status=404)


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
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                return JsonResponse({"message": "Error decoding image: " + str(e)}, status=400)

            # Convert image to RGB (face_recognition requires RGB format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using face_recognition
            face_locations = face_recognition.face_locations(rgb_image)


            if len(face_locations) == 0:
                return JsonResponse({"message": "No face detected in the image."}, status=400)
            
           # Get first detected face location
            top, right, bottom, left = face_locations[0]

            # Crop the face region
            cropped_face = rgb_image[top:bottom, left:right]

            if cropped_face.size == 0:
                return JsonResponse({"message": "Face cropping failed due to incorrect bounding box."}, status=400)

            # Resize the face to standard size (224x224)
            square_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)

            # Generate face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, [face_locations[0]])

            if len(face_encodings) == 0:
                return JsonResponse({"message": "Face encoding failed. Try another image with better lighting."}, status=400)

            # Resize to a fixed square size, like 224x224
            square_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)

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
            cv2.imwrite(image_path, square_face)

            #Create the face image entry linked to the employee
            FaceImage.objects.create(employee=person, image=image_path)

            return JsonResponse({"message": "Image uploaded successfully."}, status=200)

        return JsonResponse({"message": "No image data provided."}, status=400)

    return render(request, 'attendance/face-registration.html', {'employee_number': employee_number, "user": Employee.objects.get(employee_number=employee_number)})

# Face Verification Process
def face_verification(request):
    # Ensure faces are loaded only once
    if not cache.get("known_faces"):
        load_known_faces(KNOWN_FACES_DIR)
    return render(request, 'attendance/face-verification.html')

async def face_recognition_test(request):
    """Employee face_recognition_test """
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

            #This function checking for Face-Spoofing attacks
            class_idx, confidence, message = await async_detect_fake_face(img_data)

            #Check if the img_data is Real or Fake
            if message == 'Real':
                #Start the Facial Recognition Process
                verify = await async_recognize_faces(img_data)  
                logger.info("Verification result: %s", verify)

                #Verify may contains employee's name and employee_id
                if verify and isinstance(verify, list) and len(verify) > 0:
                    employee = None
                    employee_data = verify[0]
                    employee_number = employee_data.get('employee_number')
                    
                    if employee_number:  # Check if employee_id is not None
                        # Fetch employee profile using employee_id asynchronously
                        employee =  await get_employee_by_id(employee_number)
                    
                    if employee:  # Check if employee is found
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        return JsonResponse({
                            "result": [{
                                "message": f"{employee_data.get('name')} matched",
                                "name": employee_data.get('name'),
                                "employee_number": employee_number,
                                "profile_image_url": profile_image_url,
                                "class_idx": class_idx,
                                "confidence": confidence,
                                "is_face_genuine": message,
                            }]
                        }, status=200)
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

from django.core.files.base import ContentFile
@csrf_exempt
def upload_image(request):
    if request.method == "POST":
        image_data = request.POST.get("image")
        if image_data:
            # Save the image to the server (optional)
            image_data = image_data.split(",")[1]  # Remove the data URL prefix
            image = base64.b64decode(image_data)

            # Optionally, save the image to a local file or use it directly
            image_file = ContentFile(image, "uploaded_image.jpg")

            # Process the image (e.g., save to canvas or do additional work)
            # Here we call the anti-spoofing API with the base64 image
            response = detect_face_spoof(image_file)

            return JsonResponse(response)

        return JsonResponse({"error": "No image data received"}, status=400)
    return render(request, "training/anti-spoofing.html")

def get_top_employees():
    """Fetch the top 10 employees with the highest work hours."""
    employees = Employee.objects.all()
    # Sort using the improved total_hours_worked method
    sorted_employees = sorted(employees, key=lambda emp: emp.total_hours_worked(), reverse=True)
    return sorted_employees[:10]

# Notifications Views Start
# Mark Notification as Read
@csrf_exempt
def mark_notification_read(request, id):
    if request.method == "POST":
        try:
            notif = Notification.objects.get(pk=id)
            if not notif.is_read:
                notif.is_read = True
                notif.save()
                return JsonResponse({"status": "success"})
            return JsonResponse({"status": "already_read"})
        except Notification.DoesNotExist:
            return JsonResponse({"error": "Notification not found"}, status=404)

# Mark All Notifications as Read
@csrf_exempt
@csrf_exempt
def mark_all_notifications_read(request):
    if request.method == "POST" and request.user.is_authenticated:
        try:
            employee = request.user.employee  # assuming OneToOne relation from User to Employee
            unread_notifications = Notification.objects.filter(employee=employee, is_read=False)
            count = unread_notifications.update(is_read=True)
            return JsonResponse({"status": "success", "updated_count": count})
        except AttributeError:
            return JsonResponse({"error": "Employee not found for user"}, status=400)
    return JsonResponse({"error": "Unauthorized or invalid request"}, status=403)

@csrf_exempt
@login_required
def delete_notification(request, id):
    if request.method == 'POST':
        try:
            notif = Notification.objects.get(id=id, employee=request.user.employee)
            notif.delete()
            return JsonResponse({'status': 'success'})
        except Notification.DoesNotExist:
            return JsonResponse({'status': 'not_found'}, status=404)
    return JsonResponse({'status': 'invalid_method'}, status=405)

# Notifications Views End
from collections import Counter, defaultdict
STATUS_PRIORITY = {
    "EARLY": 1,
    "PRESENT": 2,
    "LATE": 3,
    "ABSENT": 4
}

def get_dominant_status(statuses):
    return sorted(statuses, key=lambda s: STATUS_PRIORITY.get(s, 99))[0]

def get_status_summary_per_day(date):
    daily_records = ShiftRecord.objects.filter(date=date)
    per_employee_status = defaultdict(list)

    for record in daily_records:
        per_employee_status[record.employee.id].append(record.status)

    status_counter = {status: 0 for status in STATUS_PRIORITY.keys()}

    for statuses in per_employee_status.values():
        dominant_status = get_dominant_status(statuses)
        status_counter[dominant_status] += 1

    return status_counter

@login_required
def dashboard(request):
    user = request.user
    try:
        employee = Employee.objects.get(user=user)
        notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
        total_employees = Employee.objects.count()
        top_10_employees = get_top_employees()

        today = now().date()
        manila_tz = pytz_timezone('Asia/Manila')
        current_time = now().astimezone(manila_tz)
        current_hour = current_time.hour
        timemode = 'am' if 6 <= current_hour < 12 else 'pm'

        work_hours = WorkHours.objects.first()
        if not work_hours:
            return JsonResponse({"message": "Work hours not set."}, status=400)

        shiftstatus = ShiftRecord.objects.filter(employee=employee, date=today).last()
        shiftlogs = ShiftRecord.objects.filter(employee=employee).order_by('-date')
        check_in_time = shiftstatus.clock_in if shiftstatus else None
        check_out_time = shiftstatus.clock_out if shiftstatus else None

        can_check_in = check_in_time is None
        can_check_out = check_in_time is not None and check_out_time is None

        # Date range for current month
        first_day_of_month = today.replace(day=1)
        last_day_of_month = (first_day_of_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

        # Generate monthly attendance data
        attendance_data = {
            "labels": [date.strftime("%Y-%m-%d") for date in all_dates],
            "EARLY": [],
            "PRESENT": [],
            "LATE": [],
            "ABSENT": [],
        }

        for date in all_dates:
            summary = get_status_summary_per_day(date)
            for status in attendance_data:
                if status != "labels":
                    attendance_data[status].append(summary.get(status, 0))

        # Compute today's and yesterday's attendance summary
        today_summary = get_status_summary_per_day(today)
        yesterday_summary = get_status_summary_per_day(today - timedelta(days=1))

        active_today = today_summary["EARLY"] + today_summary["PRESENT"] + today_summary["LATE"]
        active_yesterday = yesterday_summary["EARLY"] + yesterday_summary["PRESENT"] + yesterday_summary["LATE"]

        if active_yesterday > 0:
            active_today_percentage = ((active_today - active_yesterday) / active_yesterday) * 100
        else:
            active_today_percentage = 0

        if active_today_percentage > 0:
            active_today_trend = "increase"
        elif active_today_percentage < 0:
            active_today_trend = "decrease"
        else:
            active_today_trend = "no change"

        # Attendance overview (today)
        absent_count = total_employees - active_today
        attendance_overview = {
            "EARLY": today_summary["EARLY"],
            "PRESENT": today_summary["PRESENT"],
            "LATE": today_summary["LATE"],
            "ABSENT": absent_count,
        }

        return render(request, 'user/dashboard.html', {
            'user': user,
            'employee': employee,
            'notifications': notifications,
            'top_employees': top_10_employees,
            'total_employees': total_employees,
            'all_dates': all_dates,
            'shiftlogs': shiftlogs,
            'shiftstatus': shiftstatus,
            'check_in_time': check_in_time,
            'check_out_time': check_out_time,
            'can_check_in': can_check_in,
            'can_check_out': can_check_out,
            'can_clock_in': work_hours.can_clock_in(),
            'opening_time': work_hours.open_time,
            'closing_time': work_hours.close_time,
            'attendance_data': json.dumps(attendance_data),
            'attendance_overview': attendance_overview,
            'active_today': active_today,
            'active_today_percentage': active_today_percentage,
            'active_today_trend': active_today_trend,
            'timemode': timemode,
        })

    except Employee.DoesNotExist:
        return redirect('employee-registration')

@login_required
def attendance_sheet(request):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=request.user)
        notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
        total_employees = Employee.objects.all().count()
        
        # Get today's date
        today = timezone.now().date()
        first_day_of_month = today.replace(day=1)
        last_day_of_month = (first_day_of_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)

        # Generate a list of all dates in the current month
        all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

        # Fetch attendance records for the employee for the current month
        shift_records = ShiftRecord.objects.filter(employee=employee, date__month=today.month, date__year=today.year)


        # Pass all the necessary data to the template
        return render(request, 'user/attendance_sheet.html', { 
            'user': user, 
            'employee': employee,
            'total_employees': total_employees,
            'all_dates': all_dates,
            'shift_records': shift_records,
            'notifications': notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
@login_required
def attendance_sheet_date(request, month, year):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=user)
        notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
        total_employees = Employee.objects.all().count()
        
        # Convert month and year to an integer (in case they are strings from the URL)
        month = int(month)
        year = int(year)

        # Dynamically get first and last day of the selected month and year
        first_day_of_month = datetime.datetime(year, month, 1).date()
        if month == 12:
            last_day_of_month = datetime.datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            last_day_of_month = datetime.datetime(year, month + 1, 1).date() - timedelta(days=1)

        # Generate a list of all dates in the selected month
        all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

        selected_month = f"{year}-{str(month).zfill(2)}"  # Format YYYY-MM for input

        # Fetch attendance records for the current user/employee for the selected month
        shift_records = ShiftRecord.objects.filter(employee=employee, date__month=month, date__year=year)
        
        # Pass all the necessary data to the template
        return render(request, 'user/attendance_sheet.html', { 
            "selected_month": selected_month,
            'user': user, 
            'employee': employee,
            'total_employees': total_employees,
            'all_dates': all_dates,
            'shift_records': shift_records,
            'current_path': request.path,  # Pass request path to the template
            'notifications': notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')

@login_required
def employee_attendance_sheet(request, month, year, employee_number=None):
    user = request.user  # Get the logged-in user
    notifications = Notification.objects.filter(employee__user=user).order_by('-created_at')[:20]
    try:
        total_employees = Employee.objects.all().count()
        
        # Convert month and year to an integer (in case they are strings from the URL)
        month = int(month)
        year = int(year)

        # Dynamically get first and last day of the selected month and year
        first_day_of_month = datetime.datetime(year, month, 1).date()
        if month == 12:
            last_day_of_month = datetime.datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            last_day_of_month = datetime.datetime(year, month + 1, 1).date() - timedelta(days=1)

        # Generate a list of all dates in the selected month
        all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

        selected_month = f"{year}-{str(month).zfill(2)}"  # Format YYYY-MM for input

        # Determine which employee’s records to fetch
        if employee_number:
            try:
                employee = Employee.objects.get(employee_number=employee_number)
            except Employee.DoesNotExist:
                employee = None  # Set to None if not found
        else:
            employee = Employee.objects.get(user=user)  # Default to logged-in user
            
        shift_records = ShiftRecord.objects.filter(employee=employee, date__month=month, date__year=year)
        
        # Pass all the necessary data to the template
        return render(request, 'user/attendance_sheet.html', { 
            "selected_month": selected_month,
            'user': user, 
            'employee': employee,
            'employee_number': employee_number,
            'total_employees': total_employees,
            'all_dates': all_dates,
            'shift_records': shift_records,
            'notifications': notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
# School Event Calendar
@login_required
def school_event_calendar(request):
    user = request.user
    employee = get_object_or_404(Employee, user=user)  # Prevents crash if employee doesn't exist
    notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
    current_year = timezone.now().year
    url = f"https://date.nager.at/api/v3/PublicHolidays/{current_year}/PH"

    try:
        response = requests.get(url, timeout=5)  # Added timeout to prevent hanging requests
        response.raise_for_status()  # Raises an error for HTTP codes 4xx/5xx
        holidays = response.json()
        
    except requests.RequestException as e:
        holidays = []  # If API fails, return an empty list
        print(f"Error fetching holidays: {e}")  # Log error for debugging
        
    
    # Get all events from the database
    events = Event.objects.all()
    all_events = []

     # Define color mapping for each type of holiday
    holiday_type_colors = {
        'Public': 'blue',
        'Bank': 'green',
        'School': 'yellow',
        'Authorities': 'orange',
        'Optional': 'purple',
        'Observance': 'gray',
    }

    # Convert holidays to FullCalendar format (blue color)
    for holiday in holidays:

        holiday_type = holiday.get("types", [])  # Get types (may be empty)
        # Set the color based on the first type (if available)
        color = holiday_type_colors.get(holiday_type[0], 'blue') if holiday_type else 'blue'

        all_events.append({
            "title": holiday["localName"],  # Holiday name
            "start": holiday["date"],  # YYYY-MM-DD
            "description": holiday["name"],
            'startEditable': False,
            "allDay": True,
            "color": color,  # Holidays are blue
            "extendedProps": {
                "holidayType": holiday_type[0] if holiday_type else "Unknown",  # Add holiday type
            }
        })
    

    # Convert user-created events
    for event in events:
        # Check if the event has a recurring pattern (days_of_week)
        if event.days_of_week:
            if event.url is not None:
                all_events.append({
                    "id": event.id,
                    "daysOfWeek": event.days_of_week,  # List of days the event occurs
                    "title": event.title,
                    'startEditable': False,
                    "startRecur": event.start.isoformat(),
                    "endRecur": event.end.isoformat(),
                    "description": event.description,
                    "url": event.url,
                    "allDay": event.all_day,
                    "color": "red",  # User events are red
                })
            else:
                all_events.append({
                    "id": event.id,
                    "daysOfWeek": event.days_of_week,  # List of days the event occurs
                    "title": event.title,
                    'startEditable': False,
                    "startRecur": event.start.isoformat(),
                    "endRecur": event.end.isoformat(),
                    "description": event.description,
                    "allDay": event.all_day,
                    "color": "red",  # User events are red
                })
            
        if not event.days_of_week:
            if event.url is not None:
                all_events.append({
                    "id": event.id,
                    "title": event.title,
                    'startEditable': False,
                    "start": event.start.isoformat(),
                    "end": event.end.isoformat(),
                    "description": event.description,
                    "url": event.url,
                    "allDay": event.all_day,
                    "color": "red",  # User events are red
                })
            else:
                all_events.append({
                    "id": event.id,
                    "title": event.title,
                    'startEditable': False,
                    "start": event.start.isoformat(),
                    "end": event.end.isoformat(),
                    "description": event.description,
                    "allDay": event.all_day,
                    "color": "red",  # User events are red
                })

    return render(request, 'user/event_calendar.html', {
        'user': user,
        'employee': employee,
        'events': json.dumps(all_events),  # Pass all events to template
        'notifications': notifications,
    })
    
@login_required
def profile_view(request):
    user = request.user  # Get the logged-in user
    notifications = Notification.objects.filter(employee__user=user).order_by('-created_at')[:20]
    try:
        employee = Employee.objects.get(user=request.user)

        # Pass all the necessary data to the template
        return render(request, 'user/profile.html', { 
            'user': user, 
            'employee': employee,
            'notifications': notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
@login_required
def update_profile(request):
    if request.method == "POST":
        user = request.user
        employee = user.employee  # adjust if your relation is different

        employee.first_name = request.POST.get('first_name', employee.first_name)
        employee.middle_name = request.POST.get('middle_name') or ''
        employee.last_name = request.POST.get('last_name', employee.last_name)
        employee.birth_date = request.POST.get('birth_date', employee.birth_date)
        employee.email = request.POST.get('email', employee.email)
        employee.contact_number = request.POST.get('contact_number', employee.contact_number)
        employee.gender = request.POST.get('gender', employee.gender)

        employee.save()
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid method'})


@login_required
def create_event(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON request data
            print(data)
            # Convert start & end to DateTime
            start = dt.fromisoformat(data.get("start"))
            end = dt.fromisoformat(data.get("end"))

            # Convert daysOfWeek list into a comma-separated string for MultiSelectField
            days_of_week = ",".join(data.get("daysOfWeek", []))

            # Create and save event
            event = Event.objects.create(
                title=data.get("title"),
                description=data.get("description", ""),
                start=start,
                end=end,
                url=data.get("url", ""),
                days_of_week=days_of_week,
                all_day=data.get("allDay", False)
            )

            return JsonResponse({"message": "Event created successfully!", "event_id": event.id}, status=201)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
    
@login_required
def manage_face_images(request):
    try:
        user = request.user
        employee = Employee.objects.get(user=user)
        notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
        face_images = FaceImage.objects.filter(employee=employee)  # Or filter for specific employee if needed

        # Pass all the necessary data to the template
        return render(request, 'user/manage_face.html', { 
            'user': user, 
            'employee': employee,
            'face_images': face_images,
            "notifications": notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    

@login_required
def delete_face_image(request, image_id):
    face_image = get_object_or_404(FaceImage, id=image_id)
    if request.method == 'POST':
        face_image.delete()
        return redirect('manage_face_images')  # Redirect back to the face image management page
    return render(request, 'confirm_delete.html', {'face_image': face_image})

#HR Management
@login_required
def employee_management(request):
    try:
        user = request.user
        notifications = Notification.objects.filter(employee__user=user).order_by('-created_at')[:20]
        # Manually check if the user is in the HR group
        if not user.groups.filter(name='HR ADMIN').exists():
            raise PermissionDenied  # Ensures a 403 Forbidden response

        employee = Employee.objects.get(user=user)  # Get the logged-in employee
        
        employees = Employee.objects.all()
        total_employees = employees.count()
        
        return render(request, 'user/employee_management.html', {
            'user': user,
            'employee': employee,
            'employees': employees,
            'total_employees': total_employees,
            'notifications': notifications,

        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
#File Request Leave
def request_leave_view(request):
    employee = request.user.employee
    if request.method == 'POST':
        form = LeaveRequestForm(request.POST, request.FILES)
        if form.is_valid():
            leave = form.save(commit=False)
            leave.employee = request.user.employee  # assuming user is linked to Employee
            leave.status = 'PENDING'
            leave.save()
            return redirect('dashboard')  # or wherever you list their requests
    else:
        form = LeaveRequestForm()
    return render(request, 'user/file_leave.html', {'form': form, 'employee': employee})

@login_required
def employee_details(request, employee_number):
    """HR Management for Employee Profile"""
    try:
        user= request.user  # Get the logged-in user
        employee = Employee.objects.get(user=user)  # Get the logged-in employee
        notifications = Notification.objects.filter(employee=employee).order_by('-created_at')[:20]
        employee_profile = get_object_or_404(Employee, employee_number=employee_number)  # Get the employee profile by employee_number
        attendance_records = ShiftRecord.objects.filter(employee=employee_profile).order_by('-date')

        return render(request, 'user/employee_profile.html', {
            'user': user,
            'employee': employee,
            'employee_profile': employee_profile,
            'attendance_records': attendance_records,
            'notifications': notifications,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')

@login_required
def announcement_board(request):
    try:
        user = request.user
        employee = Employee.objects.get(user=user)
        announcements = Announcement.objects.filter(is_active=True).order_by('-created_at')
        return render(request, 'user/announcement.html', {
            'user': user,
            'employee': employee,
            'announcements': announcements
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')

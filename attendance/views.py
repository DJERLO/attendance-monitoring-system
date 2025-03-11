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
from django.http import FileResponse, HttpResponse, JsonResponse
from django.template.loader import get_template
from django.urls import reverse
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q
from pytz import timezone as pytz_timezone
from asgiref.sync import sync_to_async
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from attendance.forms import EmployeeRegistrationForm, UserRegistrationForm
from attendance.models import ShiftRecord, Employee, FaceImage
from xhtml2pdf import pisa
from .model_evaluation import detect_face_spoof
from PIL import Image
import torch
from torchvision import models

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')  # Path to the known faces directory
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')

#index.html
def index(request):
    return redirect('account_login')

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
def check_in_at_am(employee_number):
    """Checking-In at AM (sync function for async call)"""
    today = timezone.now().date()  # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_in_at_am__date=today).exists()  # Check if already checked in for AM

async def clock_in_at_am(employee_number):
    """Clocking-In at AM"""
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee

    # Await the result of check-in function
    already_checked_in = await check_in_at_am(employee_number)  # Await the async call
    if not already_checked_in:
        # Create a new shift record in a sync-to-async context
        shift_record = await sync_to_async(ShiftRecord.objects.create)(
            employee=employee, clock_in_at_am=timezone.now()
        )
        return shift_record
    return None  # Or raise an exception if desired

@sync_to_async
def check_out_at_am(employee_number):
    """Checking-In at AM (sync function for async call)"""
    today = timezone.now().date()  # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_out_at_am__date=today).exists()  # Check if already checked out for AM

async def clock_out_at_am(employee_number):
    """Clocking-Out at AM"""
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee
    today = timezone.now().date()  # Get today's date

    # Find the shift record for AM and await
    shift_record = await sync_to_async(lambda: ShiftRecord.objects.filter(employee=employee, clock_in_at_am__date=today).first())()
    
    if shift_record:
        # Update clock-out time in a sync-to-async context
        shift_record.clock_out_at_am = timezone.now()
        await sync_to_async(shift_record.save)()
        return shift_record
    return None  # Or raise an exception if no record is found

@sync_to_async
def check_in_at_pm(employee_number):
    """Checking-In at PM (sync function for async call)"""
    today = timezone.now().date()  # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_in_at_pm__date=today).exists()  # Check if already checked in for PM

async def clock_in_at_pm(employee_number):
    """Clocking in for the afternoon shift."""
    today = timezone.now().date()  # Get today's date
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee

    # Check if the employee has already checked in for the afternoon shift
    already_checked_in = await check_in_at_pm(employee_number)  # Await the async call
    
    if not already_checked_in:
        # Use get_or_create to retrieve or create the shift record for today
        shift_record, created = await sync_to_async(ShiftRecord.objects.get_or_create)(
            date=today,
            employee=employee,
            defaults={"clock_in_at_pm": timezone.now()}  # Sets clock_in_at_pm only if a new record is created
        )
        
        # If the record already exists but clock_in_at_pm is not set, update it
        if not created and shift_record.clock_in_at_pm is None:
            shift_record.clock_in_at_pm = timezone.now()
            await sync_to_async(shift_record.save)()  # Save the updated shift record

        return shift_record  # Return the updated or newly created shift record

    return None  # Optional: return None if already checked in, or raise an exception if desired

@sync_to_async
def check_out_at_pm(employee_number):
    """Checking-In at PM (sync function for async call)"""
    today = timezone.now().date()  # Get today's date
    employee = get_object_or_404(Employee, employee_number=employee_number)  # Get employee
    return ShiftRecord.objects.filter(employee=employee, clock_out_at_pm__date=today).exists()  # Check if already checked out for PM

async def clock_out_at_pm(employee_number):
    """Clocking-Out at PM"""
    employee = await get_employee_by_id(employee_number)  # Await async call to fetch employee
    today = timezone.now().date()  # Get today's date

    # Find the shift record for PM and await
    shift_record = await sync_to_async(lambda: ShiftRecord.objects.filter(employee=employee, clock_in_at_pm__date=today).first())()
    
    if shift_record:
        # Update clock-out time in a sync-to-async context
        shift_record.clock_out_at_pm = timezone.now()
        await sync_to_async(shift_record.save)()
        return shift_record
    return None  # Or raise an exception if no record is found

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

async def checking_in_at_am(request):
    """Handles the attendance check-in at AM face recognition."""
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
                        checkin_exists =  await check_in_at_am(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        #If employee hasn't check_in yet
                        if not checkin_exists:
                            try:
                                #Clock-in that employee
                                submit =  await clock_in_at_am(employee_number)
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

async def checking_in_at_pm(request):
    """Handles the attendance check-in at PM face recognition."""
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
                        checkin_exists =  await check_in_at_pm(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        #If employee hasn't check_in yet
                        if not checkin_exists:
                            try:
                                #Clock-in that employee
                                submit =  await clock_in_at_pm(employee_number)
                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked in today's afternoon shift.",
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
                                    "message": f"{employee_data.get('name')} has already checked in for today's afternoon shift.",
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

async def checking_out_at_am(request):
    """Handles the attendance check-in at AM face recognition."""
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
                        checkin_exists =  await check_in_at_am(employee_number)
                        checkout_exists =  await check_out_at_am(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        # If employee has checked in and not checked out, allow clocking out
                        if checkin_exists and not checkout_exists:
                            try:
                                # Clock-Out that employee
                                submit = await clock_out_at_am(employee_number)
                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked out today's morning shift.",
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
                                        "message": f"{employee_data.get('name')} has already checked out for today's morning shift.",
                                    }]
                                }, status=409)
                        
                        # If Employee hasn't Check-In Yet for morning shift.
                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} hasn't checked in for the morning shift.",
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

async def checking_out_at_pm(request):
    """Handles the attendance check-out at PM face recognition."""
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
                        checkin_exists =  await check_in_at_pm(employee_number)
                        checkout_exists =  await check_out_at_pm(employee_number)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        # If employee has checked in and not checked out, allow clocking out
                        if checkin_exists and not checkout_exists:
                            try:
                                # Clock-Out that employee
                                submit = await clock_out_at_pm(employee_number)
                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_number": employee_number,
                                        "profile_image_url": profile_image_url,
                                        "message": f"{employee_data.get('name')} has checked out today's afternoon shift.",
                                    }]
                                }, status=200)
                            except Exception as e:
                                logger.error("Error recording attendance: %s", e)
                                return JsonResponse({
                                    "result": [{
                                        "message": "Error recording attendance. Please try again later."
                                    }]
                                }, status=500)

                        # If Employee already Check-In and Check-Out in the afternoon shift    
                        elif checkin_exists and checkout_exists:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} has already checked out for today's afternoon shift.",
                                }]
                            }, status=409)    #Conflict Status
                        
                        # If Employee hasn't Check-In Yet for afternoon shift.
                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} hasn't checked in for the afternoon shift.",
                                }]
                            }, status=400) #Bad Request

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
                else:
                    messages.error(request, "Please set a password to complete registration.")
                    return redirect('user-registration')

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
        "user_has_social_account": user_has_social_account,
        'user_form': user_form,
        'employee_form': employee_form,
    })

# Upload Face Images
@login_required
def user_face_registration(request, employee_number):
    if request.method == 'POST':
        data = json.loads(request.body)  # Load JSON payload
        image_data = data.get('image')  # Get the base64 image data
        
        try:
            person = Employee.objects.get(employee_number=employee_number)
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
            except Exception as e:
                return JsonResponse({"message": "Error decoding image: " + str(e)}, status=400)

             # Convert image bytes to a NumPy array and read as OpenCV image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return JsonResponse({"message": "No face detected in the image."}, status=400)
            
            # Crop the first detected face and make it square
            (x, y, w, h) = faces[0]
            if w > h:
                pad = (w - h) // 2
                cropped_face = image[y - pad:y + h + pad, x:x + w]
            else:
                pad = (h - w) // 2
                cropped_face = image[y:y + h, x - pad:x + w + pad]

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

    return render(request, 'attendance/face-registration.html', {'employee_number': employee_number})

# Face Verification Process
def face_verification(request):
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

@login_required
def dashboard(request):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=request.user)
        total_employees = Employee.objects.all().count()
        
        # Get today's date and time
        today = timezone.now().date()
        manila_tz = pytz_timezone('Asia/Manila')
        current_time = timezone.now().astimezone(manila_tz)  # Convert to Manila timezone
        current_hour = current_time.hour
        is_am = 6 <= current_hour < 12  # From 6:00 AM to 11:59 AM

        

        # Fetch attendance based on the time of day
        if is_am:
            timemode = 'am'
        else:
            timemode = 'pm'


        shiftstatus = ShiftRecord.objects.filter(employee=employee, date=today).first()
        shiftlogs = ShiftRecord.objects.filter(employee=employee).order_by('-date')
        check_in_time = getattr(shiftstatus, f"clock_in_at_{timemode}", None)
        check_out_time = getattr(shiftstatus, f"clock_out_at_{timemode}", None)

        # Determine button states based on check-in and check-out times
        can_check_in = check_in_time is None
        can_check_out = check_in_time is not None and check_out_time is None

        first_day_of_month = today.replace(day=1)
        last_day_of_month = (first_day_of_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)

        # Generate a list of all dates in the current month
        all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

        # Fetch attendance records for the employee for the current month
        shift_records = ShiftRecord.objects.filter(employee=employee, date__month=today.month, date__year=today.year)

        # Count of employees who are active today
        active_today = ShiftRecord.objects.filter(date=today).count()

        # Fetch the number of active employees from the previous day
        yesterday = today - timedelta(days=1)
        active_yesterday = ShiftRecord.objects.filter(date=yesterday).count()

        # Calculate the percentage increase or decrease in active employees
        if active_yesterday > 0:
            active_today_percentage = ((active_today - active_yesterday) / active_yesterday) * 100
        else:
            active_today_percentage = 0  # Avoid division by zero if there were no active employees yesterday
        
        # Determine if it's an increase or decrease
        if active_today_percentage > 0:
            active_today_trend = "increase"
        elif active_today_percentage < 0:
            active_today_trend = "decrease"
        else:
            active_today_trend = "no change"

        # Pass all the necessary data to the template
        return render(request, 'user/dashboard.html', { 
            'user': user, 
            'employee': employee,
            'total_employees': total_employees,
            'all_dates': all_dates,
            'shift_records': shift_records,
            'active_today': active_today,
            'active_today_percentage': active_today_percentage,
            'active_today_trend': active_today_trend,  # Pass the trend to the template
            'shiftstatus': shiftstatus,
            'shiftlogs': shiftlogs,
            'timemode': timemode,
            'check_in_time': check_in_time,
            'check_out_time': check_out_time,
            'can_check_in': can_check_in,
            'can_check_out': can_check_out,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')

@login_required
def attendance_sheet(request):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=request.user)

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
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
@login_required
def attendance_sheet_date(request, month, year):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=user)
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
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')

def employee_attendance_sheet(request, month, year, employee_number=None):
    user = request.user  # Get the logged-in user
    
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
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    
@login_required
def profile_view(request):
    user = request.user  # Get the logged-in user
    
    try:
        employee = Employee.objects.get(user=request.user)

        # Pass all the necessary data to the template
        return render(request, 'user/profile.html', { 
            'user': user, 
            'employee': employee,
        })
    
    except Employee.DoesNotExist:
        # Redirect to employee registration to continue
        return redirect('employee-registration')
    

import datetime
import os
import base64
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from django.conf import settings
from django.utils import timezone
from asgiref.sync import sync_to_async
import logging
import json
from attendance.forms import EmployeeRegistrationForm
from attendance.models import CheckIn, Employee, FaceImage


logger = logging.getLogger(__name__)

from .recognize_faces import recognize_faces_from_image, detect_fake_face  # Ensure these functions are implemented
@sync_to_async
def async_recognize_faces(img_data):
    """Asynchronous wrapper for face recognition."""
    return recognize_faces_from_image(img_data)

@sync_to_async
def async_detect_fake_face(img_data):
    """Asynchronous wrapper for fake face detection."""
    return detect_fake_face(img_data)

@sync_to_async
def get_employee_by_id(employee_id):
    """Fetch employee from the database by employee_id."""
    return get_object_or_404(Employee, employee_id=employee_id)

@sync_to_async
def check_in(employee_id):
    today = timezone.now().date() # Get Todat's Date
    employee = get_object_or_404(Employee, employee_id=employee_id) #Get Employee Id Number and find that person
    return CheckIn.objects.filter(employee=employee, timestamp__date=today).exists() #Check if the person already check-in

@sync_to_async
def mark_attendance(employee_id):
    employee = get_object_or_404(Employee, employee_id=employee_id)
    return CheckIn.objects.create(employee=employee)

def check_attendance(request):
    """Renders the attendance page."""
    return render(request, 'attendance/check.html')

async def attendance(request):
    """Handles the attendance face recognition."""
    logger.info("Received request: %s", request.body)  # Log the incoming request
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

            result = await async_detect_fake_face(data['image'])

            if result == 'Real':
                verify = await async_recognize_faces(data['image'])  # Again, use original data
                logger.info("Verification result: %s", verify)

                # Assuming verify contains a list of dictionaries with name and employee_id
                if verify and isinstance(verify, list) and len(verify) > 0:
                    employee = None
                    employee_data = verify[0]
                    employee_id = employee_data.get('employee_id')
                    
                    if employee_id:  # Check if employee_id is not None
                        # Fetch employee profile using employee_id asynchronously
                        employee = await get_employee_by_id(employee_id)
                    
                    if employee:  # Check if employee is found
                        # Check if the employee has already checked in today
                        checkin_exists = await check_in(employee_id)
                        profile_image_url = request.build_absolute_uri(employee.avatar_url)
                        
                        if not checkin_exists:
                            
                            try:
                                submit = await mark_attendance(employee_id)
                                print(submit)
                                return JsonResponse({
                                    "result": [{
                                        "name": employee_data.get('name'),
                                        "employee_id": employee_id,
                                        "profile_image_url": profile_image_url
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
                                    "employee_id": employee_id,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} has already checked in today.",
                                }]
                            }, status=400)

                    else:
                        logger.warning("Employee with ID %s not found.", employee_id)
                        return JsonResponse({"result": [{"message": "Employee not found."}]}, status=404)
            
            if result == 'Fake':
                print('Spoofing Detected')
                return JsonResponse({"result":[{"message": "Possible Spoofing Detected"}]}, status=200)
            

        except json.JSONDecodeError as e:
            logger.error("JSON Decode Error: %s", e)
            return JsonResponse({"error": ["Invalid JSON"]}, status=400)
        except Exception as e:
            logger.error("Error: %s", e)
            return JsonResponse({"error": [str(e)]}, status=500)

    return JsonResponse({"result":[{"status": "No Content"}]}, status=200)

# Register New Employee
def employee_registration(request):
    if request.method == 'POST':
        form = EmployeeRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            person = form.save()  # Save the employee
            return redirect('facial-registration', employee_id=person.employee_id)  # Redirect to the face upload view
        else:
            print("Form errors:", form.errors)  # Debug statement for form errors
    else:
        form = EmployeeRegistrationForm()
    
    return render(request, 'attendance/employee-registration.html', {'form': form})

# Upload Face Images
# Upload Face Images
def upload_face_images(request, employee_id):
    if request.method == 'POST':
        data = json.loads(request.body)  # Load JSON payload
        image_data = data.get('image')  # Get the base64 image data
        person = Employee.objects.get(employee_id=employee_id)

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
            employee_folder = os.path.join(settings.MEDIA_ROOT, 'known_faces', f"{person.employee_id} - {person.first_name} {person.last_name}")
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
            image_filename = f"{person.employee_id}_face_{timestamp}.jpg"
            image_path = os.path.join(employee_folder, image_filename)

            # Save the image as a file
            with open(image_path, 'wb') as destination:
                destination.write(image_bytes)

            # Optionally, create the face image entry linked to the employee
            FaceImage.objects.create(employee=person, image=image_path)

            return JsonResponse({"message": "Image uploaded successfully."}, status=200)

        return JsonResponse({"message": "No image data provided."}, status=400)

    return render(request, 'attendance/face-registration.html', {'employee_id': employee_id})



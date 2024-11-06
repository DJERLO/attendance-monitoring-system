import base64

import datetime
import io
import json
import os
from venv import logger
import cv2
from django.conf import settings
from django.http import JsonResponse
from ninja import Body, NinjaAPI
import numpy as np
from attendance.model_evaluation import detect_face_spoof
from attendance.models import Employee, FaceImage
from attendance.views import async_detect_fake_face, async_recognize_faces, check_in_at_am, check_in_at_pm, check_out_at_am, check_out_at_pm, clock_in_at_am, clock_in_at_pm, clock_out_at_am, clock_out_at_pm, get_employee_by_id
from pydantic import BaseModel
from typing import List, Optional
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

api = NinjaAPI(
    title="Attendance Monitoring System",
    version="1.0.0",
    description="The Facial Recognition Attendance Monitoring System includes a set of APIs that facilitate communication between the frontend user interface and the backend services. These APIs are designed to handle user interactions, data processing, and system responses efficiently. Below are the specifications for each API endpoint.",
)

class SuccessResponse(BaseModel):
    result: List[dict]

class ErrorResponse(BaseModel):
    error: List[str]

class EmployeeCheckInDetail(BaseModel):
    name: str
    employee_number: str
    profile_image_url: str
    message: str

class EmployeeCheckInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]
    
class AlreadyCheckedInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]

class StatusMessage(BaseModel):
    message: str

class SpoofingDetectedResponse(BaseModel):
    result: List[StatusMessage]
    
class NotFoundResponse(BaseModel):
    result: List[StatusMessage]


# Define the async check_in_morning_shift function as an API endpoint
@api.post('/attendance/am/check-in/', tags=["Check-In"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def check_in_morning_shift(request, base64_image: str):
    """
    Handles the attendance check-in for the morning shift. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    """
    
    logger.info("Received request: %s", request.body)
    
    try:
        img_data = base64_image # Example of ImgData is data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...

        # Extract base64 string from data URL
        if img_data.startswith('data:image/jpeg;base64,'):
            img_data = img_data.replace('data:image/jpeg;base64,', '')
            img_data = base64.b64decode(img_data)
        else:
            return JsonResponse({"error": "Invalid image format"}, status=400)

        # This function checks for face spoofing attacks
        class_idx, confidence, message = await async_detect_fake_face(img_data)

        # Check if the img_data is Real or Fake
        if message == 'Real':
            # Start the Facial Recognition Process
            verify = await async_recognize_faces(img_data)  
            logger.info("Verification result: %s", verify)

            # Verify may contain employee's name and employee_id
            if verify and isinstance(verify, list) and len(verify) > 0:
                employee = None
                employee_data = verify[0]
                employee_number = employee_data.get('employee_number')
                
                if employee_number:  # Check if employee_id is not None
                    # Fetch employee profile using employee_id asynchronously
                    employee = await get_employee_by_id(employee_number)
                
                if employee:  # Check if employee is found
                    # Check if the employee has already checked in today
                    checkin_exists = await check_in_at_am(employee_number)
                    profile_image_url = request.build_absolute_uri(employee.avatar_url)
                    
                    # If employee hasn't checked in yet
                    if not checkin_exists:
                        try:
                            # Clock-in that employee
                            submit = await clock_in_at_am(employee_number)
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

# Define the async check_in_afternoon_shift function as an API endpoint
@api.post('/attendance/pm/check-in/', tags=["Check-In"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def check_in_afternoon_shift (request, base64_image: str):
    """
    Handles the attendance check-in for the afternoon shift. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    """
    logger.info("Received request: %s", request.body)
    if request.method == 'POST':
        try:
            img_data = base64_image

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

# Define the async check_out_morning_shift function as an API endpoint
@api.post('/attendance/am/check-out/', tags=["Check-Out"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def check_out_morning_shift (request, base64_image: str):
    """
    Handles the attendance check-out for the morning shift. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    """
    logger.info("Received request: %s", request.body)
    if request.method == 'POST':
        try:
            
            img_data = base64_image

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
                                        "message": f"{employee_data.get('name')} has checked out today.",
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
                                        "message": f"{employee_data.get('name')} has checked out today.",
                                    }]
                                }, status=409)
                        
                        # If Employee hasn't Check-In Yet for morning shift.
                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} hasn't checked in yet!",
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

# Define the async check_out_afternoon_shift function as an API endpoint
@api.post('/attendance/pm/check-out/', tags=["Check-Out"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def check_out_afternoon_shift (request, base64_image: str):
    """
    Handles the attendance check-in for the morning shift. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    """
    logger.info("Received request: %s", request.body)
    if request.method == 'POST':
        try:
            
            img_data = base64_image

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
                                        "message": f"{employee_data.get('name')} has checked out today.",
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
                                    "message": f"{employee_data.get('name')} has already checked out for today afternoon shift.",
                                }]
                            }, status=409)    #Conflict Status
                        
                        # If Employee hasn't Check-In Yet for afternoon shift.
                        else:
                            return JsonResponse({
                                "result": [{
                                    "name": employee_data.get('name'),
                                    "employee_number": employee_number,
                                    "profile_image_url": profile_image_url,
                                    "message": f"{employee_data.get('name')} hasn't checked in yet!",
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

class SuccessFaceRegistration(BaseModel):
    message: str

class EmployeeNotFound(BaseModel):
    message: str

class ErrorAtDecodingImage(BaseModel):
    message: str

class ErrorAtFaceRegistration(BaseModel):
    message: str

@api.post('face/registration', tags=["Registration"], response={
    201: SuccessFaceRegistration, 
    400: ErrorAtDecodingImage,  
    404: EmployeeNotFound, 
    500: ErrorAtFaceRegistration
})
def user_face_registration(request):
    """
    Endpoint to register an employee's face image.

    **Request Payload (JSON):**
    ```
    {
        "employee_number": "string",  # Employee ID or number
        "base64_image": "string"  # Base64-encoded image data
    }
    ```

    **Response Codes:**
    - 201: Success, face image registered
    - 400: No Face Detected or other validation issues.
    - 404: Employee not found
    - 500: Internal server error during registration

    **Example JSON Payload:**
    ```json
    {
        "employee_number": "12345",
        "base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAA..."
    }
    ```

    Parameters:
    - request: The request object containing JSON payload.

    Returns:
    - JSON response indicating success or error message.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            employee_number = data.get("employee_number")
            base64_image = data.get("base64_image")

            # Check if necessary data is provided
            if not employee_number or not base64_image:
                return JsonResponse({"message": "Employee number and image data are required."}, status=400)

            try:
                # Employee validation
                person = Employee.objects.get(employee_number=employee_number)
            except Employee.DoesNotExist:
                return JsonResponse({"message": "Employee not found."}, status=404)

            # Decode base64 image data
            image_bytes = base64.b64decode(base64_image.split(",")[1]) if "base64," in base64_image else base64.b64decode(base64_image)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Face detection and processing
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return JsonResponse({"message": "No face detected in the image."}, status=400)

            (x, y, w, h) = faces[0]
            cropped_face = image[y:y+h, x:x+w]
            square_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)

            # Directory creation for storing images
            employee_folder = os.path.join(settings.MEDIA_ROOT, 'known_faces', f"{person.employee_number} - {person.first_name} {person.last_name}")
            os.makedirs(employee_folder, exist_ok=True)

            # Image management for limiting the count of stored images
            existing_face_images = FaceImage.objects.filter(employee=person)
            if existing_face_images.count() >= 5:
                oldest_face_image = existing_face_images.order_by('uploaded_at').first()
                if oldest_face_image and os.path.exists(oldest_face_image.image.path):
                    os.remove(oldest_face_image.image.path)
                    oldest_face_image.delete()

            # Save the new image with a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{person.employee_number}_face_{timestamp}.jpg"
            image_path = os.path.join(employee_folder, image_filename)
            cv2.imwrite(image_path, square_face)

            # Store image metadata in the database
            FaceImage.objects.create(employee=person, image=image_path)

            return JsonResponse({"message": "Image uploaded successfully."}, status=201)

        except Exception as e:
            return JsonResponse({"message": "An error occurred: " + str(e)}, status=500)

    return JsonResponse({"message": "Invalid request method."}, status=405)

class SuccessCheckFaceSpoofing(BaseModel):
    class_idx: int
    confidence: float
    message: str
    coordinates: dict  # Using dict to represent the coordinates (x, y, w, h)

class SuccessAntiFaceSpoofing(BaseModel):
    result: List[SuccessCheckFaceSpoofing]

class InvalidImageFormat(BaseModel):
    result: str

class ErrorAtFaceSpoofing(BaseModel):
    error: str

@api.post('face/anti-spoof', tags=["Anti-Spoof"], response={
    200: SuccessAntiFaceSpoofing, 
    404: InvalidImageFormat,  
    500: ErrorAtFaceRegistration
}) 
def anti_spoof(request):
    """
    Endpoint for anti-spoofing face recognition.

    This API accepts a POST request with an image provided as a base64-encoded 
    data URI. The image is processed to detect faces, and each detected face 
    is analyzed for spoofing attempts. The response includes the spoofing 
    detection result for each face, including confidence levels and detection 
    messages. The system uses OpenCV to detect faces and a model to assess 
    spoofing attempts.

    Request:
        - A JSON object containing a 'base64_image' field with a base64-encoded 
          image data URI.

    Response:
        - A JSON object with 'results' containing:
            - 'class_idx': Index indicating the spoof detection result.
            - 'confidence': Confidence level of the detection.
            - 'message': A message indicating the spoof detection outcome.
            - 'coordinates': Coordinates of the detected face in the image.
        - If an error occurs, a JSON object with an 'error' message and HTTP status 400.
    
    **Example JSON Payload:**
    ```json
    {
        "base64_image": "string"  # Base64-encoded image data
    }
    ```

    **Response Codes:**
    - 200: Success
    - 400: No Face Detected or other validation issues.
    - 500: Internal server error during checking and vaidation

    **Example JSON Payload:**
    ```json
    {
        "base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAA..."
    }
    ```
    If no faces are detected or an error occurs, an empty 'results' list or 
    error message is returned.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            image_data = data.get("base64_image").split(",")[1]
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

               # Append the results for this particular face to the results list
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

            # Convert processed image back to base64
            _, img_buffer = cv2.imencode('.jpg', image_np)  # Convert the processed image to JPG format
            processed_image_base64 = base64.b64encode(img_buffer).decode('utf-8')  # Encode image to base64

            return JsonResponse({
                'results': results,
                'processed_image': f"data:image/jpeg;base64,{processed_image_base64}"  # Send base64 image as part of the response
            })

            return JsonResponse({'results': results})  # Return results for all detected faces
        except Exception as e:
            return JsonResponse({'error': "No Data URI image Found"}, status=404)


    
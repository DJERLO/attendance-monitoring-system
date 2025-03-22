import base64
import io
import json
import os
import cv2
import numpy as np
from ninja import NinjaAPI, Query, Form, File
from ninja_jwt.authentication import JWTAuth
from ninja_jwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.http import JsonResponse
from ninja_jwt.exceptions import TokenError
from venv import logger
from datetime import date, datetime
from django.conf import settings
from django.http import JsonResponse
from ninja.files import UploadedFile
from django.db.models import F
from django.db.models.functions import Concat
from django.db.models import Value
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from attendance.model_evaluation import detect_face_spoof
from attendance.models import Employee, FaceImage, ShiftRecord
from attendance.schema import AlreadyCheckedInResponse, AttendanceEmployeeFilterSchema, EmployeeCheckInResponse, EmployeeFacialRegistration, EmployeeNotFound, EmployeeRegistrationResponse, EmployeeRegistrationSchema, ErrorAtDecodingImage, ErrorAtFaceRegistration, ErrorResponse, ImageSchema, InvalidImageFormat, NotFoundResponse, ShiftRecordSchema, SpoofingDetectedResponse, SuccessAntiFaceSpoofing, SuccessFaceRegistration, Unauthorized, UserNotFound, UserRegisterResponse, UserRegisterSchema
from attendance.views import async_detect_fake_face, async_recognize_faces, check_in, check_out, clock_in, clock_out,  get_employee_by_id
from pydantic import BaseModel
from typing import List
from PIL import Image


# Initialize API with JWT authentication
api = NinjaAPI(
    title="Attendance Monitoring System",
    version="1.0.0",
    description="The Facial Recognition Attendance Monitoring System includes a set of APIs that facilitate communication between the frontend user interface and the backend services. These APIs are designed to handle user interactions, data processing, and system responses efficiently. Below are the specifications for each API endpoint.",
)

# Custom Token Generation API
@api.post("/token", tags=['Auth'])
def generate_token(request, username: str, password: str):
    user = User.objects.filter(username=username).first()
    if user and user.check_password(password):
        refresh = RefreshToken.for_user(user)
        return {
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "user": {
                "first_name": user.first_name,
                "email": user.email
            }
        }
    return JsonResponse({"error": "Invalid credentials"}, status=400)

# Custom Refresh Token API
@api.post("/token/refresh/", tags=['Auth'])
def refresh_access_token(request, refresh_token: str):
    """
    Takes a valid refresh token and returns a new access token.
    """
    try:
        refresh = RefreshToken(refresh_token)  # Validate refresh token
        new_access_token = str(refresh.access_token)  # Generate new access token
        return {"access": new_access_token}
    except TokenError:  # Catch invalid or expired token
        return JsonResponse({"error": "Invalid or expired refresh token"}, status=401)
    except Exception as e:  # Catch unexpected errors
        return JsonResponse({"error": str(e)}, status=500)

# Override the verify token response
@api.post("/token/verify/", tags=['Auth'])
def verify_token(request, token: str):
    from ninja_jwt.tokens import UntypedToken

    try:
        UntypedToken(token)  # This checks if the token is valid
        return {"message": "Token is valid"}
    except TokenError:  # Catch JWT errors properly
        return JsonResponse({"error": "Invalid or expired token"}, status=401)
    
@api.post("/token/blacklist/", tags=['Auth'])
def blacklist_token(request, refresh_token: str):
    try:
        refresh = RefreshToken(refresh_token)  # Validate refresh token
        refresh.blacklist()  # Blacklist the token
        return JsonResponse({"message": "Token has been blacklisted"}, status=200)
    except TokenError:  # Catch invalid or expired token
        return JsonResponse({"error": "Invalid or expired token"}, status=401)
    except Exception as e:  # Catch unexpected errors
        return JsonResponse({"error": str(e)}, status=500)

# API Endpoints for User Registration
@api.post("/user/register", summary="User Registration", tags=["User Registration"], auth=JWTAuth(), response={
    201: UserRegisterResponse,
    401: Unauthorized
})
def register_user(request, payload: Form[UserRegisterSchema]):
    try:
        user = User.objects.create_user(
            username = payload.username,
            first_name = payload.first_name,
            last_name = payload.last_name,
            email = payload.email,
            password = payload.password,
        )
        return JsonResponse({"message": "User registered successfully", "user_id": user.id}, status=201)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        return JsonResponse({"error": "An unexpected error occurred"}, status=500)
    
@api.post("/user/employee/register", summary="Employee Registration", tags=["User Registration"], auth=JWTAuth(),  response={
    201: EmployeeRegistrationResponse,
    401: Unauthorized,
    404: UserNotFound,
})
def register_employee(request,  payload: Form[EmployeeRegistrationSchema], profile_image: UploadedFile = File(...)):
    try:
        # Check if the user exists
        try:
            user = User.objects.get(id=payload.user_id)
        except User.DoesNotExist:
            return JsonResponse({"error": "User does not exist"}, status=404)

        # Check if an Employee is already linked to this user
        if hasattr(user, 'employee'): 
            return JsonResponse({"error": "An employee record already exists for this user"}, status=409)


        # Create the Employee model
        employee = Employee.objects.create(
            user=user,
            employee_number=payload.employee_number,
            first_name=payload.first_name,
            middle_name=payload.middle_name,
            last_name=payload.last_name,
            email=user.email,
            contact_number=payload.contact_number,
            profile_image=profile_image
        )
        return JsonResponse({"message": "Employee registered successfully", "employee_number": employee.number}, status=201)

    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": "An unexpected error occurred"}, status=500)
    
@api.post('user/face-registration', summary="Facial Registration for Face Recognition", auth=JWTAuth(), tags=["User Registration"], response={
    201: SuccessFaceRegistration, 
    400: ErrorAtDecodingImage,
    401: Unauthorized,  
    404: EmployeeNotFound, 
    500: ErrorAtFaceRegistration
})
def user_face_registration(request, payload: EmployeeFacialRegistration):
    """
    Endpoint to register an employee's face image.

    This API accepts a POST request with an image provided as a base64-encoded data URI. The image is processed for Face Recognition System.

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
# End (User Registration)

#Start (Attendance Logging) Endpoints

@api.get('/attendance', summary="Retrieve all attendance shift records or filter by employee number to fetch specific shift records.", tags=["Attendance Logging"], response=List[ShiftRecordSchema], auth=JWTAuth())
def get_attendance(request, filters: AttendanceEmployeeFilterSchema = Query(...)):
     # Fetch attendance records and include related employee details
    attendance = ShiftRecord.objects.select_related('employee').all()

    # Apply filters for date
    if filters.date:
        attendance = attendance.filter(date=filters.date)

    # Apply filters for employee_number
    if filters.employee_number:
        attendance = attendance.filter(employee__employee_number=filters.employee_number)

    # Annotate employee details in the queryset
    attendance = attendance.annotate(
        employee_number=F('employee__employee_number'),
        name=Concat(F('employee__first_name'), Value(' '), F('employee__last_name'))
    )

    # Return serialized data
    return list(attendance.values(
        'employee_number',
        'name',
        'date',
        'clock_in',
        'clock_out',
        'status',
    ))

# Define the async check_in_morning_shift function as an API endpoint
@api.post('/attendance/check-in/', summary="Attendance Check-In", auth=JWTAuth(), tags=["Attendance Logging"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse,
    401: Unauthorized, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def checking_in(request, payload: ImageSchema):
    """
    Handles the attendance check-in. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    """
    
    logger.info("Received request: %s", request.body)
    
    try:
        data = json.loads(request.body)
        img_data = data.get("base64_image")

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
                    checkin_exists = await check_in(employee_number)
                    profile_image_url = request.build_absolute_uri(employee.avatar_url)
                    
                    # If employee hasn't checked in yet
                    if not checkin_exists:
                        try:
                            # Clock-in that employee
                            submit = await clock_in(employee_number)
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
@api.post('/attendance/check-out/',summary="Attendance Check-Out", auth=JWTAuth(), tags=["Attendance Logging"], response={
    200: EmployeeCheckInResponse, 
    400: AlreadyCheckedInResponse,
    401: Unauthorized, 
    403: SpoofingDetectedResponse, 
    404: NotFoundResponse, 
    500: ErrorResponse})
async def checking_out(request, payload: ImageSchema):
    """
    Handles the attendance check-out. This endpoint accepts image data for facial recognition to verify employee identity. Upon successful verification, it records the attendance of the employee. If the employee has already checked in for the day or if face spoofing is detected, appropriate responses are returned to inform the user of the status of their check-in attempt.
    
    """
    logger.info("Received request: %s", request.body)
    if request.method == 'POST':
        try:
            
            data = json.loads(request.body)
            img_data = data.get("base64_image")

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




@api.post('face/anti-spoof', summary="Face Anti-Spoofing Detection", tags=["Face Anti-Spoofing"], response={
    200: SuccessAntiFaceSpoofing, 
    404: InvalidImageFormat,  
    500: ErrorAtFaceRegistration
}) 
def anti_spoof(request, payload: ImageSchema):
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
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            img_data = data.get("base64_image")
            
            
            # Extract base64 string from data URL
            if img_data.startswith('data:image/jpeg;base64,'):
                img_data = img_data.replace('data:image/jpeg;base64,', '')
                img_data = base64.b64decode(img_data)
            else:
                return JsonResponse({"error": "Invalid image format"}, status=400)
            
            image = Image.open(io.BytesIO(img_data)).convert("RGB")  # Convert to RGB

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
            }, status=200)

        except Exception as e:
            return JsonResponse({'error': e}, status=404)
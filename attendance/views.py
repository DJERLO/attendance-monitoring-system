import base64
from django.shortcuts import render
from django.http import JsonResponse
from asgiref.sync import sync_to_async
from .recognize_faces import recognize_faces_from_image, detect_fake_face  # Ensure these functions are implemented
import logging
import json

logger = logging.getLogger(__name__)

@sync_to_async
def async_recognize_faces(img_data):
    """Asynchronous wrapper for face recognition."""
    return recognize_faces_from_image(img_data)

@sync_to_async
def async_detect_fake_face(img_data):
    """Asynchronous wrapper for fake face detection."""
    return detect_fake_face(img_data)

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

            # Check for fake face detection
            face_status = await async_detect_fake_face(data['image'])  # Use the original base64 data for detection
            
            if face_status == "Real":
                # If it's a real face, proceed with face recognition
                verify = await async_recognize_faces(data['image'])  # Again, use original data
                return JsonResponse({"result": verify}, status=200)
            
            if face_status == "Fake":
                print("Possible Face Detection Error")
                return JsonResponse({"result": [{"message": "Face not recognized"}]}, status=200)

        except json.JSONDecodeError as e:
            logger.error("JSON Decode Error: %s", e)
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error("Error: %s", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "No Content"}, status=204)

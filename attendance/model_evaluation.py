import base64
import io
import os
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models

# Load the model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'training_resnet50.pth')

model = models.resnet50(weights='IMAGENET1K_V1')

# Adjust the final fully connected layer (fc) for 2 classes
num_features = model.fc.in_features  # Access the fc layer instead of classifier
model.fc = nn.Linear(num_features, 2)  # Adjust for the number of classes (2 in this case)

# Load your trained model
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

model.eval()  # Set the model to evaluation mode

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your image transformations
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),  # Center crop to 224x224 (or another size)
    transforms.Resize((224, 224)),  # Resize if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def detect_face_spoof(base64_image):

    # Decode the base64 image if not already in bytes
    if isinstance(base64_image, str):
        base64_image = base64.b64decode(base64_image)

    image = Image.open(io.BytesIO(base64_image)).convert("RGB")  # Convert to RGB

    # Apply transformations
    image_tensor = data_transforms(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move tensor to the appropriate device

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map prediction to class label
    class_idx = predicted.item()
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()

    message = 'Real' if class_idx == 1 else 'Fake'

    return class_idx, confidence, message

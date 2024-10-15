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
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')

model = models.mobilenet_v2(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # Adjust for number of classes

# Load your trained model
try:
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),           # Convert images to tensors
])

def predict_from_base64(base64_image):
    # Decode the base64 image
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB

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

    message = 'Real' if class_idx == 0 else 'Fake'

    return class_idx, confidence, message

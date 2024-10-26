import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
from PIL import Image
import kagglehub 
from dotenv import load_dotenv  # Import to load environment variables

# Load environment variables from .env file
load_dotenv()
real_count = 0
fake_count = 0
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.getenv('DATASET_PATH')

# Step 1: Define dataset download path from the .env file
path = kagglehub.dataset_download("minhnh2107/casiafasd")
print("Dataset downloaded to:", path)  # Print the download path

MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, 'training_resnet50.pth')

# Define paths for training images
train_color_folder = os.path.join(path, 'train_img', 'train_img', 'color')
train_depth_folder = os.path.join(path, 'train_img', 'train_img', 'depth')

# Step 2: Create a custom dataset class to load images and labels
class RealFakeDataset(torch.utils.data.Dataset):
    def __init__(self, color_folder, depth_folder, transform=None):
        self.color_folder = color_folder
        self.depth_folder = depth_folder
        self.transform = transform
        self.images = []
        self.real_count = 0  # Initialize instance variables
        self.fake_count = 0

        # Load images from color and depth folders
        self._load_images(color_folder)
        self._load_images(depth_folder)

    def _load_images(self, folder):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):
                if 'real' in filename:
                    label = 1  # 1 for Real
                    self.real_count += 1  # Increment real count
                elif 'fake' in filename:
                    label = 0  # 0 for Fake
                    self.fake_count += 1  # Increment fake count
                else:
                    print(f"Warning: '{filename}' does not contain 'real' or 'fake'. Skipping.")
                    continue
                self.images.append((os.path.join(folder, filename), label))

        # Log the total number of images loaded
        print(f"Total images loaded from {folder}: {len(self.images)}")
        # Count the number of real and fake images
        print(f"Count - Real: {self.real_count}, Fake: {self.fake_count}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return ['Fake', 'Real']  # Define class names for convenience

# Step 3: Define your dataset and transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
])

# Load your dataset
train_dataset = RealFakeDataset(train_color_folder, train_depth_folder, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

real_count = train_dataset.real_count  # Access the real count from the dataset
fake_count = train_dataset.fake_count  # Access the fake count from the dataset

# Step 4: Define your model (using ResNet-50 instead of MobileNetV2)
model = models.resnet50(weights='IMAGENET1K_V1')  # Use pre-trained weights
num_features = model.fc.in_features  # Get the number of input features to the final fully connected layer

# Print the number of input features to the final layer
print("Number of input features to the final layer:", num_features)
print("Train dataset classes: ", len(train_dataset.classes))

# Adjust the final fully connected layer for the number of classes
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # 2 classes: Real and Fake

# Step 5: Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = torch.tensor([len(train_dataset) / (2 * real_count), len(train_dataset) / (2 * fake_count)]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 10  # Set number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Step 7: Save the trained model    
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)

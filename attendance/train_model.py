import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim

# Step 1: Define your dataset path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, 'real_and_fake_face')

# Step 2: Define your dataset and transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the desired size
    transforms.ToTensor(),           # Convert images to tensors
])

# Load your dataset
train_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 3: Define your model (using a pre-trained model for simplicity)
from torchvision.models import ResNet18_Weights  # Import the weight enum
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Use weights parameter
num_features = model.fc.in_features

# Print the number of input features to the final layer
print("Number of input features to the final layer:", num_features)
print("train dataset: ", len(train_dataset.classes))

# Adjust the final fully connected layer for the number of classes
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Adjust for number of classes

# Step 4: Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
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

# Step 6: Save the trained model
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')  # Adjust the path as needed
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to", MODEL_PATH)

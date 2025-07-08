import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 🖼️ Path to the directory containing test images
image_directory = r'train_test_data/testing_data/cat'

# Step 1: Load pre-trained VGG16
model = models.vgg16(pretrained=True)
model.eval()

# Step 2: Preprocess the image (defined once)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Iterate through all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Filter for image files
        image_path = os.path.join(image_directory, filename)

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Preprocess the image
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)

        # Get predicted class
        _, predicted_class = output.max(1)
        print(f"Image: {filename}, Predicted class index: {predicted_class.item()}")

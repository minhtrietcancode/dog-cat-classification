import torch
from torchvision import transforms
from PIL import Image
import os
from model import DogCatClassifier # Import the model we defined

# Define VGG mean and standard deviation for normalization (must match training)
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def predict_image(model_path, image_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model and load the trained weights
    model = DogCatClassifier(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # Define transformations for the input image (must match validation/test transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ])

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0) # Add batch dimension
        image = image.to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).item() # Convert sigmoid output to 0 or 1

    # Interpret prediction
    if prediction == 0:
        return "Cat"
    else:
        return "Dog"

if __name__ == '__main__':
    # Example Usage:
    # Ensure 'dog_cat_classifier_best.pth' is in the 'src/model/' directory
    # And provide a path to a new image you want to classify.
    
    # For demonstration, let's assume you have an image in the data/ directory.
    # Replace with the actual path to your downloaded model and a test image.
    
    model_weights_path = 'src/model/dog_cat_classifier_best.pth' # Relative to project root
    
    # !!! IMPORTANT !!!
    # Replace this with the actual path to an image you want to classify.
    # For example, if you have a dog image at 'data/dog/dog_example.jpg'
    # you would use 'data/dog/dog_example.jpg' (relative to project root)
    # or an absolute path.

    # Example: Classify a cat image from your test data
    example_image_path_cat = 'train_test_data/testing_data/cat/cat_0251.jpg' 
    # Example: Classify a dog image from your test data
    example_image_path_dog = 'train_test_data/testing_data/dog/dog_0251.jpg' 

    print(f"Classifying image: {example_image_path_cat}")
    predicted_class_cat = predict_image(model_weights_path, example_image_path_cat)
    if predicted_class_cat:
        print(f"The image is predicted to be a: {predicted_class_cat}")

    print(f"\nClassifying image: {example_image_path_dog}")
    predicted_class_dog = predict_image(model_weights_path, example_image_path_dog)
    if predicted_class_dog:
        print(f"The image is predicted to be a: {predicted_class_dog}") 
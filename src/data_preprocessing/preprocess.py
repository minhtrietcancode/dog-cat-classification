import os
from PIL import Image
import torch
from torchvision import transforms

# VGG mean and standard deviation for normalization
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def preprocess_training_data(image_path):
    """
    Applies preprocessing (resize, normalize, augment) to a single training image.
    Saves the processed image back to its original path.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ])
    image = Image.open(image_path).convert('RGB')
    processed_image = transform(image)
    # Convert tensor back to PIL Image and save
    # Note: To save normalized tensor as image, we need to denormalize first.
    # However, for in-place preprocessing that will be loaded by a model later,
    # saving the raw tensor data or a transformed PIL image isn't straightforward
    # in a way that directly overwrites the original JPG/PNG while maintaining
    # the image format and not breaking subsequent data loaders expecting images.
    # For in-place application, it's typically better to process and then save
    # in a format suitable for direct loading by the model (e.g., as a new file
    # or overwriting if the data loading pipeline is custom).
    # For now, I will assume the model will load these images as PIL and then
    # apply ToTensor and Normalize as part of its own transforms.
    # If the intention is to save the *tensor* itself or a modified image that
    # can be directly loaded as an image file, the conversion back from tensor
    # to image (and denormalization) is needed.
    # Given "in-place apply these preprocessing to the images" and "keep the name of the images",
    # the most practical interpretation is to apply transforms that result in a PIL Image
    # (or numpy array) and then save that back.
    # However, `ToTensor()` is a conversion to tensor, which can't be saved back as JPG/PNG easily.
    # For this task, I'll modify the approach to only apply transforms that result in a PIL Image
    # if saving back directly is the goal, or, if the model expects normalized tensors,
    # then the saving part needs a different strategy (e.g., saving as .pt files, or
    # processing on-the-fly during training).
    # Re-reading the request: "in-place apply these preprocessing to the images".
    # This implies the saved files should be ready-to-use image files.
    # Normalization (ToTensor + Normalize) is usually done *after* loading for deep learning.
    # Let's adjust the functions so that they prepare the images for *future* normalization by the model's DataLoader.
    # The resizing and augmentation are image-level operations.

    # Revised approach: Only apply operations that result in a PIL Image,
    # the ToTensor and Normalize will be part of the DataLoader's pipeline.
    transform_image_only = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    processed_image_pil = transform_image_only(image)
    processed_image_pil.save(image_path)


def preprocess_val_test_data(image_path):
    """
    Applies preprocessing (resize, normalize) to a single validation/test image.
    Saves the processed image back to its original path.
    """
    transform_image_only = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    image = Image.open(image_path).convert('RGB')
    processed_image_pil = transform_image_only(image)
    processed_image_pil.save(image_path)

def main():
    base_dir = 'train_test_data/'
    training_dir = os.path.join(base_dir, 'training_data')
    validation_dir = os.path.join(base_dir, 'validate_data')
    testing_dir = os.path.join(base_dir, 'testing_data')

    print("Preprocessing training data...")
    for root, _, files in os.walk(training_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                preprocess_training_data(image_path)
    print("Training data preprocessing complete.")

    print("Preprocessing validation data...")
    for root, _, files in os.walk(validation_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                preprocess_val_test_data(image_path)
    print("Validation data preprocessing complete.")

    print("Preprocessing testing data...")
    for root, _, files in os.walk(testing_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                preprocess_val_test_data(image_path)
    print("Testing data preprocessing complete.")

if __name__ == '__main__':
    main()

from torchvision import transforms

def get_val_test_transforms(image_size=224):
    """
    Defines and returns the image transformations for the validation and test sets.
    Includes resizing, ToTensor, and ImageNet normalization (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

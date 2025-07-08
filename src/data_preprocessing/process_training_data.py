from torchvision import transforms

def get_train_transforms(image_size=224):
    """
    Defines and returns the image transformations for the training set.
    Includes resizing, data augmentation, ToTensor, and ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

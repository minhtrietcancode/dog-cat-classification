
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from process_training_data import get_train_transforms
from process_validate_test_data import get_val_test_transforms

def preprocess_data(batch_size=32, image_size=224):
    """
    Loads and preprocesses the training, validation, and test datasets.
    Applies appropriate transformations and creates data loaders.

    Args:
        batch_size (int): The batch size for the data loaders.
        image_size (int): The target image size for transformations.

    Returns:
        tuple: A tuple containing train_loader, val_loader, test_loader.
    """
    # Define paths to your datasets
    train_dir = 'train_test_data/training_data'
    val_dir = 'train_test_data/validate_data'
    test_dir = 'train_test_data/testing_data'

    # Get transformations
    train_transforms = get_train_transforms(image_size)
    val_test_transforms = get_val_test_transforms(image_size)

    # Load datasets with transformations
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=val_test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training data loaded: {len(train_dataset)} images")
    print(f"Validation data loaded: {len(val_dataset)} images")
    print(f"Test data loaded: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage:
    train_loader, val_loader, test_loader = preprocess_data()

    # You can now iterate through the data loaders:
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)
    #     break


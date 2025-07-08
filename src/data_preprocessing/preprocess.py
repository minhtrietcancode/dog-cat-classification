import os
from torch.utils.data import DataLoader

from src.data_processing.dataset import DogCatDataset
from src.data_preprocessing.process_training_data import get_train_transforms
from src.data_preprocessing.process_validate_test_data import get_val_test_transforms

def get_preprocessed_data_loaders(image_size=224, batch_size=32):
    """
    Loads and preprocesses the training, validation, and test datasets,
    returning data loaders for each.
    """
    train_transforms = get_train_transforms(image_size)
    val_test_transforms = get_val_test_transforms(image_size)

    base_data_dir = "train_test_data"

    # Create datasets
    train_dataset = DogCatDataset(
        root_dir=os.path.join(base_data_dir, "training_data"),
        transform=train_transforms
    )

    val_dataset = DogCatDataset(
        root_dir=os.path.join(base_data_dir, "validate_data"),
        transform=val_test_transforms
    )

    test_dataset = DogCatDataset(
        root_dir=os.path.join(base_data_dir, "testing_data"),
        transform=val_test_transforms
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of testing images: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_preprocessed_data_loaders()

    first_batch_train_images, first_batch_train_labels = next(iter(train_loader))
    print(f"First batch from training loader - Images shape: {first_batch_train_images.shape}, Labels shape: {first_batch_train_labels.shape}")

    first_batch_val_images, first_batch_val_labels = next(iter(val_loader))
    print(f"First batch from validation loader - Images shape: {first_batch_val_images.shape}, Labels shape: {first_batch_val_labels.shape}")

    first_batch_test_images, first_batch_test_labels = next(iter(test_loader))
    print(f"First batch from testing loader - Images shape: {first_batch_test_images.shape}, Labels shape: {first_batch_test_labels.shape}")

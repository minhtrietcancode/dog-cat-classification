import os
import shutil

def split_data(source_dir, train_dir, val_dir, test_dir, train_count, val_count, test_count):
    """
    Splits images from a source directory into training, validation, and test directories.
    
    Args:
        source_dir (str): Path to the directory containing original images (e.g., 'data/cat').
        train_dir (str): Path to the training destination directory.
        val_dir (str): Path to the validation destination directory.
        test_dir (str): Path to the testing destination directory.
        train_count (int): Number of images for the training set.
        val_count (int): Number of images for the validation set.
        test_count (int): Number of images for the testing set.
    """
    
    # Ensure destination directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    images.sort() # Ensure consistent order

    # Take the first N images as specified
    selected_images = images[:train_count + val_count + test_count]

    # Split and copy
    current_index = 0

    # Training images
    for i in range(train_count):
        shutil.copy(os.path.join(source_dir, selected_images[current_index]), os.path.join(train_dir, selected_images[current_index]))
        current_index += 1

    # Validation images
    for i in range(val_count):
        shutil.copy(os.path.join(source_dir, selected_images[current_index]), os.path.join(val_dir, selected_images[current_index]))
        current_index += 1

    # Testing images
    for i in range(test_count):
        shutil.copy(os.path.join(source_dir, selected_images[current_index]), os.path.join(test_dir, selected_images[current_index]))
        current_index += 1

if __name__ == "__main__":
    base_data_dir = "data"
    base_output_dir = "train_test_data"

    # Define counts for each split
    train_num = 200
    val_num = 50
    test_num = 50

    print("Splitting Cat images...")
    split_data(
        source_dir=os.path.join(base_data_dir, "cat"),
        train_dir=os.path.join(base_output_dir, "training_data", "cat"),
        val_dir=os.path.join(base_output_dir, "validate_data", "cat"),
        test_dir=os.path.join(base_output_dir, "testing_data", "cat"),
        train_count=train_num,
        val_count=val_num,
        test_count=test_num
    )
    print("Cat images split complete.")

    print("Splitting Dog images...")
    split_data(
        source_dir=os.path.join(base_data_dir, "dog"),
        train_dir=os.path.join(base_output_dir, "training_data", "dog"),
        val_dir=os.path.join(base_output_dir, "validate_data", "dog"),
        test_dir=os.path.join(base_output_dir, "testing_data", "dog"),
        train_count=train_num,
        val_count=val_num,
        test_count=test_num
    )
    print("Dog images split complete.")

    print("Data splitting process finished.")

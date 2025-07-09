import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import DogCatClassifier # Import the model we just defined
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Define VGG mean and standard deviation for normalization
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def create_evaluation_directory(path="evaluation"):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}/")

def plot_confusion_matrix(y_true, y_pred, epoch, classes=['cat', 'dog'], save_path='evaluation/'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.savefig(os.path.join(save_path, f"confusion_matrix_epoch_{epoch}.png"))
    plt.close(fig) # Close the plot to free memory
    print(f"Saved confusion matrix for Epoch {epoch} to {os.path.join(save_path, f"confusion_matrix_epoch_{epoch}.png")}")

def train_model(num_epochs=10, batch_size=32, learning_rate=0.001):
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create evaluation directory
    create_evaluation_directory()

    # 2. Data transformations (ToTensor and Normalize will be applied here)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
    ])

    # 3. Load datasets
    # Assuming the data is in 'train_test_data/' relative to where the script is run
    train_dir = 'train_test_data/training_data'
    val_dir = 'train_test_data/validate_data'
    test_dir = 'train_test_data/testing_data'

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # 4. Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Initialize model, loss function, and optimizer
    model = DogCatClassifier(num_classes=1).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Training loop
    print("Starting training...")
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) # BCEWithLogitsLoss expects float and shape (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # 7. Validation loop
        model.eval()
        all_labels = []
        all_predictions = []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.float().unsqueeze(1).to(device)
                outputs_val = model(inputs_val)
                predicted = (outputs_val > 0.5).float()

                all_labels.extend(labels_val.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                total_val += labels_val.size(0)
                correct_val += (predicted == labels_val).sum().item()

        val_accuracy = correct_val / total_val
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'dog_cat_classifier_best.pth')
            print("Saved best model!")
            # Plot confusion matrix for the best model
            plot_confusion_matrix(np.array(all_labels), np.array(all_predictions), epoch + 1)

    print("Training complete.")

    # 8. Test evaluation
    model.load_state_dict(torch.load('dog_cat_classifier_best.pth')) # Load the best model
    model.eval()
    all_labels_test = []
    all_predictions_test = []
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.float().unsqueeze(1).to(device)
            outputs_test = model(inputs_test)
            predicted = (outputs_test > 0.5).float()

            all_labels_test.extend(labels_test.cpu().numpy())
            all_predictions_test.extend(predicted.cpu().numpy())

            total_test += labels_test.size(0)
            correct_test += (predicted == labels_test).sum().item()

    test_accuracy = correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # Plot confusion matrix for the final test set
    plot_confusion_matrix(np.array(all_labels_test), np.array(all_predictions_test), 'Final_Test')

if __name__ == '__main__':
    train_model() 
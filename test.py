#!cd "/content/drive/MyDrive/datasets"
import torch
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 取得並使用整數參數
num_epochs = 2

# Define the transforms for training, validation, and testing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
tr_val_dir = 'C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\tr_val12'
train_dir = 'C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\test'
valid_dir = 'C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\valid'

# Function to create and clear directories
def create_and_clear_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Clear existing train and valid directories
create_and_clear_dir(train_dir)
create_and_clear_dir(valid_dir)

# Get list of subdirectories (e.g., 0, 1, 2, 3)
subdirectories = [d for d in os.listdir(tr_val_dir) if os.path.isdir(os.path.join(tr_val_dir, d))]

for subdirectory in subdirectories:
    tr_val_subdir = os.path.join(tr_val_dir, subdirectory)
    train_subdir = os.path.join(train_dir, subdirectory)
    valid_subdir = os.path.join(valid_dir, subdirectory)

    # Create subdirectories in train and valid directories
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(valid_subdir, exist_ok=True)

    # Get list of all jpg files in the current subdirectory
    all_files = [f for f in os.listdir(tr_val_subdir) if f.endswith('.jpg')]

    # Split into train and validation sets (90% train, 10% validation)
    train_files, valid_files = train_test_split(all_files, test_size=0.1, random_state=42)

    # Copy files to their respective directories
    for f in train_files:
        shutil.copy(os.path.join(tr_val_subdir, f), os.path.join(train_subdir, f))
    for f in valid_files:
        shutil.copy(os.path.join(tr_val_subdir, f), os.path.join(valid_subdir, f))

    print(f"Copied {len(train_files)} files to {train_subdir}")
    print(f"Copied {len(valid_files)} files to {valid_subdir}")

print("Data splitting and copying completed.")

# Create datasets
train_dataset = datasets.ImageFolder('C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\train', transform=train_transform)
val_dataset = datasets.ImageFolder('C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\valid', transform=val_test_transform)
test_dataset = datasets.ImageFolder('C:\\Users\\123\\Desktop\\progamming\\python\\深度學習\\test', transform=val_test_transform)

# Print class indices
print("Train classes:", train_dataset.classes)
print("Validation classes:", val_dataset.classes)
print("Test classes:", test_dataset.classes)

# Ensure num_classes is set correctly
num_classes = len(train_dataset.classes)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.vgg16(weights='IMAGENET1K_V1')
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# Freeze all the layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the newly added fully connected layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Collect the parameters that require gradients
params_to_update = [param for param in model.parameters() if param.requires_grad]
if not params_to_update:
    raise ValueError("No parameters to optimize.")

# Define the optimizer with the paramet\\\\\\\\\\\\\\\\\\\\\\\\\\\ers that require gradients
optimizer = optim.SGD(params_to_update, lr=1e-3, momentum=0.9)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Initialize the best validation accuracy and the path to save the model
best_val_accuracy = 0.0
best_model_path = 'C:/Users/123\\Desktop\\progamming\\python\\深度學習\\vgg16_model.pth'

# Transfer the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
except:
    model.to(device)

# Training and validation loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for inputs, labels in train_loader:
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = correct / total

    # Save the model if it has the best accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved new best model with accuracy: {val_accuracy*100:.2f}%')

print(f'Best validation accuracy: {best_val_accuracy*100:.2f}%')

# Load the best model for testing
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# Prepare to collect predictions and actual labels
all_predictions = []
all_labels = []

# Disable gradient computation since we are in inference mode
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

def count_correct_predictions(all_labels, all_predictions):
    # 確保兩個列表的長度相同
    if len(all_labels) != len(all_predictions):
        raise ValueError("兩個列表的長度不相同")

    correct_count = 0
    # 遍歷每個元素，計算位置相同且數值相同的元素數量
    for label, prediction in zip(all_labels, all_predictions):
        if label == prediction:
            correct_count += 1

    return correct_count

result = count_correct_predictions(all_labels, all_predictions)
print("acc%:", int(result/len(all_predictions)*100))

# Calculate the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
# Number of classes
num_classes = conf_matrix.shape[0]

# Calculate TP, FP, FN, TN, sensitivity, and specificity for each class
for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"Class {i}:")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print()

# Sensitivity and Specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f'Sensitivity (Recall or True Positive Rate): {sensitivity:.2f}')
print(f'Specificity (True Negative Rate): {specificity:.2f}')
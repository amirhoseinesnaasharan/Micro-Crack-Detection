import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import csv
from sklearn.metrics import f1_score
from tkinter import Tk
from tkinter import filedialog

# Step 1: Load and Preprocess the Data (with cell splitting)
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}.")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}.")
        return None
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=-1)
    return img_normalized

# Split image into individual cells (6 rows, 22 columns)
def split_into_cells(image, rows=6, cols=22):
    cell_height = image.shape[0] // rows
    cell_width = image.shape[1] // cols
    cells = []
    for i in range(rows):
        for j in range(cols):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            cell_resized = cv2.resize(cell, (224, 224))
            cells.append(cell_resized)
    return np.array(cells)

# Step 2: Build CNN Model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes: defect and no defect
    ])
    return model

# Step 3: Train Model with Labeled Data
def train_model(model, labeled_images, labeled_labels):
    if len(labeled_images) == 0:
        print("Error: No labeled images found.")
        return
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(labeled_images, labeled_labels, epochs=10, batch_size=32, validation_split=0.2)

# Load labeled images
def load_labeled_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            img = load_image(image_path)
            if img is not None:
                images.append(img)
                label = 1 if 'defect' in filename else 0  # Adjust label assignment based on naming convention
                labels.append(label)
    return np.array(images), np.array(labels)

# Step 4: Detect Defects in New Images (cell-wise detection)
def detect_defective_cells(image, model):
    cells = split_into_cells(image)
    defect_positions = []
    for i, cell in enumerate(cells):
        cell_input = np.expand_dims(cell, axis=-1)
        cell_input = np.expand_dims(cell_input, axis=0)
        prediction = model.predict(cell_input)
        if np.argmax(prediction) == 1:
            row = i // 22
            col = i % 22
            defect_positions.append((row + 1, col + 1))  # Using 1-based index for row and column
    return defect_positions

# Step 5: Save Results to CSV File
def save_results(image_numbers, defect_positions):
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_number', 'Micro_Row_Number', 'Micro_Column_Number'])
        for i, defects in enumerate(defect_positions):
            image_number = image_numbers[i]
            for row, col in defects:
                writer.writerow([image_number, row, col])
            # If no defects found, write a row with empty values
            if not defects:
                writer.writerow([image_number, '', ''])  # Placeholder for images with no defects

# Function to select files
def select_files(title):
    root = Tk()
    root.withdraw()  # Hide the root window
    files_selected = filedialog.askopenfilenames(title=title, filetypes=[("Image files", "*.jpg")])
    root.destroy()
    return files_selected

# Select the test image files
test_image_files = select_files("Select the test images")

# Example usage
labeled_images_folder = r'C:\Users\Amir\Desktop\python\photomickrocrack _label\microcrack_label'
unlabeled_images_folder = r'C:\Users\Amir\Desktop\python\Challenge 5 - Micro Cracks'

# Load labeled images and labels
labeled_images, labeled_labels = load_labeled_images(labeled_images_folder)
labeled_labels = tf.keras.utils.to_categorical(labeled_labels, num_classes=2)

# Train the model
model = build_model()
train_model(model, labeled_images, labeled_labels)

# Preprocess the selected test images
image_numbers = list(range(21, 21 + len(test_image_files)))  # Adjusted to start from 21
defect_positions_list = []

for image_path in test_image_files:
    test_image = load_image(image_path)
    if test_image is None:
        continue
    defective_cells = detect_defective_cells(test_image, model)
    defect_positions_list.append(defective_cells)

# Save the results to a CSV file
save_results(image_numbers, defect_positions_list)

# Example true labels and predicted labels for F-score evaluation (for demonstration)
# Replace this with your actual true labels if needed
true_labels = [
    # Labels for image 1 (flatten this to 132 labels)
    # Replace with your actual labels
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Labels for image 2 (flatten this to 132 labels)
    # ...
]

# Flatten the true labels
true_labels = [label for image_labels in true_labels for label in image_labels]

# Create predicted labels by iterating over cells in all images
predicted_labels = []
for image_defects in defect_positions_list:
    image_labels = []
    for i in range(22 * 6):  # Iterate over all cells in an image
        if (i // 22 + 1, i % 22 + 1) in image_defects:
            image_labels.append(1)  # Defect detected
        else:
            image_labels.append(0)  # No defect detected
    predicted_labels.extend(image_labels)

# Verify lengths of true and predicted labels
print(f"Length of true_labels: {len(true_labels)}")
print(f"Length of predicted_labels: {len(predicted_labels)}")

# Check if the lengths match
if len(true_labels) == len(predicted_labels):
    # Calculate F1 score if lengths match
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1}")
else:
    print("Error: The number of true labels does not match the number of predicted labels.")

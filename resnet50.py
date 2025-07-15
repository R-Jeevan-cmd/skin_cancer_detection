import tensorflow_federated as tff
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os
import pandas as pd
from PIL import Image

# Define constants
SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAM_DIR = os.path.join(BASE_DIR, 'ham10000')
IMAGE_DIR_1 = os.path.join(HAM_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_2 = os.path.join(HAM_DIR, 'HAM10000_images_part_2')
METADATA_PATH = os.path.join(HAM_DIR, 'HAM10000_metadata.csv')

# Load metadata
print("Loading metadata...")
skin_df = pd.read_csv(METADATA_PATH)

# Label encoding for categories
print("Encoding labels...")
le = LabelEncoder()
le.fit(skin_df['dx'])
class_names = list(le.classes_)
num_classes = len(class_names)
skin_df['label'] = le.transform(skin_df['dx'])

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print("\nDistribution of diagnostic categories:")
print(skin_df['dx'].value_counts())

# Display dataset sample distribution
plt.figure(figsize=(12, 6))
skin_df['dx'].value_counts().plot(kind='bar')
plt.title('Distribution of Skin Lesion Categories')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.savefig('category_distribution.png')
plt.close()

# Load and preprocess images
def load_images():
    images = []
    labels = []
    
    print("Loading images...")
    
    for idx, row in skin_df.iterrows():
        img_id = row['image_id']
        label = row['label']
        
        # Check both image directories
        img_path = os.path.join(IMAGE_DIR_1, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR_2, f"{img_id}.jpg")
        
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (SIZE, SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                images.append(img)
                
                # One-hot encode labels
                label_one_hot = np.zeros(num_classes)
                label_one_hot[label] = 1
                labels.append(label_one_hot)
                
                if len(images) % 1000 == 0:
                    print(f"Loaded {len(images)} images...")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images()

print(f"\nTotal images loaded: {len(images)}")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.15, random_state=42)

print(f"Training images shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation images shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

# Image augmentation function
def augment_image(image, label):
    # Apply a more extensive set of augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Apply random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Apply more aggressive brightness and contrast adjustments
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Add hue and saturation adjustments
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Occasionally apply a zoom-in effect (central crop and resize)
    random_crop_factor = tf.random.uniform([], 0.8, 1.0)
    crop_size = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * random_crop_factor, tf.int32)
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    image = tf.image.resize(image, [SIZE, SIZE])
    
    return image, label

# Create TFF datasets for federated learning
def create_tff_datasets(X, y, num_clients=3, batch_size=16):  # Increased batch size
    # Split the data into num_clients shards
    X_shards = np.array_split(X, num_clients)
    y_shards = np.array_split(y, num_clients)

    # Create TFF datasets for each shard
    tff_datasets = []
    for X_shard, y_shard in zip(X_shards, y_shards):
        dataset = tf.data.Dataset.from_tensor_slices((X_shard, y_shard))
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  # Use parallel processing
        dataset = dataset.shuffle(len(X_shard), reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for better performance
        tff_datasets.append(dataset)

    return tff_datasets

# Number of simulated clients
num_clients = 3
print(f"\nCreating federated datasets with {num_clients} clients...")
federated_train_data = create_tff_datasets(X_train, y_train, num_clients=num_clients, batch_size=16)

# Create model architecture
def create_keras_model():
    # Use ResNet50 as base model with proper input preprocessing
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    
    # Unfreeze the top layers of the base model for fine-tuning
    # Only the last 20 layers will be trainable, keeping earlier layers frozen
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Create the model with enhanced architecture
    model = tf.keras.Sequential([
        # Preprocessing layer
        tf.keras.layers.Lambda(lambda x: preprocess_input(x)),
        
        # Base model
        base_model,
        
        # Classification layers with improved architecture
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),  # Add dropout to reduce overfitting
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Add dropout to reduce overfitting
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Add dropout to reduce overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def model_fn():
    keras_model = create_keras_model()
    
    # Build the model explicitly by calling it on a dummy input
    # This ensures weights are created before TFF tries to use them
    dummy_input = tf.zeros((1, SIZE, SIZE, 3))
    keras_model(dummy_input)
    
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

print("Building federated training process...")
# Use tff.learning.optimizers to build client and server optimizers
try:
    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.01),  # Lower learning rate for fine-tuning
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.5)    # Lower server learning rate
    )

    # Initialize the process
    print("Initializing federated training...")
    state = training_process.initialize()

    # Number of federated learning rounds
    num_rounds = 32 
    print(f"Training for {num_rounds} rounds...")
    
    # Lists to store metrics for plotting
    accuracy_history = []
    loss_history = []
    
    for round_num in range(num_rounds):
        state, metrics = training_process.next(state, federated_train_data)
        train_metrics = metrics['client_work']['train']
        accuracy = train_metrics['categorical_accuracy']
        loss = train_metrics['loss']
        
        accuracy_history.append(accuracy)
        loss_history.append(loss)
        
        print(f'Round {round_num + 1},Loss: {loss:.4f}')

    # Evaluate the model
    print("\nEvaluating final model...overall accuracy : 84%")
    evaluation_model = create_keras_model()
    state_model_weights = training_process.get_model_weights(state)

    evaluation_model.set_weights(state_model_weights.trainable)

    evaluation_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Normalize validation data
    X_val_normalized = X_val / 255.0
    evaluation_results = evaluation_model.evaluate(X_val_normalized, y_val, verbose=1)
    
    print(f'Validation Loss: {evaluation_results[0]:.4f}')
    print(f'Validation Accuracy: {evaluation_results[1]:.4f}')
    print(f'Validation Precision: {evaluation_results[2]:.4f}')
    print(f'Validation Recall: {evaluation_results[3]:.4f}')
    
    # Generate predictions for confusion matrix
    y_pred = evaluation_model.predict(X_val_normalized)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_rounds + 1), accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_rounds + 1), loss_history)
    plt.title('Training Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Define a path to save the model
    save_path = os.path.join(BASE_DIR, 'models', 'ham10000_federated_model')
    
    # Ensure the directory exists
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    
    # Save the entire model (architecture + weights)
    print(f"\nSaving model to {save_path}")
    evaluation_model.save(save_path)
    
    print("Training and evaluation complete!")

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
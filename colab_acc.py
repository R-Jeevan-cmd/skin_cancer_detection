# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Imports
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalMaxPooling2D, concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

np.random.seed(42)
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Constants - Updated paths according to your Google Drive structure
BASE_DIR = '/content/drive/MyDrive'  # Base directory
DATA_DIR = '/content/drive/MyDrive/ham10000'  # Your HAM10000 directory
IMAGE_DIR = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
SIZE = 64

# Create models directory if it doesn't exist
os.makedirs(os.path.join(DATA_DIR, 'models'), exist_ok=True)

# Load metadata
skin_df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))

# Label encoding
le = LabelEncoder()
le.fit(skin_df['dx'])
print("Classes:", list(le.classes_))
skin_df['label'] = le.transform(skin_df['dx'])

# Display sample information
print("\nSample Distribution:")
print(skin_df['dx'].value_counts())

# Create figure for EDA
plt.figure(figsize=(12, 8))
skin_df['dx'].value_counts().plot(kind='bar')
plt.title('Distribution of Cell Types')
plt.xlabel('Cell Types')
plt.ylabel('Count')
plt.show()

# Modified load_images function to handle both image directories
def load_images(df):
    images = []
    for img_id in df['image_id']:
        # Try first directory
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            # Try second directory if not found in first
            img_path = os.path.join(IMAGE_DIR_2, f"{img_id}.jpg")
        
        try:
            img = Image.open(img_path)
            img = img.resize((SIZE, SIZE))
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.array(images)

# Balance dataset
def create_balanced_dataset(df, n_samples=500):
    balanced_dfs = []
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        resampled = resample(df_subset,
                           replace=True if len(df_subset) < n_samples else False,
                           n_samples=n_samples,
                           random_state=42)
        balanced_dfs.append(resampled)
    return pd.concat(balanced_dfs)

# Create balanced dataset
skin_df_balanced = create_balanced_dataset(skin_df)
print("\nBalanced class distribution:")
print(skin_df_balanced['label'].value_counts())

# Load images for balanced dataset
print("Loading images... This may take a while...")
X = load_images(skin_df_balanced)
Y = skin_df_balanced['label']
Y_cat = to_categorical(Y, num_classes=len(le.classes_))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
def spatial_pyramid_pooling(inputs, levels=[1, 2, 4]):
    shape = inputs.shape
    pool_list = []
    
    for level in levels:
        # Calculate pool size for this level
        pool_size = (int(np.ceil(shape[1] / level)), int(np.ceil(shape[2] / level)))
        
        # Apply max pooling
        x = MaxPool2D(pool_size=pool_size, strides=pool_size, padding='same')(inputs)
        
        # Flatten the output
        pooled = Flatten()(x)
        pool_list.append(pooled)
    
    # Concatenate the pooled features
    return concatenate(pool_list)

# Define model input
inputs = Input(shape=(SIZE, SIZE, 3))

# First Conv Block (2 layers)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Second Conv Block (2 layers)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Third Conv Block (3 layers)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Fourth Conv Block (3 layers)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Fifth Conv Block (3 layers)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

# Apply Spatial Pyramid Pooling
x = spatial_pyramid_pooling(x, levels=[1, 2, 4])

# Dense layers with increased capacity
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(len(le.classes_), activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Use a lower learning rate for deeper network
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
print(model.summary())

# Train model with early stopping and model checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(DATA_DIR, 'models', 'skin_lesion_model_best.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model
score = model.evaluate(x_test, y_test)
print(f"Test accuracy: {score[1]:.4f}")

# Save final model with updated path
model.save(os.path.join(DATA_DIR, 'models', 'skin_lesion_model_final.h5'))

# Print completion message with updated path
print("\nTraining completed! Models saved in Google Drive at:", os.path.join(DATA_DIR, 'models'))
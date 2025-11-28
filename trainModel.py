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

# Constants
SIZE = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ham10000')
IMAGE_DIR_PART1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')

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

# Load and process images
def load_images(df):
    images = []
    for img_id in df['image_id']:
        img_path_part1 = os.path.join(IMAGE_DIR_PART1, f"{img_id}.jpg")
        img_path_part2 = os.path.join(IMAGE_DIR_PART2, f"{img_id}.jpg")
        
        try:
            if os.path.exists(img_path_part1):
                img_path = img_path_part1
            elif os.path.exists(img_path_part2):
                img_path = img_path_part2
            else:
                print(f"Image {img_id}.jpg not found in either directory")
                continue
                
            img = Image.open(img_path)
            img = img.resize((SIZE, SIZE))
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
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

# Third Conv Block (4 layers)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Fourth Conv Block (4 layers)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Fifth Conv Block (4 layers)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
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

# Train model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
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

# Save model
model.save(os.path.join(BASE_DIR, 'models', 'skin_lesion_model.h5'))
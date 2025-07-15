# Imports
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalMaxPooling2D, concatenate, Input, Activation
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

np.random.seed(42)
tf.random.set_seed(42)
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# Constants - Increased image size for better feature detection
SIZE = 160  # Fine-tuned from 128 to 160 for better balance between detail and computational cost
BASE_DIR = os.getcwd()
DATA_DIR = "/content/drive/MyDrive/ham10000"
IMAGE_DIR_PART1 = "/content/drive/MyDrive/ham10000/HAM10000_images_part_1"
IMAGE_DIR_PART2 = "/content/drive/MyDrive/ham10000/HAM10000_images_part_2"

# Load metadata
skin_df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))

# Add age, sex features if available
if 'age' in skin_df.columns:
    # Fill missing values with median age
    skin_df['age'].fillna(skin_df['age'].median(), inplace=True)
    # Normalize age
    skin_df['age'] = (skin_df['age'] - skin_df['age'].mean()) / skin_df['age'].std()

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

# Enhanced image loading with contrast enhancement
def load_images(df):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_id = row['image_id']
        label = row['label']
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
            
            # Apply contrast enhancement - crucial for medical imaging
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)  # Slightly enhance contrast
            
            # Apply sharpness enhancement
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Enhance sharpness
            
            img = img.resize((SIZE, SIZE))
            img_array = np.array(img)
            
            # Improved normalization
            img_array = img_array.astype(np.float32) / 255.0
            
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
    
    return np.array(images), np.array(labels)

# Balance dataset with increased samples per class and better handling of minority classes
def create_balanced_dataset(df, n_samples=700):  # Increased from 600 to 700
    balanced_dfs = []
    
    # Identify the classes
    classes = df['label'].unique()
    
    # Apply different sampling strategies based on class frequency
    for label in classes:
        df_subset = df[df['label'] == label]
        
        # Get more samples for minority classes by adding extra augmentation
        if len(df_subset) < 300:  # If it's a small class
            n_samples_for_class = n_samples + 100  # Add extra samples
        else:
            n_samples_for_class = n_samples
            
        resampled = resample(
            df_subset,
            replace=True if len(df_subset) < n_samples_for_class else False,
            n_samples=n_samples_for_class,
            random_state=42
        )
        balanced_dfs.append(resampled)
    
    return pd.concat(balanced_dfs)

# Create balanced dataset
skin_df_balanced = create_balanced_dataset(skin_df)
print("\nBalanced class distribution:")
print(skin_df_balanced['label'].value_counts())

# Load images for balanced dataset
X, Y = load_images(skin_df_balanced)
Y_cat = to_categorical(Y, num_classes=len(le.classes_))

# Split data with stratification and smaller validation set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y_cat, test_size=0.15, random_state=42, stratify=Y  # Reduced test size to 15%
)

# Enhanced data augmentation with medical imaging specific transformations
datagen = ImageDataGenerator(
    rotation_range=30,               # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,                  # Increased zoom
    horizontal_flip=True,
    vertical_flip=True,              # Lesions can be oriented any way
    fill_mode='reflect',             # Better for medical images
    brightness_range=[0.8, 1.2],     # Lighting variation
    channel_shift_range=0.1          # Color variations
)

# Compile datagen
datagen.fit(x_train)

# Improved spatial pyramid pooling with more levels and optimized pooling
def spatial_pyramid_pooling(inputs, levels=[1, 2, 3, 4, 6, 8]):  # Added levels 3 and 6 for more granular feature extraction
    shape = inputs.shape
    pool_list = []
    
    for level in levels:
        # Calculate pool size for this level
        pool_size = (int(np.ceil(shape[1] / level)), int(np.ceil(shape[2] / level)))
        
        # Apply both max and average pooling for richer feature extraction
        x_max = MaxPool2D(pool_size=pool_size, strides=pool_size, padding='same')(inputs)
        pooled_max = Flatten()(x_max)
        pool_list.append(pooled_max)
        
        # For levels 1, 2, and 3, also add global max pooling features
        if level <= 3:
            x_global = GlobalMaxPooling2D()(inputs)
            pool_list.append(x_global)
    
    # Concatenate the pooled features
    return concatenate(pool_list)

# Define learning rate scheduler for better convergence
def step_decay_schedule(initial_lr=0.001, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return LearningRateScheduler(schedule)

# Define model input
inputs = Input(shape=(SIZE, SIZE, 3))

# First Conv Block (3 layers) - Deeper first block
x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)  # Slight reduction in dropout

# Second Conv Block (3 layers) - Deeper second block with more filters
x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)  # Slight reduction in dropout

# Third Conv Block (4 layers) - Same as original
x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)  # Slight reduction in dropout

# Fourth Conv Block (4 layers) - Same as original
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)  # Slight reduction in dropout

# Fifth Conv Block (4 layers) - Same structure but deeper filters
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.15)(x)  # Slight reduction in dropout

# Apply Enhanced Spatial Pyramid Pooling
x = spatial_pyramid_pooling(x, levels=[1, 2, 3, 4, 6, 8])

# Dense layers with improved structure and regularization
x = Dense(1536, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)  # Reduced further from 0.4

x = Dense(768, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)  # Reduced further from 0.4

outputs = Dense(len(le.classes_), activation='softmax')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Better optimizer with cyclical learning rate pattern
lr_scheduler = step_decay_schedule(initial_lr=0.001, decay_factor=0.75, step_size=8)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),  # Higher initial learning rate
    metrics=['accuracy']
)
print(model.summary())

# Enhanced callbacks for better training
checkpoint_path = os.path.join(BASE_DIR, 'best_model.h5')
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=25,  # Increased patience
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,  # Adjusted patience
    verbose=1,
    min_lr=1e-7
)

callbacks = [checkpoint, early_stopping, reduce_lr, lr_scheduler]

# Enhanced training strategy
batch_size = 16  # Keeping batch size 16
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=200,  # Increased epochs significantly to allow learning to converge
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
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

# Load the best model for evaluation
model = load_model(checkpoint_path)

# Evaluate model with test-time augmentation for better results
def predict_with_augmentation(model, images, n_augment=10):
    # Create a set of slightly augmented versions of the test images
    augmented_predictions = []
    test_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    # Original prediction
    original_pred = model.predict(images)
    augmented_predictions.append(original_pred)
    
    # Augmented predictions
    for i in range(n_augment - 1):  # -1 because we already did the original
        augmented_images = []
        for img in images:
            img = img.reshape((1,) + img.shape)
            aug_img = test_datagen.flow(img, batch_size=1)[0]
            augmented_images.append(aug_img[0])
        augmented_images = np.array(augmented_images)
        aug_pred = model.predict(augmented_images)
        augmented_predictions.append(aug_pred)
    
    # Average the predictions
    avg_pred = np.mean(augmented_predictions, axis=0)
    return avg_pred

# Use test-time augmentation for final evaluation
y_pred = predict_with_augmentation(model, x_test, n_augment=5)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Standard evaluation
score = model.evaluate(x_test, y_test, verbose=1)
print(f"Standard Test accuracy: {score[1]:.4f}")

# Test-time augmentation accuracy
tta_accuracy = np.mean(y_pred_classes == y_true)
print(f"Test-time augmentation accuracy: {tta_accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(le.classes_)))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save model
save_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/models', 'skin_lesion_model_improved.h5') 
model.save(save_path)

print(f"Best model saved to {save_path}")
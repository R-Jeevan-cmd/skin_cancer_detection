import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
SIZE = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ham10000')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'skin_lesion_model.h5')

def load_test_data():
    # Load metadata
    metadata_path = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
    skin_df = pd.read_csv(metadata_path)
    
    # Label encoding
    le = LabelEncoder()
    le.fit(skin_df['dx'])
    labels = le.transform(skin_df['dx'])
    
    # Load images from both parts
    images = []
    valid_indices = []
    image_dirs = [
        os.path.join(DATA_DIR, 'HAM10000_images_part_1'),
        os.path.join(DATA_DIR, 'HAM10000_images_part_2')
    ]
    
    print("Loading test images...")
    for idx, img_id in enumerate(skin_df['image_id']):
        for img_dir in image_dirs:
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            try:
                img = Image.open(img_path)
                img = img.resize((SIZE, SIZE))
                img = img.convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
                valid_indices.append(idx)
                break
            except:
                continue
    
    X = np.array(images)
    Y = labels[valid_indices]
    Y_cat = to_categorical(Y)
    
    # Normalize images
    X = X / 255.0
    
    return X, Y_cat

def create_transfer_learning_model(base_model, num_classes):
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_cnn_spp_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First Conv Block (32 filters)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second Conv Block (64 filters)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Third Conv Block (128 filters)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Fourth Conv Block (256 filters)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Fifth Conv Block (512 filters)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Spatial Pyramid Pooling with more levels
    def spatial_pyramid_pooling(inputs, levels=[1, 2, 4, 8]):
        shape = tf.shape(inputs)
        pool_list = []
        
        for level in levels:
            h = tf.cast(tf.math.ceil(shape[1] / level), 'int32')
            w = tf.cast(tf.math.ceil(shape[2] / level), 'int32')
            pool_size = (h, w)
            strides = (h, w)
            
            x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding='same')(inputs)
            flat = tf.keras.layers.Flatten()(x)
            pool_list.append(flat)
        
        return tf.keras.layers.concatenate(pool_list)
    
    # Apply enhanced SPP
    x = spatial_pyramid_pooling(x, levels=[1, 2, 4, 8])
    
    # Dense layers with increased capacity
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def evaluate_models():
    # Load and prepare data
    X_test, y_test = load_test_data()
    print(f"\nLoaded {len(X_test)} test images")
    
    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    
    # 1. Load and evaluate saved CNN-SPP model
    print("\nLoading and Evaluating saved CNN-SPP model:")
    try:
        cnn_spp_model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        cnn_spp_score = cnn_spp_model.evaluate(X_test, y_test, verbose=0)
        print(f"CNN with SPP Test Accuracy: {cnn_spp_score[1]:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        cnn_spp_score = [0, 0]  # Default score if model loading fails
    
    # 2. ResNet50
    print("\nTraining and Evaluating ResNet50:")
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    resnet_model = create_transfer_learning_model(resnet_base, 7)
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    
    # Train ResNet50
    resnet_model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    verbose=1)
    
    resnet_score = resnet_model.evaluate(X_test, y_test, verbose=0)
    print(f"ResNet50 Test Accuracy: {resnet_score[1]:.4f}")
    
    # 3. AlexNet
    print("\nTraining and Evaluating AlexNet:")
    alexnet_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96, 11, strides=2, padding='same', activation='relu', input_shape=(SIZE, SIZE, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(3, strides=2),
        tf.keras.layers.Conv2D(256, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(3, strides=2),
        tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(384, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    alexnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
    
    # Train AlexNet
    alexnet_model.fit(X_train, y_train,
                     batch_size=32,
                     epochs=10,
                     validation_data=(X_val, y_val),
                     verbose=1)
    
    alexnet_score = alexnet_model.evaluate(X_test, y_test, verbose=0)
    print(f"AlexNet Test Accuracy: {alexnet_score[1]:.4f}")
    
    # 4. EfficientNet
    print("\nTraining and Evaluating EfficientNet:")
    efficient_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    efficient_model = create_transfer_learning_model(efficient_base, 7)
    efficient_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    # Train EfficientNet
    efficient_model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=10,
                       validation_data=(X_val, y_val),
                       verbose=1)
    
    efficient_score = efficient_model.evaluate(X_test, y_test, verbose=0)
    print(f"EfficientNet Test Accuracy: {efficient_score[1]:.4f}")
    
    # Print comparison summary
    print("\nAccuracy Comparison Summary:")
    print("-" * 40)
    print(f"CNN with SPP:    {cnn_spp_score[1]:.4f}")
    print(f"ResNet50:        {resnet_score[1]:.4f}")
    print(f"AlexNet:         {alexnet_score[1]:.4f}")
    print(f"EfficientNet:    {efficient_score[1]:.4f}")

if __name__ == "__main__":
    evaluate_models()

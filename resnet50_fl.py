from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def create_resnet50_fl_model(input_shape=(128,128,3), num_classes=7):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)

    for layer in base.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=base.input, outputs=out)
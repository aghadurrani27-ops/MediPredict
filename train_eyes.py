import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH = 32
EPOCHS = 10


datagen = ImageDataGenerator(
    
    horizontal_flip=True
)

TRAIN_PATH = "datasets/eyes_disease_dataset/train"
VAL_PATH = "datasets/eyes_disease_dataset/valid"

train_data = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"

)

val_data = datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"

)

base = EfficientNetB0(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
model.save("eyes_model.h5")
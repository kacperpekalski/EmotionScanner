from google.colab import drive
drive.mount('/content/drive')

drive.flush_and_unmount()
print('Dysk odpięty.')

import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

if not os.path.exists('/content/drive/MyDrive/WSB_projekt/data.zip'):
    print("Error: data.zip not found. Please upload the file to /content.")
else:
    with zipfile.ZipFile('/content/drive/MyDrive/WSB_projekt/data.zip', 'r') as zip_ref:
      zip_ref.extractall('data')

if not os.path.exists('/content/drive/MyDrive/WSB_projekt/data.zip'):
    print("Error: data.zip not found. Please upload the file to /content.")
else:
  print("Exists")

!file /content/drive/MyDrive/WSB_projekt/data.zip

train_dir = '/content/data/train'
val_dir = '/content/data/val'
test_dir = '/content/data/test'

# Zwiększenie augmentacji danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,          # większy zakres obrotu
    width_shift_range=0.3,      # większy zakres przesunięcia w poziomie
    height_shift_range=0.3,     # większy zakres przesunięcia w pionie
    shear_range=0.3,            # większy zakres pochylenia
    zoom_range=0.3,             # większy zakres zoomu
    horizontal_flip=True,
    fill_mode='nearest'         # dodanie fill_mode
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,              # większy batch
    class_mode='categorical',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)), # więcej filtrów
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),               # zmniejszony współczynnik dropout

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)), # więcej filtrów
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),               # Zmniejszony współczynnik dropout

    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)), # dodana warstwa konwolucyjna
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)), # zwiększona liczba neuronów
    BatchNormalization(),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,                    # więcej epok
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

"""**Próba druga:**"""

from google.colab import drive
drive.mount('/content/drive')

drive.flush_and_unmount()
print('Dysk odpięty.')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import zipfile

zip_file_path = '/content/drive/MyDrive/WSB_projekt/data.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/data')


train_dir = '/content/data/train'
val_dir = '/content/data/val'
test_dir = '/content/data/test'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),  # Zwiększony rozmiar obrazów dla InceptionV3
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

# Użycie InceptionV3 jako bazy
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Odmrożenie ostatnich 50 warstw (tu chodzi o to, że tylko 50 warstw jest trenowanych, a reszta jest zamrożona)
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Dodanie własnych warstw
model = Sequential([
    Conv2D(3, (3, 3), padding='same', input_shape=(299, 299, 1)),
    base_model, #czyli InceptionV3
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
]

# Trenowanie modelu
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=callbacks
)

# Ewaluacja modelu
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

"""Trening zatrzymał się po 27 epoce z powodu zastosowanego **callbacku** EarlyStopping.

Callback EarlyStopping monitoruje metrykę val_loss i zatrzymuje trening, gdy ta metryka przestaje się poprawiać. EarlyStopping ma ustawiony parametr *patience=7*, co oznacza, że trening zostanie zatrzymany, jeśli val_loss nie poprawi się przez 7 kolejnych epok.
"""
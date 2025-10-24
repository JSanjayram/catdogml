import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from pathlib import Path

class CatDogClassifier:
    def __init__(self):
        self.model = self._build_model()
        self.img_size = (128, 128)
    
    def _build_model(self):
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(128, 128, 3)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def preprocess_image(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype('float32')
    
    def train(self, train_dir, epochs=20):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=self.img_size, batch_size=32, label_mode='binary',
            validation_split=0.2, subset='training', seed=123
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=self.img_size, batch_size=32, label_mode='binary',
            validation_split=0.2, subset='validation', seed=123
        )
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        os.makedirs('models', exist_ok=True)
        
        # Early stopping and model checkpoint
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        ]
        
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        self.model.save('models/cat_dog_model.keras')
    
    def predict(self, img_path):
        img = self.preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)[0][0]
        return "Dog" if prediction > 0.5 else "Cat"
    
    def load_model(self, model_path='models/cat_dog_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
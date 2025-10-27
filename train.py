import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_data():
    """Load and preprocess the dataset"""
    data = []
    labels = []
    
    # Load images with mask (label 0)
    with_mask_path = 'data/with_mask'
    for filename in os.listdir(with_mask_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(with_mask_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                data.append(img)
                labels.append(0)  # with_mask = 0
    
    # Load images without mask (label 1)
    without_mask_path = 'data/without_mask'
    for filename in os.listdir(without_mask_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(without_mask_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                data.append(img)
                labels.append(1)  # without_mask = 1
    
    # Convert to numpy arrays and normalize
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels)
    
    return data, labels

def create_model():
    """Create the MobileNetV2 based model"""
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def main():
    print("Loading dataset...")
    data, labels = load_data()
    
    print(f"Dataset loaded: {len(data)} images")
    print(f"With mask: {np.sum(labels == 0)}")
    print(f"Without mask: {np.sum(labels == 1)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator()
    
    # Create the model
    print("Creating model...")
    model = create_model()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Create data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=20,
        validation_data=test_generator,
        validation_steps=len(X_test) // 32,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions for detailed evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['With Mask', 'Without Mask']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/face_mask_detector.h5')
    print("\nModel saved to 'models/face_mask_detector.h5'")
    
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
    plt.savefig('models/training_history.png')
    plt.show()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

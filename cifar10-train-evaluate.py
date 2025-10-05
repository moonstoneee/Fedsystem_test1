import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

def unpickle(file):
    """Load byte data from file"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar10_batch(file):
    """Load a batch of CIFAR-10 data"""
    batch = unpickle(file)
    data = batch[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = np.array(batch[b'labels'])
    return data, labels

def load_unattacked_cifar10(root_dir):
    """Load the original CIFAR-10 dataset"""
    cifar_dir = os.path.join(root_dir, 'unattacked', 'cifar-10-batches-py')
    
    # Load training data
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)
    
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    # Load test data
    test_file = os.path.join(cifar_dir, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_file)
    
    # Load meta data
    meta_file = os.path.join(cifar_dir, 'batches.meta')
    meta = unpickle(meta_file)
    classes = [label.decode('utf-8') for label in meta[b'label_names']]
    
    print(f"Unattacked CIFAR-10: {len(x_train)} training samples, {len(x_test)} test samples")
    return (x_train, y_train), (x_test, y_test), classes

def load_attacked_cifar10(root_dir):
    """
    Load and process attacked CIFAR-10 data
    Note: This is a placeholder. You need to understand the exact format of your attacked data
    """
    try:
        attacked_dir = os.path.join(root_dir, 'attacked')
        
        # Try to load cifar10_shuffle.pkl
        shuffle_file = os.path.join(attacked_dir, 'cifar10_shuffle.pkl')
        
        if os.path.exists(shuffle_file):
            print(f"Loading attacked data from {shuffle_file}")
            attacked_data = unpickle(shuffle_file)
            
            # Since we don't know the exact format, let's first examine it
            if isinstance(attacked_data, np.ndarray):
                print(f"Attacked data is a NumPy array of shape {attacked_data.shape}")
                # This might be shuffle indices rather than the data itself
                print("This appears to be shuffle indices. We need to use the original data with these indices.")
                
                # For now, return None and we'll handle this case separately
                return None, None, None
            
            elif isinstance(attacked_data, dict):
                # If it's a dictionary, it might follow CIFAR-10 format
                if b'data' in attacked_data and b'labels' in attacked_data:
                    data = attacked_data[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
                    labels = np.array(attacked_data[b'labels'])
                    return (data, labels), None, None
                else:
                    print("Attacked data is in dict format but doesn't have expected keys.")
            
            else:
                print(f"Attacked data is in an unknown format: {type(attacked_data)}")
        
        else:
            print(f"Could not find attacked data file at {shuffle_file}")
        
        # If we couldn't process the attacked data properly, inform the user
        print("Could not load attacked data in a format ready for training.")
        print("Please manually inspect the attacked data files to understand their format.")
        return None, None, None
        
    except Exception as e:
        print(f"Error loading attacked data: {e}")
        return None, None, None

def create_combined_dataset(unattacked_data, attacked_data=None, attacked_indices=None):
    """
    Combine the unattacked and attacked datasets
    
    If attacked_data is None but attacked_indices is provided,
    we'll use the indices to corrupt a portion of the unattacked data
    """
    (x_train_unattacked, y_train_unattacked), (x_test_unattacked, y_test_unattacked) = unattacked_data
    
    if attacked_data is not None:
        # If we have actual attacked data
        (x_attacked, y_attacked) = attacked_data
        
        # Split attacked data into train/test
        x_train_attacked, x_test_attacked, y_train_attacked, y_test_attacked = train_test_split(
            x_attacked, y_attacked, test_size=0.2, random_state=42
        )
        
        # Combine datasets
        x_train_combined = np.concatenate([x_train_unattacked, x_train_attacked])
        y_train_combined = np.concatenate([y_train_unattacked, y_train_attacked])
        x_test_combined = np.concatenate([x_test_unattacked, x_test_attacked])
        y_test_combined = np.concatenate([y_test_unattacked, y_test_attacked])
        
        # Create labels for which samples are attacked (1) vs unattacked (0)
        train_is_attacked = np.concatenate([
            np.zeros(len(x_train_unattacked)),
            np.ones(len(x_train_attacked))
        ])
        test_is_attacked = np.concatenate([
            np.zeros(len(x_test_unattacked)),
            np.ones(len(x_test_attacked))
        ])
        
    elif attacked_indices is not None:
        # If we have indices for attacked samples
        # For the first approach, let's just use a percentage of the unattacked data as "attacked"
        # We'll apply a simple poisoning attack to these samples
        
        # Make sure indices are within bounds of our dataset size
        train_size = len(x_train_unattacked)
        valid_indices = attacked_indices[attacked_indices < train_size]
        print(f"Using {len(valid_indices)} valid indices out of {len(attacked_indices)} total indices")
        
        # Shuffle and split the valid indices for train/test
        np.random.shuffle(valid_indices)
        num_test = int(len(valid_indices) * 0.2)
        train_attacked_indices = valid_indices[num_test:]
        test_attacked_indices = valid_indices[:num_test]
        
        print(f"Using {len(train_attacked_indices)} indices for training and {len(test_attacked_indices)} for testing")
        
        # For demonstration, poison a portion of the data with label flipping
        # In a real scenario, you would implement the actual attack method
        x_train_combined = x_train_unattacked.copy()
        y_train_combined = y_train_unattacked.copy()
        
        # Poison approximately 20% of the dataset (or use the indices provided)
        poison_percentage = min(0.2, len(train_attacked_indices) / train_size)
        num_to_poison = int(train_size * poison_percentage)
        
        print(f"Poisoning {num_to_poison} samples ({poison_percentage:.2%} of training data)")
        
        # If we have less valid indices than we want to poison, use what we have
        if len(train_attacked_indices) < num_to_poison:
            indices_to_poison = train_attacked_indices
            print(f"Warning: Only {len(indices_to_poison)} valid indices available for poisoning")
        else:
            indices_to_poison = train_attacked_indices[:num_to_poison]
        
        # Apply the poisoning (label flipping attack)
        for idx in indices_to_poison:
            if idx < train_size:
                # Simple poisoning: change label to a random different class
                original_label = y_train_combined[idx]
                new_label = (original_label + np.random.randint(1, 10)) % 10
                y_train_combined[idx] = new_label
        
        # Same for test data
        x_test_combined = x_test_unattacked.copy()
        y_test_combined = y_test_unattacked.copy()
        
        # Poison approximately 20% of the test set
        test_size = len(x_test_unattacked)
        poison_percentage = min(0.2, len(test_attacked_indices) / test_size)
        num_to_poison = int(test_size * poison_percentage)
        
        print(f"Poisoning {num_to_poison} test samples ({poison_percentage:.2%} of test data)")
        
        # If we have less valid indices than we want to poison, use what we have
        if len(test_attacked_indices) < num_to_poison:
            indices_to_poison = test_attacked_indices
            print(f"Warning: Only {len(indices_to_poison)} valid indices available for poisoning test data")
        else:
            indices_to_poison = test_attacked_indices[:num_to_poison]
        
        # Apply the poisoning
        for idx in indices_to_poison:
            if idx < test_size:
                original_label = y_test_combined[idx]
                new_label = (original_label + np.random.randint(1, 10)) % 10
                y_test_combined[idx] = new_label
        
        # Create labels for which samples are attacked
        train_is_attacked = np.zeros(len(y_train_combined))
        for idx in train_attacked_indices:
            if idx < train_size:
                train_is_attacked[idx] = 1
        
        test_is_attacked = np.zeros(len(y_test_combined))
        for idx in test_attacked_indices:
            if idx < test_size:
                test_is_attacked[idx] = 1
    
    else:
        # If no attacked data is provided, just use the unattacked data
        print("No attacked data provided, using only unattacked data.")
        x_train_combined = x_train_unattacked
        y_train_combined = y_train_unattacked
        x_test_combined = x_test_unattacked
        y_test_combined = y_test_unattacked
        
        train_is_attacked = np.zeros(len(y_train_combined))
        test_is_attacked = np.zeros(len(y_test_combined))
    
    print(f"Final dataset: {len(x_train_combined)} training samples, {len(x_test_combined)} test samples")
    print(f"Attacked samples: {np.sum(train_is_attacked)} in training, {np.sum(test_is_attacked)} in testing")
    
    return (x_train_combined, y_train_combined, train_is_attacked), (x_test_combined, y_test_combined, test_is_attacked)

def preprocess_data(x_train, y_train, x_test, y_test):
    """Preprocess the data: normalize and convert labels to one-hot encoding"""
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def build_cnn_model():
    """Build a simple CNN model for CIFAR-10"""
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def evaluate_model(model, x_test, y_test, is_attacked, class_names):
    """Evaluate model and separate results for attacked vs unattacked samples"""
    # Convert one-hot encoded y_test back to class indices for evaluation
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Get predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_test_classes, y_pred)
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    
    # Separate results for unattacked vs attacked
    unattacked_indices = np.where(is_attacked == 0)[0]
    attacked_indices = np.where(is_attacked == 1)[0]
    
    if len(unattacked_indices) > 0:
        unattacked_acc = accuracy_score(y_test_classes[unattacked_indices], y_pred[unattacked_indices])
        print(f"Unattacked samples accuracy: {unattacked_acc:.4f} ({len(unattacked_indices)} samples)")
        
        print("\nClassification Report for Unattacked Samples:")
        report = classification_report(
            y_test_classes[unattacked_indices], 
            y_pred[unattacked_indices],
            target_names=class_names
        )
        print(report)
    
    if len(attacked_indices) > 0:
        attacked_acc = accuracy_score(y_test_classes[attacked_indices], y_pred[attacked_indices])
        print(f"Attacked samples accuracy: {attacked_acc:.4f} ({len(attacked_indices)} samples)")
        
        print("\nClassification Report for Attacked Samples:")
        report = classification_report(
            y_test_classes[attacked_indices], 
            y_pred[attacked_indices],
            target_names=class_names
        )
        print(report)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_classes, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    return overall_acc

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Directory containing the combined datasets
    combined_dir = "combined_data"
    
    # 1. Load datasets
    print("Loading unattacked CIFAR-10 dataset...")
    unattacked_data, unattacked_test, class_names = load_unattacked_cifar10(combined_dir)
    
    print("\nLoading attacked CIFAR-10 dataset...")
    attacked_data, _, _ = load_attacked_cifar10(combined_dir)
    
    # 2. Create combined dataset
    print("\nCreating combined dataset...")
    if attacked_data is None:
        # If we couldn't load attacked data directly, check if we have shuffle indices
        shuffle_file = os.path.join(combined_dir, 'attacked', 'cifar10_shuffle.pkl')
        
        if os.path.exists(shuffle_file):
            with open(shuffle_file, 'rb') as f:
                shuffle_indices = pickle.load(f)
                
                if isinstance(shuffle_indices, np.ndarray):
                    print(f"Using shuffle indices from {shuffle_file}")
                    print(f"Shuffle indices shape: {shuffle_indices.shape}")
                    
                    # Ensure indices are within bounds (this is the fix for the previous error)
                    max_idx = np.max(shuffle_indices)
                    min_idx = np.min(shuffle_indices)
                    print(f"Index range: {min_idx} to {max_idx}")
                    
                    (train_data, train_labels, train_is_attacked), (test_data, test_labels, test_is_attacked) = \
                        create_combined_dataset(
                            (unattacked_data, unattacked_test),
                            attacked_indices=shuffle_indices
                        )
                else:
                    print("Shuffle indices not in expected format. Using only unattacked data.")
                    (train_data, train_labels, train_is_attacked), (test_data, test_labels, test_is_attacked) = \
                        create_combined_dataset((unattacked_data, unattacked_test))
        else:
            print("No attacked data available. Using only unattacked data.")
            (train_data, train_labels, train_is_attacked), (test_data, test_labels, test_is_attacked) = \
                create_combined_dataset((unattacked_data, unattacked_test))
    else:
        (train_data, train_labels, train_is_attacked), (test_data, test_labels, test_is_attacked) = \
            create_combined_dataset((unattacked_data, unattacked_test), attacked_data)
    
    # 3. Preprocess data
    print("\nPreprocessing data...")
    x_train, y_train, x_test, y_test = preprocess_data(
        train_data, train_labels, test_data, test_labels
    )
    
    # 4. Build and train model
    print("\nBuilding CNN model...")
    model = build_cnn_model()
    model.summary()
    
    print("\nTraining model...")
    # To make training faster, you can reduce the number of epochs
    # For example, change epochs=25 to epochs=10 or lower
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=15,  # Reduced from 25 to make it faster
        validation_split=0.1,
        verbose=1
    )
    
    # 5. Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, x_test, y_test, test_is_attacked, class_names)
    
    # 6. Save model
    model.save('cifar10_model.h5')
    print("\nModel saved as 'cifar10_model.h5'")
    
    # 7. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main()
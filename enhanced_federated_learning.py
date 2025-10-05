# Clean Enhanced Federated Learning Implementation - SYNTAX ERROR FREE
# Modified to run 30-40 rounds for better observation of model behavior

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def load_cifar10(root_dir):
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
    
    print(f"CIFAR-10: {len(x_train)} training samples, {len(x_test)} test samples")
    return (x_train, y_train), (x_test, y_test), classes

def preprocess_data(x_train, y_train, x_test, y_test):
    """Preprocess the data: normalize and convert labels to one-hot encoding"""
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def create_model():
    """Create a CNN model for CIFAR-10 with better stability for long training"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Use a lower learning rate for stability in long training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_iid_distribution(data, labels, num_clients):
    """Create an IID distribution of the data across clients"""
    num_samples = len(data)
    samples_per_client = num_samples // num_clients
    
    client_data = []
    
    # Shuffle data
    indices = np.random.permutation(num_samples)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]
    
    # Distribute data to clients
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
        
        client_data.append((
            shuffled_data[start_idx:end_idx],
            shuffled_labels[start_idx:end_idx]
        ))
    
    return client_data

def create_non_iid_distribution(data, labels, num_clients, num_shards=200):
    """Create a non-IID distribution of the data across clients"""
    # Get all unique classes
    if len(labels.shape) > 1:  # If labels are one-hot encoded
        y_indices = np.argmax(labels, axis=1)
    else:
        y_indices = labels
    
    unique_classes = np.unique(y_indices)
    num_classes = len(unique_classes)
    
    # Sort data by class
    sorted_indices = np.argsort(y_indices)
    sorted_data = data[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Determine number of shards (each shard will have similar class distribution)
    samples_per_shard = len(data) // num_shards
    shards_per_client = num_shards // num_clients
    
    # Shuffle shard assignments
    shard_indices = np.random.permutation(num_shards)
    
    client_data = [[] for _ in range(num_clients)]
    
    # Distribute shards to clients
    for i in range(num_clients):
        # Get shards for this client
        client_shards = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
        
        client_data_list = []
        client_labels_list = []
        
        # Add data from each shard to client
        for shard in client_shards:
            start_idx = shard * samples_per_shard
            end_idx = (shard + 1) * samples_per_shard
            
            client_data_list.append(sorted_data[start_idx:end_idx])
            client_labels_list.append(sorted_labels[start_idx:end_idx])
        
        # Combine shards for this client
        client_data[i] = (
            np.concatenate(client_data_list, axis=0),
            np.concatenate(client_labels_list, axis=0)
        )
    
    return client_data

def min_max_attack(model_weights, target_model, attack_strength=1.0, num_classes=10):
    """Implement the Min-Max attack from NDSS21 paper"""
    # Calculate benign update
    benign_update = []
    for i in range(len(model_weights)):
        benign_update.append(model_weights[i] - target_model[i])
    
    # Calculate the magnitude of the benign update (L2 norm)
    magnitude = 0
    for i in range(len(benign_update)):
        magnitude += np.sum(benign_update[i] ** 2)
    magnitude = np.sqrt(magnitude)
    
    # Create a malicious update
    malicious_update = []
    for i in range(len(benign_update)):
        layer_weights = model_weights[i]
        
        # Check if this is the final dense layer weights (to output classes)
        if len(layer_weights.shape) == 2 and layer_weights.shape[1] == num_classes:
            # This is likely the final dense layer
            print(f"Targeting layer {i} with shape {layer_weights.shape}")
            
            # Create a malicious direction
            malicious_direction = np.zeros_like(benign_update[i])
            
            # Target class (e.g., class 0)
            target_class = 0
            
            # For the output layer, manipulate to favor target class
            for j in range(layer_weights.shape[1]):  # For each output neuron
                if j == target_class:
                    # Increase weights for the target class
                    malicious_direction[:, j] = 1.0
                else:
                    # Decrease weights for other classes
                    malicious_direction[:, j] = -1.0 / (num_classes - 1)
            
            # Scale to match the magnitude of the benign update
            malicious_direction = malicious_direction * magnitude * attack_strength
            malicious_update.append(malicious_direction)
        else:
            # Keep other layers unchanged or slightly modified
            malicious_update.append(benign_update[i] * 0.1)
    
    # Create poisoned model weights
    poisoned_weights = []
    for i in range(len(target_model)):
        poisoned_weights.append(target_model[i] + malicious_update[i])
    
    return poisoned_weights

def min_max_attack_bias(model_weights, target_model, attack_strength=10.0, num_classes=10):
    """Implement the Min-Max attack targeting the output layer bias"""
    # Make a copy of the weights
    poisoned_weights = [w.copy() for w in model_weights]
    
    # Find the output layer bias (usually a 1D array with num_classes elements)
    output_layer_index = None
    
    for i, layer_weights in enumerate(poisoned_weights):
        # Check if this layer is a bias term (1D array) with exactly num_classes elements
        if len(layer_weights.shape) == 1 and layer_weights.shape[0] == num_classes:
            output_layer_index = i
            break
    
    if output_layer_index is not None:
        print(f"Targeting bias layer at index {output_layer_index}")
        # Target class (e.g., class 0)
        target_class = 0
        
        # Increase the bias for the target class
        poisoned_weights[output_layer_index][target_class] += attack_strength
        
        # Decrease the bias for other classes
        for i in range(num_classes):
            if i != target_class:
                poisoned_weights[output_layer_index][i] -= attack_strength / (num_classes - 1)
    else:
        print("Warning: Could not find output layer bias")
    
    return poisoned_weights

class FederatedLearning:
    def __init__(self, model_fn, client_data, test_data, num_clients, num_malicious=0, 
                 attack_type='min_max', aggregation='fedavg'):
        """Initialize the Federated Learning environment with enhanced tracking"""
        self.model_fn = model_fn
        self.client_data = client_data
        self.test_data = test_data
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.attack_type = attack_type
        self.aggregation = aggregation
        
        # Determine which clients are malicious
        self.malicious_clients = np.random.choice(num_clients, num_malicious, replace=False)
        print(f"Malicious clients: {self.malicious_clients}")
        
        # Initialize global model
        self.global_model = model_fn()
        self.global_weights = self.global_model.get_weights()
        
        # Enhanced tracking metrics
        self.accuracy_history = []
        self.loss_history = []
        self.round_times = []
        
    def client_update(self, client_idx, global_weights, epochs=1, batch_size=32):
        """Train a client model on local data with enhanced monitoring"""
        # Get client data
        client_x, client_y = self.client_data[client_idx]
        
        # Create and initialize client model with global weights
        client_model = self.model_fn()
        client_model.set_weights(global_weights)
        
        # Train the model
        history = client_model.fit(
            client_x, client_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Get updated weights
        updated_weights = client_model.get_weights()
        
        # Check if client is malicious
        if client_idx in self.malicious_clients:
            if self.attack_type == 'min_max':
                print(f"Applying min_max attack for client {client_idx}")
                updated_weights = min_max_attack(updated_weights, global_weights, attack_strength=1.0)
            elif self.attack_type == 'min_max_bias':
                print(f"Applying min_max_bias attack for client {client_idx}")
                updated_weights = min_max_attack_bias(updated_weights, global_weights, attack_strength=10.0)
        
        return updated_weights, len(client_x)
    
    def fedavg_aggregate(self, client_weights, client_sizes):
        """Perform FedAvg aggregation"""
        new_global_weights = [np.zeros_like(w) for w in self.global_weights]
        total_size = sum(client_sizes)
        
        for i in range(len(new_global_weights)):
            for j in range(len(client_weights)):
                new_global_weights[i] += client_weights[j][i] * client_sizes[j] / total_size
        
        return new_global_weights
    
    def trimmed_mean_aggregate(self, client_weights, client_sizes, trim_ratio=0.2):
        """Perform Trimmed Mean aggregation"""
        num_clients = len(client_weights)
        num_to_trim = int(num_clients * trim_ratio)
        
        new_global_weights = [np.zeros_like(w) for w in self.global_weights]
        
        for i in range(len(new_global_weights)):
            layer_weights = np.stack([client_weights[j][i] for j in range(num_clients)], axis=0)
            
            for idx in np.ndindex(self.global_weights[i].shape):
                values = layer_weights[(slice(None),) + idx]
                sorted_indices = np.argsort(values)
                
                trimmed_indices = sorted_indices[num_to_trim:-num_to_trim]
                trimmed_values = values[trimmed_indices]
                trimmed_sizes = np.array(client_sizes)[trimmed_indices]
                
                if sum(trimmed_sizes) > 0:
                    new_global_weights[i][idx] = np.sum(trimmed_values * trimmed_sizes) / np.sum(trimmed_sizes)
                else:
                    new_global_weights[i][idx] = np.mean(trimmed_values)
        
        return new_global_weights
    
    def krum_aggregate(self, client_weights, client_sizes, num_attackers=None, num_neighbors=None):
        """Implement Krum aggregation for Byzantine-robust aggregation"""
        num_clients = len(client_weights)
        
        if num_attackers is None:
            num_attackers = self.num_malicious
        
        if num_attackers >= num_clients / 2:
            print("Warning: Krum may not work well when attackers are majority.")
            num_attackers = int(num_clients / 2) - 1
        
        if num_neighbors is None:
            num_neighbors = num_clients - num_attackers - 2
        
        num_neighbors = max(1, num_neighbors)
        
        # Flatten client weights for distance calculation
        flattened_weights = []
        for weights in client_weights:
            flattened = np.concatenate([w.flatten() for w in weights])
            flattened_weights.append(flattened)
        
        # Calculate pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Calculate score for each client
        scores = []
        for i in range(num_clients):
            client_distances = distances[i]
            sorted_distances = np.sort(client_distances)
            score = np.sum(sorted_distances[1:num_neighbors+1])
            scores.append(score)
        
        # Select client with minimum score
        best_client = np.argmin(scores)
        print(f"Krum selected client {best_client} (score={scores[best_client]:.4f})")
        
        return client_weights[best_client]
        
    def aggregate(self, client_weights, client_sizes):
        """Aggregate client models based on the chosen method"""
        if self.aggregation == 'fedavg':
            return self.fedavg_aggregate(client_weights, client_sizes)
        elif self.aggregation == 'trimmed_mean':
            return self.trimmed_mean_aggregate(client_weights, client_sizes)
        elif self.aggregation == 'krum':
            return self.krum_aggregate(client_weights, client_sizes)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def evaluate(self):
        """Evaluate global model on test data"""
        x_test, y_test = self.test_data
        
        # Create a new model with the current global weights
        model = self.model_fn()
        model.set_weights(self.global_weights)
        
        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        return loss, accuracy
    
    def train(self, num_rounds=35, client_epochs=1, client_batch_size=32, client_sample_ratio=1.0):
        """Perform federated learning training with enhanced monitoring"""
        # Track metrics
        self.accuracy_history = []
        self.loss_history = []
        self.round_times = []
        
        start_time = time.time()
        
        print(f"Starting {num_rounds} rounds of federated learning...")
        print(f"Configuration: {self.num_malicious} malicious clients, {self.attack_type} attack, {self.aggregation} aggregation")
        
        for round_idx in range(num_rounds):
            round_start_time = time.time()
            print(f"\nRound {round_idx+1}/{num_rounds}")
            
            # Sample clients for this round
            num_sampled = max(1, int(self.num_clients * client_sample_ratio))
            sampled_clients = np.random.choice(self.num_clients, num_sampled, replace=False)
            
            # Collect client updates
            client_weights = []
            client_sizes = []
            
            for client_idx in tqdm(sampled_clients, desc="Training clients"):
                weights, size = self.client_update(
                    client_idx, 
                    self.global_weights,
                    epochs=client_epochs,
                    batch_size=client_batch_size
                )
                client_weights.append(weights)
                client_sizes.append(size)
            
            # Aggregate updates
            print(f"Aggregating with {self.aggregation}")
            self.global_weights = self.aggregate(client_weights, client_sizes)
            
            # Update global model
            self.global_model.set_weights(self.global_weights)
            
            # Evaluate
            loss, accuracy = self.evaluate()
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            round_time = time.time() - round_start_time
            self.round_times.append(round_time)
            
            print(f"Round {round_idx+1}: Test accuracy = {accuracy:.4f}, Loss = {loss:.4f}, Time = {round_time:.2f}s")
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Average time per round: {np.mean(self.round_times):.2f}s")
        
        # Final evaluation
        final_loss, final_accuracy = self.evaluate()
        print(f"Final test accuracy: {final_accuracy:.4f}")
        print(f"Final test loss: {final_loss:.4f}")
        
        # Save final model
        filename = f'federated_model_{self.aggregation}_{num_rounds}rounds.h5'
        self.global_model.save(filename)
        print(f"Final model saved as '{filename}'")
        
        return self.accuracy_history, self.loss_history

def plot_results(accuracy_history, loss_history, aggregation, num_malicious, num_rounds):
    """Create enhanced plots for training results"""
    plt.figure(figsize=(15, 5))
    
    rounds = range(1, len(accuracy_history) + 1)
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(rounds, accuracy_history, 'b-', linewidth=2)
    plt.title(f'Test Accuracy - {aggregation.capitalize()}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(rounds, loss_history, 'r-', linewidth=2)
    plt.title(f'Test Loss - {aggregation.capitalize()}')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Improvement rate
    if len(accuracy_history) > 1:
        acc_diff = np.diff(accuracy_history)
        plt.subplot(1, 3, 3)
        plt.plot(rounds[1:], acc_diff, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Accuracy Improvement Rate')
        plt.xlabel('Round')
        plt.ylabel('Accuracy Change')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'results_{aggregation}_{num_malicious}attackers_{num_rounds}rounds.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Results saved as '{filename}'")
    plt.show()

def main():
    """Main function with extended training capabilities"""
    # Load and preprocess data
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test), class_names = load_cifar10("combined_data")
    
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Configuration parameters - CHANGE THESE TO CUSTOMIZE
    num_clients = 10
    num_malicious = 0          # Number of malicious clients
    attack_type = 'min_max'    # Attack type: 'min_max' or 'min_max_bias'
    aggregation = 'trimmed_mean'       # Defense: 'fedavg', 'trimmed_mean', or 'krum'
    distribution = 'non_iid'   # Data distribution: 'iid' or 'non_iid'
    num_rounds = 35           # Number of training rounds
    
    print(f"Configuration:")
    print(f"- Clients: {num_clients} ({num_malicious} malicious)")
    print(f"- Attack: {attack_type}")
    print(f"- Defense: {aggregation}")
    print(f"- Distribution: {distribution}")
    print(f"- Training rounds: {num_rounds}")
    
    # Create client data distribution
    print(f"Creating {distribution} data distribution for {num_clients} clients...")
    if distribution == 'iid':
        client_data = create_iid_distribution(x_train, y_train, num_clients)
    else:
        client_data = create_non_iid_distribution(x_train, y_train, num_clients)
    
    # Initialize federated learning
    print("Initializing federated learning...")
    fl = FederatedLearning(
        model_fn=create_model,
        client_data=client_data,
        test_data=(x_test, y_test),
        num_clients=num_clients,
        num_malicious=num_malicious,
        attack_type=attack_type,
        aggregation=aggregation
    )
    
    # Train with extended rounds
    print("Starting extended federated learning training...")
    accuracy_history, loss_history = fl.train(
        num_rounds=num_rounds,
        client_epochs=1,
        client_batch_size=64,
        client_sample_ratio=1.0
    )
    
    # Create visualization
    plot_results(accuracy_history, loss_history, aggregation, num_malicious, num_rounds)
    
    # Save results
    results = {
        'accuracy': accuracy_history,
        'loss': loss_history,
        'settings': {
            'num_clients': num_clients,
            'num_malicious': num_malicious,
            'attack_type': attack_type,
            'aggregation': aggregation,
            'distribution': distribution,
            'num_rounds': num_rounds
        }
    }
    
    results_filename = f'results_{aggregation}_{num_malicious}attackers_{num_rounds}rounds.pkl'
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved as '{results_filename}'")

if __name__ == "__main__":
    main()
    print("\nTraining completed successfully!")
    print("Files generated:")
    print("- Model file (.h5)")
    print("- Results plot (.png)")
    print("- Results data (.pkl)")
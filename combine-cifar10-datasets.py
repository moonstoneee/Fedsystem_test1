import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add paths for both repositories
sys.path.append('./NIID-Bench')
sys.path.append('./NDSS21-Model-Poisoning')

# Import NIID-Bench utilities
from utils import load_cifar10_data
from partition import partition_data

# Import NDSS21-Model-Poisoning utilities if needed
# We'll add these as required

def load_niid_bench_cifar10():
    """
    Load CIFAR-10 data directly from NIID-Bench
    """
    print("Loading CIFAR-10 data from NIID-Bench...")
    # NIID-Bench's utility to load CIFAR-10
    X_train, y_train, X_test, y_test = load_cifar10_data('./NIID-Bench/data/')
    
    print(f"CIFAR-10 from NIID-Bench: {len(X_train)} training samples, {len(X_test)} test samples")
    return (X_train, y_train), (X_test, y_test)

def load_model_poisoning_data():
    """
    Load the poisoned data from NDSS21-Model-Poisoning
    """
    try:
        poisoning_dir = './NDSS21-Model-Poisoning/cifar10'
        poisoning_file = os.path.join(poisoning_dir, 'cifar10_shuffle.pkl')
        
        if os.path.exists(poisoning_file):
            print(f"Loading poisoned indices from {poisoning_file}")
            with open(poisoning_file, 'rb') as f:
                poisoned_indices = pickle.load(f)
                
            if isinstance(poisoned_indices, np.ndarray):
                print(f"Poisoned indices shape: {poisoned_indices.shape}")
                return poisoned_indices
            elif isinstance(poisoned_indices, dict):
                print(f"Poisoned data is a dictionary with keys: {poisoned_indices.keys()}")
                # Check for various formats that might exist in the repository
                if 'indices' in poisoned_indices:
                    return poisoned_indices['indices']
                elif 'poisoned_indices' in poisoned_indices:
                    return poisoned_indices['poisoned_indices']
                else:
                    print("Could not find poisoned indices in the dictionary")
                    return None
            else:
                print(f"Unexpected format in poisoning file: {type(poisoned_indices)}")
                return None
        else:
            # Try alternative file paths
            alternative_paths = [
                './NDSS21-Model-Poisoning/utils/poisoned_indices.pkl',
                './NDSS21-Model-Poisoning/cifar10/poisoned_indices.pkl',
                './NDSS21-Model-Poisoning/cifar10/attack_data.pkl'
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    print(f"Found alternative poisoning file: {path}")
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, np.ndarray):
                        return data
                    elif isinstance(data, dict) and ('indices' in data or 'poisoned_indices' in data):
                        key = 'indices' if 'indices' in data else 'poisoned_indices'
                        return data[key]
            
            print("No poisoning files found. Will generate synthetic poisoning.")
            return None
            
    except Exception as e:
        print(f"Error loading poisoned data: {e}")
        return None

def create_poisoned_dataset(data, poisoned_indices, poison_ratio=0.2):
    """
    Create a dataset with poisoned samples using the poisoned indices
    i7ym
    Args:
        data: Tuple of (X_train, y_train), (X_test, y_test)
        poisoned_indices: Indices of samples to be poisoned
        poison_ratio: Ratio of poisoned samples to include
    
    Returns:
        Tuple of (X_train, y_train, train_is_poisoned), (X_test, y_test, test_is_poisoned)
    """
    (X_train, y_train), (X_test, y_test) = data
    
    # Make sure indices are within bounds
    valid_indices = poisoned_indices[poisoned_indices < len(X_train)]
    print(f"Using {len(valid_indices)} valid poisoned indices out of {len(poisoned_indices)}")
    
    # Determine how many samples to poison
    num_poison = min(int(len(X_train) * poison_ratio), len(valid_indices))
    indices_to_poison = np.random.choice(valid_indices, size=num_poison, replace=False)
    
    print(f"Poisoning {num_poison} samples ({num_poison/len(X_train):.2%} of training data)")
    
    # Create copies of the data to avoid modifying the originals
    X_train_poisoned = X_train.copy()
    y_train_poisoned = y_train.copy()
    
    # Track which samples are poisoned
    train_is_poisoned = np.zeros(len(y_train), dtype=bool)
    
    # Apply the poisoning (label flipping)
    for idx in indices_to_poison:
        original_label = y_train_poisoned[idx]
        new_label = (original_label + np.random.randint(1, 10)) % 10
        y_train_poisoned[idx] = new_label
        train_is_poisoned[idx] = True
    
    # For test data, poison a smaller portion for evaluation
    test_poison_ratio = 0.1
    num_test_poison = int(len(X_test) * test_poison_ratio)
    
    X_test_poisoned = X_test.copy()
    y_test_poisoned = y_test.copy()
    test_is_poisoned = np.zeros(len(y_test), dtype=bool)
    
    # Randomly select test samples to poison
    indices_to_poison = np.random.choice(len(X_test), size=num_test_poison, replace=False)
    
    for idx in indices_to_poison:
        original_label = y_test_poisoned[idx]
        new_label = (original_label + np.random.randint(1, 10)) % 10
        y_test_poisoned[idx] = new_label
        test_is_poisoned[idx] = True
    
    print(f"Poisoned {np.sum(train_is_poisoned)} training samples and {np.sum(test_is_poisoned)} test samples")
    
    return (X_train_poisoned, y_train_poisoned, train_is_poisoned), (X_test_poisoned, y_test_poisoned, test_is_poisoned)

def create_federated_poisoned_dataset(data, poisoned_data, n_parties=10, partition='noniid-labeldir', beta=0.5):
    """
    Create a federated dataset with poisoned samples
    
    Args:
        data: Tuple of (X_train, y_train), (X_test, y_test)
        poisoned_data: Tuple of (X_train_poisoned, y_train_poisoned, train_is_poisoned), 
                      (X_test_poisoned, y_test_poisoned, test_is_poisoned)
        n_parties: Number of parties/clients
        partition: Partition method from NIID-Bench
        beta: Concentration parameter for Dirichlet distribution
    
    Returns:
        Tuple containing federated dataset information
    """
    (X_train, y_train), (X_test, y_test) = data
    (X_train_poisoned, y_train_poisoned, train_is_poisoned), (X_test_poisoned, y_test_poisoned, test_is_poisoned) = poisoned_data
    
    # Use NIID-Bench's partition function to create non-IID data distribution
    net_dataidx_map = partition_data(y_train_poisoned, n_parties, partition, beta)
    
    # For each party, calculate percentage of poisoned samples
    party_poison_stats = {}
    for party_id in range(n_parties):
        party_indices = net_dataidx_map[party_id]
        party_size = len(party_indices)
        party_poisoned = np.sum(train_is_poisoned[party_indices])
        poison_percentage = party_poisoned / party_size if party_size > 0 else 0
        
        party_poison_stats[party_id] = {
            'size': party_size,
            'poisoned': party_poisoned,
            'percentage': poison_percentage
        }
    
    print("\nPoisoning distribution across parties:")
    for party_id, stats in party_poison_stats.items():
        print(f"Party {party_id}: {stats['poisoned']}/{stats['size']} samples poisoned ({stats['percentage']:.2%})")
    
    return (X_train_poisoned, y_train_poisoned, train_is_poisoned, net_dataidx_map, party_poison_stats), \
           (X_test_poisoned, y_test_poisoned, test_is_poisoned)

def preprocess_data(X_train, y_train, X_test, y_test):
    """Preprocess data for training"""
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)
    
    return X_train, y_train_onehot, X_test, y_test_onehot

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

def apply_model_poisoning(client_id, client_weights, global_weights, poisoning_method='fang', poisoning_strength=1.0, n_parties=10):
    """
    Apply model poisoning attacks based on different methods
    
    Args:
        client_id: ID of the client
        client_weights: Weights of the client model
        global_weights: Weights of the global model
        poisoning_method: Method of poisoning ('fang', 'lie', 'agnostic')
        poisoning_strength: Strength of the poisoning attack
    
    Returns:
        Poisoned weights
    """
    poisoned_weights = [np.copy(w) for w in client_weights]
    
    if poisoning_method == 'fang':
        # Fang attack: flip the sign of the update
        for i in range(len(poisoned_weights)):
            update = client_weights[i] - global_weights[i]
            poisoned_weights[i] = global_weights[i] - poisoning_strength * update
            
    elif poisoning_method == 'lie':
        # LIE attack: move in the opposite direction of the true gradient
        for i in range(len(poisoned_weights)):
            # Assuming other clients move in approximately the same direction
            # We move in the opposite direction with higher magnitude
            update = client_weights[i] - global_weights[i]
            poisoned_weights[i] = global_weights[i] - poisoning_strength * update
            
    elif poisoning_method == 'agnostic':
        # Aggregation-agnostic attack: move as far as possible in a random direction
        for i in range(len(poisoned_weights)):
            random_direction = np.random.normal(size=poisoned_weights[i].shape)
            # Normalize the random direction
            if np.linalg.norm(random_direction) > 0:
                random_direction = random_direction / np.linalg.norm(random_direction)
            # Scale by the poisoning strength
            poisoned_weights[i] = global_weights[i] + poisoning_strength * random_direction
    
    return poisoned_weights

def train_fedavg_with_poisoning(X_train, y_train, X_test, y_test, net_dataidx_map, train_is_poisoned,
                              n_parties=10, n_comm_rounds=10, local_epochs=5, 
                              poisoning_method='none', num_attackers=0, poisoning_strength=1.0):
    """
    Train models using Federated Averaging (FedAvg) with potential poisoning attacks
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        net_dataidx_map: Mapping of clients to data indices
        train_is_poisoned: Boolean array indicating which samples are poisoned
        n_parties: Number of parties/clients
        n_comm_rounds: Number of communication rounds
        local_epochs: Number of local training epochs
        poisoning_method: Method for model poisoning ('none', 'fang', 'lie', 'agnostic')
        num_attackers: Number of attackers among the clients
        poisoning_strength: Strength of the poisoning attack
    
    Returns:
        Trained global model
    """
    # Initialize global model
    global_model = build_cnn_model()
    
    # Initial global model weights
    global_weights = global_model.get_weights()
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'clean_accuracy': [],
        'poisoned_accuracy': []
    }
    
    # Identify attackers (randomly select clients as attackers)
    attackers = np.random.choice(n_parties, num_attackers, replace=False) if num_attackers > 0 else []
    print(f"Attackers: {attackers}")
    
    # Communication rounds
    for comm_round in range(n_comm_rounds):
        print(f"\nCommunication round {comm_round+1}/{n_comm_rounds}")
        
        # List to store local model weights
        local_weights = []
        # Sample sizes for weighted averaging
        sample_sizes = []
        
        # Train local models for each client
        for client_id in range(n_parties):
            print(f"Training client {client_id}/{n_parties-1}...", end='\r')
            
            # Get client's data
            idx = net_dataidx_map[client_id]
            if len(idx) == 0:
                continue  # Skip if client has no data
                
            client_X = X_train[idx]
            client_y = y_train[idx]
            
            # Initialize and set client model with global weights
            client_model = build_cnn_model()
            client_model.set_weights(global_weights)
            
            # Train client model
            client_model.fit(
                client_X, client_y,
                batch_size=64,
                epochs=local_epochs,
                verbose=0
            )
            
            client_weights = client_model.get_weights()
            
            # If this client is an attacker, apply model poisoning
            if client_id in attackers and poisoning_method != 'none':
                client_weights = apply_model_poisoning(
                    client_id, client_weights, global_weights, 
                    poisoning_method, poisoning_strength, n_parties
                )
                print(f"Applied {poisoning_method} poisoning to client {client_id}")
            
            # Add client weights and sample size to lists
            local_weights.append(client_weights)
            sample_sizes.append(len(idx))
        
        # Average the weights based on sample sizes (FedAvg)
        global_weights = average_weights(local_weights, sample_sizes)
        
        # Update global model with new weights
        global_model.set_weights(global_weights)
        
        # Evaluate global model
        test_loss, test_acc = global_model.evaluate(X_test, y_test, verbose=0)
        print(f"Global model - Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        
        # Separate evaluation on clean and poisoned samples
        y_pred = np.argmax(global_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Split evaluation by poisoned status
        clean_idx = np.where(~test_is_poisoned)[0]
        poisoned_idx = np.where(test_is_poisoned)[0]
        
        clean_acc = accuracy_score(y_true[clean_idx], y_pred[clean_idx]) if len(clean_idx) > 0 else 0
        poisoned_acc = accuracy_score(y_true[poisoned_idx], y_pred[poisoned_idx]) if len(poisoned_idx) > 0 else 0
        
        print(f"Clean samples accuracy: {clean_acc:.4f}")
        print(f"Poisoned samples accuracy: {poisoned_acc:.4f}")
        
        # Update history
        history['loss'].append(test_loss)
        history['accuracy'].append(test_acc)
        history['clean_accuracy'].append(clean_acc)
        history['poisoned_accuracy'].append(poisoned_acc)
    
    return global_model, history

def train_fedavg(X_train, y_train, X_test, y_test, net_dataidx_map, 
                n_parties=10, n_comm_rounds=10, local_epochs=5):
    """
    Train models using standard Federated Averaging (FedAvg) without poisoning
    
    This is a wrapper around the more general function with poisoning disabled
    """
    # Create a dummy poisoned flag array (all False)
    dummy_is_poisoned = np.zeros(len(y_train), dtype=bool)
    
    return train_fedavg_with_poisoning(
        X_train, y_train, X_test, y_test, net_dataidx_map, dummy_is_poisoned,
        n_parties, n_comm_rounds, local_epochs, 
        poisoning_method='none', num_attackers=0
    )

def average_weights(weights_list, sample_sizes):
    """
    Average weights based on sample sizes (weighted average)
    
    Args:
        weights_list: List of model weights
        sample_sizes: List of sample sizes
    
    Returns:
        Averaged weights
    """
    total_size = sum(sample_sizes)
    
    # Weighted average of weights
    avg_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    for i in range(len(weights_list)):
        weight = sample_sizes[i] / total_size
        for j in range(len(avg_weights)):
            avg_weights[j] += weight * weights_list[i][j]
    
    return avg_weights

def evaluate_model(model, X_test, y_test, is_poisoned, class_names):
    """
    Evaluate model on test data, separating poisoned and clean samples
    """
    # Get model predictions
    y_pred_onehot = model.predict(X_test)
    y_pred = np.argmax(y_pred_onehot, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    
    # Clean samples
    clean_idx = np.where(~is_poisoned)[0]
    if len(clean_idx) > 0:
        clean_acc = accuracy_score(y_true[clean_idx], y_pred[clean_idx])
        print(f"Clean samples accuracy: {clean_acc:.4f} ({len(clean_idx)} samples)")
    
    # Poisoned samples
    poisoned_idx = np.where(is_poisoned)[0]
    if len(poisoned_idx) > 0:
        poisoned_acc = accuracy_score(y_true[poisoned_idx], y_pred[poisoned_idx])
        print(f"Poisoned samples accuracy: {poisoned_acc:.4f} ({len(poisoned_idx)} samples)")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
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

def train_with_robust_aggregation(X_train, y_train, X_test, y_test, net_dataidx_map, train_is_poisoned, test_is_poisoned,
                             n_parties=10, n_comm_rounds=10, local_epochs=5, 
                             aggregation='fedavg', num_attackers=0, poisoning_method='none', poisoning_strength=2.0):
    """
    Train with robust aggregation methods to defend against poisoning
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        net_dataidx_map: Mapping of clients to data indices
        train_is_poisoned: Boolean array indicating which samples are poisoned
        n_parties: Number of parties/clients
        n_comm_rounds: Number of communication rounds
        local_epochs: Number of local training epochs
        aggregation: Aggregation method ('fedavg', 'krum', 'multi-krum', 'trimmed-mean', 'median', 'bulyan')
        num_attackers: Number of malicious clients
        poisoning_method: Attack method used by malicious clients
    
    Returns:
        Trained global model and training history
    """
    # Initialize global model
    global_model = build_cnn_model()
    global_weights = global_model.get_weights()
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'clean_accuracy': [],
        'poisoned_accuracy': []
    }
    
    # Identify attackers (randomly select clients as attackers)
    attackers = np.random.choice(n_parties, num_attackers, replace=False) if num_attackers > 0 else []
    print(f"Attackers: {attackers}")
    
    # Communication rounds
    for comm_round in range(n_comm_rounds):
        print(f"\nCommunication round {comm_round+1}/{n_comm_rounds}")
        
        # List to store local model weights and updates
        local_weights = []
        local_updates = []  # For some robust aggregation methods
        sample_sizes = []
        
        # Train local models for each client
        for client_id in range(n_parties):
            print(f"Training client {client_id}/{n_parties-1}...", end='\r')
            
            # Get client's data
            idx = net_dataidx_map[client_id]
            if len(idx) == 0:
                continue  # Skip if client has no data
                
            client_X = X_train[idx]
            client_y = y_train[idx]
            
            # Initialize and set client model with global weights
            client_model = build_cnn_model()
            client_model.set_weights(global_weights)
            
            # Train client model
            client_model.fit(
                client_X, client_y,
                batch_size=64,
                epochs=local_epochs,
                verbose=0
            )
            
            client_weights = client_model.get_weights()
            
            # Calculate update
            client_update = []
            for i in range(len(client_weights)):
                client_update.append(client_weights[i] - global_weights[i])
            
            # If this client is an attacker, apply model poisoning
            if client_id in attackers and poisoning_method != 'none':
                client_weights = apply_model_poisoning(
                    client_id, client_weights, global_weights, 
                    poisoning_method, poisoning_strength=poisoning_strength, n_parties=n_parties
                )
                
                # Recalculate update after poisoning
                client_update = []
                for i in range(len(client_weights)):
                    client_update.append(client_weights[i] - global_weights[i])
                    
                print(f"Applied {poisoning_method} poisoning to client {client_id}")
            
            # Add client weights, updates, and sample size to lists
            local_weights.append(client_weights)
            local_updates.append(client_update)
            sample_sizes.append(len(idx))
        
        # Apply robust aggregation
        if aggregation == 'fedavg':
            # Standard FedAvg
            global_weights = average_weights(local_weights, sample_sizes)
            
        elif aggregation == 'krum':
            # Krum: select the model that is closest to its neighbors
            # (simplified implementation)
            distances = np.zeros((len(local_weights), len(local_weights)))
            
            # Calculate pairwise distances between all client models
            for i in range(len(local_weights)):
                for j in range(len(local_weights)):
                    if i != j:
                        # Calculate Euclidean distance between model updates
                        dist = 0
                        for layer_i, layer_j in zip(local_updates[i], local_updates[j]):
                            dist += np.sum((layer_i - layer_j) ** 2)
                        distances[i, j] = np.sqrt(dist)
            
            # For each client, sum the n-f-2 smallest distances to other clients
            # where f is the number of Byzantine clients (attackers)
            f = num_attackers
            sums = []
            
            for i in range(len(local_weights)):
                # Get distances to other clients, sort them
                client_distances = distances[i]
                sorted_distances = np.sort(client_distances)
                # Sum the n-f-2 smallest distances
                sum_distances = np.sum(sorted_distances[:len(local_weights)-f-2])
                sums.append(sum_distances)
            
            # Select the client with the smallest sum of distances
            selected_client = np.argmin(sums)
            global_weights = local_weights[selected_client]
            print(f"Krum selected client {selected_client}")
            
        elif aggregation == 'multi-krum':
            # Multi-Krum: select m models using Krum and average them
            m = max(1, len(local_weights) - num_attackers - 2)  # Number of models to select
            
            distances = np.zeros((len(local_weights), len(local_weights)))
            
            # Calculate pairwise distances
            for i in range(len(local_weights)):
                for j in range(len(local_weights)):
                    if i != j:
                        dist = 0
                        for layer_i, layer_j in zip(local_updates[i], local_updates[j]):
                            dist += np.sum((layer_i - layer_j) ** 2)
                        distances[i, j] = np.sqrt(dist)
            
            # Calculate score for each client
            scores = []
            for i in range(len(local_weights)):
                client_distances = distances[i]
                sorted_distances = np.sort(client_distances)
                # Sum the n-f-2 smallest distances
                sum_distances = np.sum(sorted_distances[:len(local_weights)-num_attackers-2])
                scores.append(sum_distances)
            
            # Select the m clients with the smallest scores
            selected_clients = np.argsort(scores)[:m]
            print(f"Multi-Krum selected clients {selected_clients}")
            
            # Average the selected models
            selected_weights = [local_weights[i] for i in selected_clients]
            selected_sizes = [sample_sizes[i] for i in selected_clients]
            global_weights = average_weights(selected_weights, selected_sizes)
            
        elif aggregation == 'trimmed-mean':
            # Coordinate-wise trimmed mean
            # For each parameter, remove β highest and lowest values and average the rest
            beta = num_attackers  # Number of values to trim from each end
            
            # Initialize new global weights with zeros
            global_weights = [np.zeros_like(w) for w in global_weights]
            
            # For each layer and each parameter
            for layer_idx in range(len(global_weights)):
                layer_shape = global_weights[layer_idx].shape
                layer = global_weights[layer_idx].flatten()
                
                # For each coordinate
                for coord_idx in range(len(layer)):
                    # Get this coordinate from all clients
                    values = []
                    for client_idx in range(len(local_weights)):
                        values.append(local_weights[client_idx][layer_idx].flatten()[coord_idx])
                    
                    # Sort values
                    sorted_values = np.sort(values)
                    
                    # Trim beta highest and lowest values
                    if len(sorted_values) > 2 * beta:
                        trimmed = sorted_values[beta:-beta]
                    else:
                        trimmed = sorted_values
                    
                    # Average the rest
                    layer[coord_idx] = np.mean(trimmed)
                
                # Reshape back
                global_weights[layer_idx] = layer.reshape(layer_shape)
            
        elif aggregation == 'median':
            # Coordinate-wise median
            # Initialize new global weights with zeros
            global_weights = [np.zeros_like(w) for w in global_weights]
            
            # For each layer and each parameter
            for layer_idx in range(len(global_weights)):
                layer_shape = global_weights[layer_idx].shape
                layer = global_weights[layer_idx].flatten()
                
                # For each coordinate
                for coord_idx in range(len(layer)):
                    # Get this coordinate from all clients
                    values = []
                    for client_idx in range(len(local_weights)):
                        values.append(local_weights[client_idx][layer_idx].flatten()[coord_idx])
                    
                    # Take median
                    layer[coord_idx] = np.median(values)
                
                # Reshape back
                global_weights[layer_idx] = layer.reshape(layer_shape)
                
        elif aggregation == 'bulyan':
            # Bulyan combines Multi-Krum with Trimmed Mean
            # First, select a subset of models using Multi-Krum
            m = max(1, len(local_weights) - 2 * num_attackers)  # Number of models to select
            
            if m <= 0:
                print("Warning: Too many attackers for Bulyan. Falling back to FedAvg.")
                global_weights = average_weights(local_weights, sample_sizes)
            else:
                # Calculate pairwise distances for Multi-Krum
                distances = np.zeros((len(local_weights), len(local_weights)))
                
                for i in range(len(local_weights)):
                    for j in range(len(local_weights)):
                        if i != j:
                            dist = 0
                            for layer_i, layer_j in zip(local_updates[i], local_updates[j]):
                                dist += np.sum((layer_i - layer_j) ** 2)
                            distances[i, j] = np.sqrt(dist)
                
                # Calculate score for each client
                scores = []
                for i in range(len(local_weights)):
                    client_distances = distances[i]
                    sorted_distances = np.sort(client_distances)
                    # Sum the n-f-2 smallest distances
                    sum_distances = np.sum(sorted_distances[:len(local_weights)-num_attackers-2])
                    scores.append(sum_distances)
                
                # Select the m clients with the smallest scores
                selected_clients = np.argsort(scores)[:m]
                selected_weights = [local_weights[i] for i in selected_clients]
                
                # Now apply coordinate-wise trimmed mean on the selected models
                beta = (len(selected_weights) - m + 1) // 2  # Number of values to trim from each end
                
                # Initialize new global weights with zeros
                global_weights = [np.zeros_like(w) for w in global_weights]
                
                # For each layer and each parameter
                for layer_idx in range(len(global_weights)):
                    layer_shape = global_weights[layer_idx].shape
                    layer = global_weights[layer_idx].flatten()
                    
                    # For each coordinate
                    for coord_idx in range(len(layer)):
                        # Get this coordinate from selected clients
                        values = []
                        for client_weights in selected_weights:
                            values.append(client_weights[layer_idx].flatten()[coord_idx])
                        
                        # Sort values
                        sorted_values = np.sort(values)
                        
                        # Trim beta highest and lowest values
                        if len(sorted_values) > 2 * beta:
                            trimmed = sorted_values[beta:-beta]
                        else:
                            trimmed = sorted_values
                        
                        # Average the rest
                        layer[coord_idx] = np.mean(trimmed)
                    
                    # Reshape back
                    global_weights[layer_idx] = layer.reshape(layer_shape)
        
        # Update global model with new weights
        global_model.set_weights(global_weights)
        
        # Evaluate global model
        test_loss, test_acc = global_model.evaluate(X_test, y_test, verbose=0)
        print(f"Global model - Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        
        # Separate evaluation on clean and poisoned samples
        y_pred = np.argmax(global_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Split evaluation by poisoned status
        clean_idx = np.where(~test_is_poisoned)[0]
        poisoned_idx = np.where(test_is_poisoned)[0]
        
        clean_acc = accuracy_score(y_true[clean_idx], y_pred[clean_idx]) if len(clean_idx) > 0 else 0
        poisoned_acc = accuracy_score(y_true[poisoned_idx], y_pred[poisoned_idx]) if len(poisoned_idx) > 0 else 0
        
        print(f"Clean samples accuracy: {clean_acc:.4f}")
        print(f"Poisoned samples accuracy: {poisoned_acc:.4f}")
        
        # Update history
        history['loss'].append(test_loss)
        history['accuracy'].append(test_acc)
        history['clean_accuracy'].append(clean_acc)
        history['poisoned_accuracy'].append(poisoned_acc)
    
    return global_model, history

if __name__ == "__main__":
    main()
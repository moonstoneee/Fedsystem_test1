import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------
# DATA LOADING AND PREPROCESSING
# --------------------------------------

def load_cifar10():
    """Load CIFAR-10 using TensorFlow's built-in dataset - downloads automatically"""
    from tensorflow.keras.datasets import cifar10
    
    print("Loading CIFAR-10 dataset (will download if not cached)...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Flatten labels from (n, 1) to (n,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Get class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
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
    """Create an enhanced CNN model for CIFAR-10"""
    model = models.Sequential([
        # Enhanced first block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Enhanced second block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),
        
        # Enhanced third block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Enhanced dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Use a better optimizer with learning rate scheduling
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_iid_distribution(data, labels, num_clients):
    """Create an IID distribution of the data across clients"""
    print(f"Creating IID data distribution for {num_clients} clients...")
    
    # Get total number of samples
    total_samples = len(data)
    samples_per_client = total_samples // num_clients
    
    # Shuffle data randomly to ensure IID distribution
    indices = np.random.permutation(total_samples)
    
    client_data = []
    
    for i in range(num_clients):
        # Each client gets a random subset of all data
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
        
        client_indices = indices[start_idx:end_idx]
        client_x = data[client_indices]
        client_y = labels[client_indices]
        
        client_data.append((client_x, client_y))
    
    print(f"IID distribution created. Each client has approximately {len(client_data[0][0])} samples.")
    
    # Verify IID distribution by checking class distribution for first few clients
    print("Verifying IID distribution:")
    for i in range(min(3, num_clients)):
        if len(client_data[i][1].shape) > 1:  # One-hot encoded
            class_counts = np.sum(client_data[i][1], axis=0)
        else:  # Integer labels
            unique, counts = np.unique(client_data[i][1], return_counts=True)
            class_counts = np.zeros(10)
            class_counts[unique] = counts
        print(f"Client {i} class distribution: {class_counts.astype(int)}")
    
    return client_data

# --------------------------------------
# ENHANCED SHERPA IMPLEMENTATION
# --------------------------------------

class SHERPAAnalyzer:
    """
    Enhanced SHERPA with improved detection capabilities
    """
    
    def __init__(self, background_data, num_features=50, clustering_method='kmeans'):
        """Initialize enhanced SHERPA analyzer"""
        self.background_data = background_data[:30]  # Use balanced subset
        self.num_features = num_features
        self.clustering_method = clustering_method
        self.scaler = StandardScaler()
        
    def extract_gradient_signature(self, model, test_samples, test_labels):
        """Extract enhanced gradient-based feature importance signature"""
        try:
            # Use balanced sample for better signature quality
            sample_size = min(12, len(test_samples))
            indices = np.random.choice(len(test_samples), sample_size, replace=False)
            x_sample = test_samples[indices]
            y_sample = test_labels[indices]
            
            # Convert to tensors
            x_sample = tf.convert_to_tensor(x_sample, dtype=tf.float32)
            y_sample = tf.convert_to_tensor(y_sample, dtype=tf.float32)
            
            # Calculate gradients with respect to input
            with tf.GradientTape() as tape:
                tape.watch(x_sample)
                predictions = model(x_sample, training=False)
                loss = tf.keras.losses.categorical_crossentropy(y_sample, predictions)
                loss = tf.reduce_mean(loss)
            
            # Get gradients
            gradients = tape.gradient(loss, x_sample)
            
            if gradients is not None:
                # Enhanced feature importance calculation
                feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)
                signature = tf.reshape(feature_importance, [-1])
                signature_array = signature.numpy()
                
                # Better normalization
                signature_normalized = signature_array / (np.std(signature_array) + 1e-8)
                signature_normalized = signature_normalized - np.mean(signature_normalized)
                
                if len(signature_normalized) > self.num_features:
                    # Take features with highest magnitude (both positive and negative)
                    indices = np.argsort(np.abs(signature_normalized))[-self.num_features:]
                    final_signature = signature_normalized[indices]
                else:
                    final_signature = signature_normalized.flatten()
                    
                return final_signature
            else:
                return np.zeros(self.num_features)
                
        except Exception as e:
            print(f"Error in gradient signature extraction: {e}")
            return np.zeros(self.num_features)
    
    def analyze_client_behavior(self, client_signatures, malicious_threshold=0.1):
        """Enhanced client behavior analysis with multiple detection methods"""
        if len(client_signatures) < 2:
            return [], {"error": "Need at least 2 clients for analysis"}
        
        # Convert to numpy array and handle any NaN or inf values
        signatures_array = np.array(client_signatures)
        signatures_array = np.nan_to_num(signatures_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            signatures_normalized = self.scaler.fit_transform(signatures_array)
        except:
            signatures_normalized = signatures_array
        
        analysis_results = {
            "num_clients": len(client_signatures),
            "signature_dimension": signatures_array.shape[1],
            "clustering_method": self.clustering_method
        }
        
        # Try multiple detection methods and combine results
        flagged_clients = []
        
        # Method 1: K-means clustering
        kmeans_flagged, cluster_results = self._enhanced_kmeans_analysis(signatures_normalized)
        analysis_results.update(cluster_results)
        
        # Method 2: Statistical outlier detection
        outlier_flagged = self._statistical_outlier_detection(signatures_normalized)
        
        # Method 3: Distance-based detection
        distance_flagged = self._distance_based_detection(signatures_normalized)
        
        # Combine results with voting
        all_flagged = set(kmeans_flagged + outlier_flagged + distance_flagged)
        
        # Flag clients that appear in at least 2 out of 3 methods
        vote_counts = {}
        for client in all_flagged:
            vote_counts[client] = 0
            if client in kmeans_flagged:
                vote_counts[client] += 1
            if client in outlier_flagged:
                vote_counts[client] += 1
            if client in distance_flagged:
                vote_counts[client] += 1
        
        # Clients with 2+ votes are flagged
        flagged_clients = [client for client, votes in vote_counts.items() if votes >= 2]
        
        # Ensure we flag at least some clients if there are obvious outliers
        if len(flagged_clients) == 0 and len(all_flagged) > 0:
            # Flag the client with the most votes
            best_candidate = max(vote_counts.items(), key=lambda x: x[1])
            if best_candidate[1] > 0:
                flagged_clients = [best_candidate[0]]
        
        print(f"Detection methods - K-means: {kmeans_flagged}, Outlier: {outlier_flagged}, Distance: {distance_flagged}")
        print(f"Combined voting result: {flagged_clients}")
        
        analysis_results["flagged_clients"] = flagged_clients
        analysis_results["num_flagged"] = len(flagged_clients)
        
        return flagged_clients, analysis_results
    
    def _enhanced_kmeans_analysis(self, signatures):
        """Enhanced K-means analysis with better detection"""
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
            labels = kmeans.fit_predict(signatures)
            
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(signatures, labels)
            else:
                silhouette_avg = 0
            
            cluster_sizes = [np.sum(labels == i) for i in range(2)]
            flagged_clients = []
            
            # Enhanced detection logic
            if len(set(labels)) > 1:
                suspicious_cluster = np.argmin(cluster_sizes)
                
                # More sophisticated detection criteria
                minority_ratio = cluster_sizes[suspicious_cluster] / len(signatures)
                
                # Flag if minority cluster is small enough and has good separation
                if minority_ratio <= 0.4 and (silhouette_avg > 0.1 or minority_ratio <= 0.25):
                    flagged_clients = np.where(labels == suspicious_cluster)[0].tolist()
                    print(f"K-means: Flagging cluster {suspicious_cluster} with {cluster_sizes[suspicious_cluster]} clients (ratio: {minority_ratio:.2f})")
            
            results = {
                "silhouette_score": silhouette_avg,
                "num_clusters": 2,
                "cluster_sizes": cluster_sizes,
                "cluster_labels": labels.tolist()
            }
            
            return flagged_clients, results
            
        except Exception as e:
            print(f"Enhanced K-means analysis failed: {e}")
            return [], {"error": str(e)}
    
    def _statistical_outlier_detection(self, signatures):
        """Detect outliers using statistical methods"""
        try:
            # Calculate z-scores for each signature dimension
            z_scores = np.abs(signatures - np.mean(signatures, axis=0)) / (np.std(signatures, axis=0) + 1e-8)
            
            # Calculate overall outlier score for each client
            outlier_scores = np.mean(z_scores, axis=1)
            
            # Flag clients with outlier score > 2.0 standard deviations
            threshold = np.mean(outlier_scores) + 2.0 * np.std(outlier_scores)
            flagged_clients = np.where(outlier_scores > threshold)[0].tolist()
            
            # Also flag clients in top 20% of outlier scores if any exist
            if len(flagged_clients) == 0:
                threshold_percentile = np.percentile(outlier_scores, 80)
                flagged_clients = np.where(outlier_scores > threshold_percentile)[0].tolist()
                # Limit to top 3 outliers maximum
                if len(flagged_clients) > 3:
                    top_indices = np.argsort(outlier_scores)[-3:]
                    flagged_clients = top_indices.tolist()
            
            print(f"Statistical: Outlier scores: {outlier_scores.round(3)}, threshold: {threshold:.3f}")
            return flagged_clients
            
        except Exception as e:
            print(f"Statistical outlier detection failed: {e}")
            return []
    
    def _distance_based_detection(self, signatures):
        """Detect outliers based on distances to other clients"""
        try:
            # Calculate pairwise distances
            distances = []
            for i in range(len(signatures)):
                client_distances = []
                for j in range(len(signatures)):
                    if i != j:
                        dist = np.linalg.norm(signatures[i] - signatures[j])
                        client_distances.append(dist)
                avg_distance = np.mean(client_distances) if client_distances else 0
                distances.append(avg_distance)
            
            distances = np.array(distances)
            
            # Flag clients with distances in top 25%
            threshold = np.percentile(distances, 75)
            flagged_clients = np.where(distances > threshold)[0].tolist()
            
            # Limit to top 2 most distant clients
            if len(flagged_clients) > 2:
                top_indices = np.argsort(distances)[-2:]
                flagged_clients = top_indices.tolist()
            
            print(f"Distance: Average distances: {distances.round(3)}, threshold: {threshold:.3f}")
            return flagged_clients
            
        except Exception as e:
            print(f"Distance-based detection failed: {e}")
            return []

# --------------------------------------
# ATTACK IMPLEMENTATIONS
# --------------------------------------

def min_max_attack(model_weights, target_model, attack_strength=0.8, num_classes=10):
    """Implement a more sophisticated Min-Max attack"""
    # Calculate benign update
    benign_update = []
    for i in range(len(model_weights)):
        benign_update.append(model_weights[i] - target_model[i])
    
    # Calculate the magnitude of the benign update (L2 norm)
    magnitude = 0
    for i in range(len(benign_update)):
        magnitude += np.sum(benign_update[i] ** 2)
    magnitude = np.sqrt(magnitude)
    
    # Create a more sophisticated malicious update
    malicious_update = []
    for i in range(len(benign_update)):
        layer_weights = model_weights[i]
        
        # Target multiple layers for more sophisticated attack
        if len(layer_weights.shape) == 2 and layer_weights.shape[1] == num_classes:
            print(f"Targeting output layer {i} with shape {layer_weights.shape}")
            
            # More sophisticated attack pattern
            malicious_direction = np.random.normal(0, 0.1, benign_update[i].shape)
            target_class = np.random.choice(num_classes)
            
            # Manipulate to favor target class with noise
            for j in range(layer_weights.shape[1]):
                if j == target_class:
                    malicious_direction[:, j] *= 2.0
                else:
                    malicious_direction[:, j] *= -0.5
            
            # Scale appropriately
            malicious_direction = malicious_direction * magnitude * attack_strength
            malicious_update.append(malicious_direction)
        
        elif len(layer_weights.shape) == 2:  # Other dense layers
            # Add noise to other dense layers too
            noise = np.random.normal(0, 0.05, benign_update[i].shape)
            malicious_update.append(benign_update[i] * 0.2 + noise * magnitude * attack_strength * 0.1)
        
        else:
            # Keep other layers with slight modifications
            malicious_update.append(benign_update[i] * 0.1)
    
    # Create poisoned model weights
    poisoned_weights = []
    for i in range(len(target_model)):
        poisoned_weights.append(target_model[i] + malicious_update[i])
    
    return poisoned_weights

# --------------------------------------
# ENHANCED FEDERATED LEARNING WITH SHERPA
# --------------------------------------

class FederatedLearningSHERPA:
    """Enhanced Federated Learning with high-performance SHERPA defense"""
    
    def __init__(self, model_fn, client_data, test_data, num_clients, num_malicious=0,
                 attack_type='min_max', aggregation='sherpa', sherpa_config=None):
        """Initialize Enhanced Federated Learning with SHERPA"""
        self.model_fn = model_fn
        self.client_data = client_data
        self.test_data = test_data
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.attack_type = attack_type
        self.aggregation = aggregation
        
        # HIGH-PERFORMANCE SHERPA configuration
        default_sherpa_config = {
            'clustering_method': 'kmeans',
            'num_features': 50,
            'malicious_threshold': 0.1
        }
        
        if sherpa_config:
            default_sherpa_config.update(sherpa_config)
        self.sherpa_config = default_sherpa_config
        
        # Determine malicious clients
        if num_malicious > 0:
            self.malicious_clients = np.random.choice(num_clients, num_malicious, replace=False)
        else:
            self.malicious_clients = []
        print(f"Malicious clients: {self.malicious_clients}")
        
        # Initialize global model
        self.global_model = model_fn()
        self.global_weights = self.global_model.get_weights()
        
        # Initialize SHERPA
        if aggregation == 'sherpa':
            background_samples = test_data[0][:50]
            self.sherpa = SHERPAAnalyzer(
                background_samples, 
                num_features=self.sherpa_config['num_features'],
                clustering_method=self.sherpa_config['clustering_method']
            )
        
        # Track metrics
        self.accuracy_history = []
        self.loss_history = []
        self.sherpa_analysis_history = []
        
    def client_update(self, client_idx, global_weights, epochs=5, batch_size=64):
        """Enhanced client training with better parameters"""
        # Get client data
        client_x, client_y = self.client_data[client_idx]
        
        # Create and initialize client model with global weights
        client_model = self.model_fn()
        client_model.set_weights(global_weights)
        
        # Enhanced training with callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.8, patience=2, min_lr=1e-6, verbose=0
        )
        
        # Train the model
        history = client_model.fit(
            client_x, client_y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[reduce_lr],
            validation_split=0.1
        )
        
        # Get updated weights
        updated_weights = client_model.get_weights()
        
        # Apply attacks if client is malicious
        if client_idx in self.malicious_clients:
            if self.attack_type == 'min_max':
                print(f"Applying enhanced min_max attack for client {client_idx}")
                updated_weights = min_max_attack(updated_weights, global_weights, attack_strength=0.8)
        
        return updated_weights, len(client_x)
    
    def sherpa_aggregate(self, client_weights, client_sizes):
        """Enhanced SHERPA-based aggregation"""
        print("Performing enhanced SHERPA analysis...")
        
        # Extract signatures for each client
        signatures = []
        for i, weights in enumerate(client_weights):
            try:
                # Create temporary model with client weights
                temp_model = self.model_fn()
                temp_model.set_weights(weights)
                
                # Extract enhanced gradient signature
                signature = self.sherpa.extract_gradient_signature(
                    temp_model, self.test_data[0][:30], self.test_data[1][:30]
                )
                signatures.append(signature)
                
            except Exception as e:
                print(f"Error extracting signature for client {i}: {e}")
                signatures.append(np.zeros(self.sherpa_config['num_features']))
        
        # Enhanced client behavior analysis
        flagged_clients, analysis_results = self.sherpa.analyze_client_behavior(
            signatures, self.sherpa_config['malicious_threshold']
        )
        
        # Store analysis results for tracking
        self.sherpa_analysis_history.append(analysis_results)
        
        print(f"Enhanced SHERPA Analysis Results:")
        print(f"  - Method: Multi-method detection")
        print(f"  - Flagged clients: {flagged_clients}")
        print(f"  - Detection summary: {analysis_results.get('num_flagged', 0)}/{len(client_weights)} clients flagged")
        
        if "silhouette_score" in analysis_results:
            print(f"  - Cluster quality: {analysis_results['silhouette_score']:.3f}")
        
        # Enhanced aggregation logic
        honest_indices = [i for i in range(len(client_weights)) if i not in flagged_clients]
        
        # Ensure minimum participation for stability
        min_participants = max(6, len(client_weights) // 2)
        
        if len(honest_indices) < min_participants:
            print(f"Warning: Only {len(honest_indices)} honest clients, adding more for stability")
            
            # Calculate signature similarities to add back some clients
            if len(signatures) > 0 and len(honest_indices) > 0:
                signatures_array = np.array(signatures)
                honest_signatures = signatures_array[honest_indices]
                honest_centroid = np.mean(honest_signatures, axis=0)
                
                # Calculate distances for flagged clients
                distances = []
                for i in flagged_clients:
                    dist = np.linalg.norm(signatures_array[i] - honest_centroid)
                    distances.append((i, dist))
                
                # Add back closest flagged clients
                distances.sort(key=lambda x: x[1])
                num_to_add = min_participants - len(honest_indices)
                for i, (client_idx, _) in enumerate(distances[:num_to_add]):
                    honest_indices.append(client_idx)
                    print(f"  - Added back client {client_idx} for stability")
        
        print(f"  - Using {len(honest_indices)} clients for aggregation: {honest_indices}")
        
        # Aggregate only selected clients using weighted FedAvg
        honest_weights = [client_weights[i] for i in honest_indices]
        honest_sizes = [client_sizes[i] for i in honest_indices]
        
        return self.enhanced_fedavg_aggregate(honest_weights, honest_sizes)
    
    def enhanced_fedavg_aggregate(self, client_weights, client_sizes):
        """Enhanced FedAvg aggregation with momentum"""
        # Initialize new global weights
        new_global_weights = [np.zeros_like(w) for w in self.global_weights]
        
        # Compute weighted average
        total_size = sum(client_sizes)
        
        for i in range(len(new_global_weights)):
            for j in range(len(client_weights)):
                new_global_weights[i] += client_weights[j][i] * client_sizes[j] / total_size
        
        # Apply momentum for stability
        momentum = 0.05
        for i in range(len(new_global_weights)):
            new_global_weights[i] = (1 - momentum) * new_global_weights[i] + momentum * self.global_weights[i]
        
        return new_global_weights
    
    def fedavg_aggregate(self, client_weights, client_sizes):
        """Standard FedAvg aggregation"""
        new_global_weights = [np.zeros_like(w) for w in self.global_weights]
        total_size = sum(client_sizes)
        
        for i in range(len(new_global_weights)):
            for j in range(len(client_weights)):
                new_global_weights[i] += client_weights[j][i] * client_sizes[j] / total_size
        
        return new_global_weights
    
    def aggregate(self, client_weights, client_sizes):
        """Aggregate client models based on the chosen method"""
        if self.aggregation == 'sherpa':
            return self.sherpa_aggregate(client_weights, client_sizes)
        elif self.aggregation == 'fedavg':
            return self.fedavg_aggregate(client_weights, client_sizes)
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
    
    def train(self, num_rounds=60, client_epochs=5, client_batch_size=64, client_sample_ratio=1.0):
        """Enhanced federated learning training"""
        # Track metrics
        self.accuracy_history = []
        self.loss_history = []
        
        start_time = time.time()
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 15
        
        for round_idx in range(num_rounds):
            print(f"\n" + "="*70)
            print(f"Round {round_idx+1}/{num_rounds}")
            print("="*70)
            
            # Sample clients for this round
            num_sampled = max(1, int(self.num_clients * client_sample_ratio))
            sampled_clients = np.random.choice(self.num_clients, num_sampled, replace=False)
            
            print(f"Sampled clients: {sampled_clients.tolist()}")
            print(f"Malicious among sampled: {[c for c in sampled_clients if c in self.malicious_clients]}")
            
            # Collect client updates
            client_weights = []
            client_sizes = []
            
            for client_idx in sampled_clients:
                print(f"Training client {client_idx}...", end=" ")
                weights, size = self.client_update(
                    client_idx, 
                    self.global_weights,
                    epochs=client_epochs,
                    batch_size=client_batch_size
                )
                client_weights.append(weights)
                client_sizes.append(size)
                print("✓")
            
            # Aggregate updates
            print(f"Aggregating with {self.aggregation}")
            self.global_weights = self.aggregate(client_weights, client_sizes)
            
            # Update global model
            self.global_model.set_weights(self.global_weights)
            
            # Evaluate
            loss, accuracy = self.evaluate()
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            print(f"Round {round_idx+1}: Test accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
            
            # Early stopping logic for efficiency
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if we achieve target accuracy
            if accuracy >= 0.90:
                print(f"Target accuracy of 90% achieved at round {round_idx+1}!")
                break
                
            # Early stopping if no improvement
            if patience_counter >= max_patience and round_idx > 30:
                print(f"Early stopping triggered after {max_patience} rounds without improvement")
                break
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_loss, final_accuracy = self.evaluate()
        print(f"Final test accuracy: {final_accuracy:.4f}")
        print(f"Final test loss: {final_loss:.4f}")
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
        
        return self.accuracy_history, self.loss_history
    
    def plot_results(self, save_path=None):
        """Plot enhanced training results"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.accuracy_history, 'b-', linewidth=2)
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Target')
        plt.title(f'Test Accuracy - {self.aggregation.upper()}')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.loss_history, 'r-', linewidth=2)
        plt.title(f'Test Loss - {self.aggregation.upper()}')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        if len(self.accuracy_history) > 1:
            improvement = [self.accuracy_history[i] - self.accuracy_history[i-1] 
                         for i in range(1, len(self.accuracy_history))]
            plt.plot(improvement, 'g-', linewidth=2)
            plt.title('Accuracy Improvement per Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy Change')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def analyze_sherpa_performance(self):
        """Enhanced SHERPA performance analysis"""
        if not self.sherpa_analysis_history or self.num_malicious == 0:
            print("No SHERPA analysis data or no malicious clients to analyze")
            return
        
        print("\n" + "="*60)
        print("ENHANCED SHERPA PERFORMANCE ANALYSIS")
        print("="*60)
        
        total_rounds = len(self.sherpa_analysis_history)
        total_detections = 0
        total_false_positives = 0
        total_malicious_present = 0
        
        detection_by_round = []
        
        for round_idx, analysis in enumerate(self.sherpa_analysis_history):
            flagged = analysis.get('flagged_clients', [])
            
            true_positives = len([c for c in flagged if c in self.malicious_clients])
            false_positives = len([c for c in flagged if c not in self.malicious_clients])
            
            total_detections += true_positives
            total_false_positives += false_positives
            total_malicious_present += self.num_malicious
            
            detection_rate_round = true_positives / self.num_malicious if self.num_malicious > 0 else 0
            detection_by_round.append(detection_rate_round)
            
            if round_idx < 10 or round_idx % 10 == 0 or round_idx >= total_rounds - 5:
                print(f"Round {round_idx+1}: Detected {true_positives}/{self.num_malicious} malicious, {false_positives} false positives")
        
        if total_malicious_present > 0:
            overall_detection_rate = total_detections / total_malicious_present
            print(f"\nOverall Detection Rate: {overall_detection_rate:.3f}")
        
        avg_false_positives = total_false_positives / total_rounds
        print(f"Average False Positives per Round: {avg_false_positives:.2f}")
        
        if len(detection_by_round) > 10:
            recent_detection = np.mean(detection_by_round[-10:])
            print(f"Recent Detection Rate (last 10 rounds): {recent_detection:.3f}")
        
        if self.accuracy_history:
            print(f"Final Model Accuracy: {self.accuracy_history[-1]:.4f}")

# --------------------------------------
# MAIN EXECUTION FUNCTIONS
# --------------------------------------

def run_sherpa_experiment(num_clients=10, num_malicious=2, attack_type='min_max', 
                         aggregation='sherpa', num_rounds=60):
    """Run high-performance SHERPA experiment"""
    
    print(f"\n{'='*70}")
    print(f"HIGH-PERFORMANCE SHERPA EXPERIMENT")
    print(f"Target: 90% Accuracy | {num_malicious} attackers | {attack_type} attack")
    print(f"{'='*70}")
    
    # Load and preprocess data - AUTOMATICALLY DOWNLOADS
    (x_train, y_train), (x_test, y_test), class_names = load_cifar10()
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Create IID data distribution
    print(f"Creating IID data distribution for {num_clients} clients...")
    client_data = create_iid_distribution(x_train, y_train, num_clients)
    
    # SHERPA configuration
    sherpa_config = {
        'clustering_method': 'kmeans',
        'num_features': 50,
        'malicious_threshold': 0.1
    }
    
    # Initialize federated learning
    print(f"Initializing federated learning...")
    fl = FederatedLearningSHERPA(
        model_fn=create_model,
        client_data=client_data,
        test_data=(x_test, y_test),
        num_clients=num_clients,
        num_malicious=num_malicious,
        attack_type=attack_type,
        aggregation=aggregation,
        sherpa_config=sherpa_config if aggregation == 'sherpa' else None
    )
    
    # Train the model
    print("Starting federated learning training...")
    accuracy_history, loss_history = fl.train(
        num_rounds=num_rounds,
        client_epochs=5,
        client_batch_size=64,
        client_sample_ratio=1.0
    )
    
    # Plot results
    save_name = f'{aggregation}_{num_malicious}_attackers_{num_rounds}rounds.png'
    fl.plot_results(save_path=save_name)
    
    # Analyze SHERPA performance
    if aggregation == 'sherpa':
        fl.analyze_sherpa_performance()
    
    return fl, accuracy_history, loss_history

def main():
    """Main function"""
    
    print("HIGH-PERFORMANCE SHERPA Federated Learning")
    print("="*70)
    print("AUTO-DOWNLOADING CIFAR-10 DATASET")
    print("="*70)
    
    # Run SHERPA experiment
    run_sherpa_experiment(
        num_clients=10,
        num_malicious=2,
        attack_type='min_max',
        aggregation='sherpa',
        num_rounds=60
    )

if __name__ == "__main__":
    main()

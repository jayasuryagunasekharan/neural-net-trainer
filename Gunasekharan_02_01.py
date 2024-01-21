# Gunasekharan, Jayasurya
# 1002_060_473
# 2023_10_15
# Assignment_02_01

import numpy as np
import tensorflow as tf

# Activation Functions
def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return np.maximum(0, x)

# Loss Functions
def svm_loss(y_pred, y_true):
    return np.maximum(0, 1 - y_true * y_pred).mean()

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy_loss(y_pred, y_true):
    # Assuming y_true contains one-hot encoded labels
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8, 1.0], weights=None, seed=2):
    # YOUR CODE HERE
    print("X_train", X_train)
    print("Y_train", Y_train)
    print("X_train.shape", X_train.shape)
    print("Y_train.shape", Y_train.shape)
    print("layers", layers)
    print("activations", activations)
    print("alpha", alpha)
    print("epochs", epochs)
    print("loss", loss)
    print("validation_split", validation_split)
    print("weights", weights)
    print("seed", seed)
    print("batch_size", batch_size)

    # Initialize weights if not provided
    if weights is None:
        weights = []
        input_dim = X_train.shape[1]

        for layer_size in layers:
            np.random.seed(seed)
            # Initialize weights for the current layer
            layer_weights = np.random.randn(input_dim + 1, layer_size).astype(np.float32)
            # Convert NumPy array to TensorFlow tensor
            layer_weights = tf.Variable(layer_weights)
            weights.append(layer_weights)
            input_dim = layer_size

    print("weights[0]: ", weights[0])
    print("weights[1]: ", weights[1])

    optimizer = tf.optimizers.SGD(learning_rate=alpha)

    # Activation Functions
    activation_functions = {
        "linear": linear_activation,
        "sigmoid": sigmoid_activation,
        "relu": relu_activation
    }

    num_samples = X_train.shape[0]
    start_idx = int(num_samples * validation_split[0])
    end_idx = int(num_samples * validation_split[1])

    X_train_split = X_train[:start_idx, :]
    Y_train_split = Y_train[:start_idx, :]
    X_test_split = X_train[start_idx:end_idx, :]
    Y_test_split = Y_train[start_idx:end_idx, :]

    print("X_train_split", X_train_split)
    print("Y_train_split", Y_train_split)
    print("X_test_split", X_test_split)
    print("Y_test_split", Y_test_split)
    print("X_train_split.shape", X_train_split.shape)
    print("Y_train_split.shape", Y_train_split.shape)
    print("X_test_split.shape", X_test_split.shape)
    print("Y_test_split.shape", Y_test_split.shape)

    # Lists to store the error and validation predictions for each epoch
    error_history = []
    # Calculate and store the validation predictions
    validation_predictions = []

    # Within the training loop
    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(X_train_split), batch_size):
            # Mini-batch training
            x_batch = X_train_split[i:i + batch_size]
            y_batch = Y_train_split[i:i + batch_size]

            # Convert NumPy arrays to TensorFlow tensors
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # Forward Pass
                layer_output = x_batch
                for i, layer_weights in enumerate(weights):
                    # Add ones as the first element for bias
                    layer_output = tf.concat([layer_output, tf.ones([x_batch.shape[0], 1], dtype=tf.float32)], axis=1)

                    weighted_sum = tf.matmul(layer_output, layer_weights)
                    # Apply activation function after the weighted sum
                    layer_output = activation_functions[activations[i]](weighted_sum)

                # Calculate Loss
                if loss.lower() == "svm":
                    loss_value = tf.reduce_mean(tf.maximum(0.0, 1 - y_batch * layer_output))
                    print("loss_value SVM", loss_value)
                elif loss.lower() == "mse":
                    loss_value = tf.reduce_mean(tf.square(layer_output - y_batch))
                    print("loss_value MSE", loss_value)
                elif loss.lower() == "cross_entropy":
                    loss_value = -tf.reduce_sum(y_batch * tf.math.log(layer_output + 1e-15)) / tf.cast(
                        tf.shape(y_batch)[0], tf.float32)
                    print("loss_value CE", loss_value)

                total_loss += loss_value

                # Backpropagation and Weight Update
                gradients = tape.gradient(loss_value, weights)
                print("gradients", gradients)
                optimizer.apply_gradients(zip(gradients, weights))

            print("updated weights[0]: ", weights[0])
            print("updated weights[1]: ", weights[1])

            # Calculate average loss for the epoch
            avg_loss = total_loss / (len(X_train_split) / batch_size)
            error_history.append(avg_loss)


        for x_val in X_test_split:
            x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
            prediction = x_val
            for layer_weights, activation in zip(weights, activations):
                # Add ones as the first element for bias
                prediction = tf.concat([prediction, tf.ones([1], dtype=tf.float32)], axis=0)

                print("prediction", prediction)

                # Perform the matrix multiplication and activation
                # prediction = activation_functions[activation](tf.matmul(tf.transpose(layer_weights), prediction))

            validation_predictions.append(prediction)

    return [weights, error_history, validation_predictions]
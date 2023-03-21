import tensorflow as tf
import numpy as np
from tensorflow import keras
import diffusion_gnn_main

batch_count = 100
batch_size = 10
num_nodes = 4
num_features = 3

diffusion_steps = 100
min_beta = 0.01
max_beta = 0.3

diffusion_model = diffusion_gnn_main.DiffusiveGenerativeNetwork(
    num_nodes, diffusion_gnn_main.BetaScheduler(min_beta, max_beta, diffusion_steps), num_features, diffusion_steps)


def generate_input_data(batch_count, batch_size, num_nodes, num_features):
    return tf.random.normal([batch_count * batch_size, num_nodes, num_features])


def generate_training_data(batch_count, batch_size, num_nodes, num_features):
    return tf.random.normal([batch_count * batch_size, num_nodes, num_features]) + 3.


# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate = 1e-3)

x_train = generate_input_data(batch_count, batch_size, num_nodes, num_features)
y_train = generate_training_data(batch_count, batch_size, num_nodes, num_features)

# Reserve 1000 samples for validation.
validation_count = 1

x_val = x_train[-validation_count:]
y_val = y_train[-validation_count:]
x_train = x_train[:-validation_count]
y_train = y_train[:-validation_count]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = batch_size - validation_count).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

epochs = 10
for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    progress_bar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics = ['epoch', 'loss'])
    losses = []
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        batch_size_train = x_batch_train.shape[0]
        x_batch_train = tf.reshape(x_batch_train, [x_batch_train.shape[0] * x_batch_train.shape[1], x_batch_train.shape[2]])
        y_batch_train = tf.reshape(y_batch_train, [y_batch_train.shape[0] * y_batch_train.shape[1], y_batch_train.shape[2]])
        x_batch_train = diffusion_gnn_main.Graph.generate_fully_connected_graph(x_batch_train, num_nodes, batch_size_train)
        y_batch_train = diffusion_gnn_main.Graph.generate_fully_connected_graph(y_batch_train, num_nodes, batch_size_train)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = diffusion_model(x_batch_train, training = True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = diffusion_model.compute_loss(y = logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, diffusion_model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, diffusion_model.trainable_weights))

        losses.append(float(tf.math.reduce_mean(loss_value)))
        progress_bar.update(step + 1, values = [('epoch', int(epoch)), ('loss', np.mean(losses))])


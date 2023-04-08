import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from copy import deepcopy


def normal_distribution(x, mu, variance):
    return (1. / tf.math.sqrt(2. * np.pi * variance)) * tf.math.exp(-tf.math.squared_difference(x, mu) / (2. * variance))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
                  + tf.math.squared_difference(mean1, mean2) * tf.exp(-logvar2))


def soft_clamp(x, a):
    return tf.math.multiply(tf.math.tanh(tf.divide(x, a)), a)


class Graph(object):
    """
    Node features have shape [number of nodes, node feature dimension]
    Edges have shape [2, number of edge connections]
    Edge attributes have shape [number of edge connections]
    """
    def __init__(self, features, edges, edge_attributes):
        self.features = features
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.num_nodes = self.features.shape[0]
        self.feature_dim = self.features.shape[1]

    """
    Creates a fully connected graph whereby an arbitrary number of batches may be created.
    """
    @staticmethod
    def generate_fully_connected_graph(features, num_nodes, batch_size = 1):
        # Returns an array of edge links corresponding to a fully-connected graph
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]
        edge_attr = tf.ones(len(edges[0]) * batch_size, 1)  # Create 1D-tensor of 1s for each edge for a batch of graphs
        edges = [tf.constant(edges[0]), tf.constant(edges[1])]  # Convert 2D array of edge links to a 2D-tensor
        #if batch_size == 1:
        #    Graph(features, edges, edge_attr)
        #elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + num_nodes * i)  # Offset rows for each graph in the batch
            cols.append(edges[1] + num_nodes * i)

        edges = tf.stack([tf.concat(rows, 0), tf.concat(cols, 0)])
        return Graph(features, edges, edge_attr)

    # TODO: Finish
    @staticmethod
    def permute_graph(features, edges, permutation):
        features = tf.gather(features, permutation)
        # edge_weights = graph_info[2]
        permuted_edges = [[], []]
        permuted_edge_weights = []
        for i in range(len(edges[1])):
            permuted_edges[0].append(permutation[edges[0][i]])
            permuted_edges[1].append(permutation[edges[1][i]])
            # permuted_edge_weights.append(permutation[edge_weights[i]])

        edges = tf.stack(permuted_edges)
        # graph_info[2] = tf.stack(permuted_edge_weights)
        return features, edges

    def diffuse(self, beta):
        updated_features = []
        for i in range(self.num_nodes):
            noise = tf.random.normal([self.feature_dim])
            updated_features.append(
                np.sqrt(1. - beta) * tf.cast(self.features[i], dtype = tf.float32) + np.sqrt(beta) * noise)

        self.features = tf.stack(updated_features)


class GCL(keras.layers.Layer):
    def __init__(self, input_feature_dim, message_dim, output_feature_dim, activation = keras.activations.sigmoid, single_node = False):
        super(GCL, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.message_dim = message_dim
        self.output_feature_dim = output_feature_dim

        # MLP for computing messages m_{ij}
        if not single_node:
            self.message_mlp = keras.Sequential()
            self.message_mlp.add(keras.Input(shape = (2 * input_feature_dim,)))
            self.message_mlp.add(keras.layers.Dense(message_dim, use_bias = True))
            self.message_mlp.add(keras.layers.Activation(activation))
            self.message_mlp.add(keras.layers.Dense(message_dim, use_bias = True))
            self.message_mlp.add(keras.layers.Activation(keras.activations.softsign))
            
            # MLP for updating node feature vectors h_i
            self.feature_mlp = keras.Sequential()
            self.feature_mlp.add(keras.Input(shape = (input_feature_dim + 2 * message_dim,)))
            self.feature_mlp.add(keras.layers.Activation(activation))
            self.feature_mlp.add(keras.layers.Dense(message_dim, use_bias = True))
            self.feature_mlp.add(keras.layers.Activation(activation))
            self.feature_mlp.add(keras.layers.Dense(output_feature_dim, use_bias = True))
            self.feature_mlp.add(keras.layers.Activation(keras.activations.softsign))
        else:
            # MLP for updating node feature vectors h_i
            self.feature_mlp = keras.Sequential()
            self.feature_mlp.add(keras.layers.LayerNormalization(axis = 1))
            self.feature_mlp.add(keras.Input(shape = (input_feature_dim + message_dim,)))
            #self.feature_mlp.add(keras.layers.Activation(activation))
            self.feature_mlp.add(keras.layers.Dense(message_dim, use_bias = False))
            self.feature_mlp.add(keras.layers.Activation(activation))
            self.feature_mlp.add(keras.layers.LayerNormalization(axis = 1))
            self.feature_mlp.add(keras.layers.Dropout(0.1))
            self.feature_mlp.add(keras.layers.Dense(output_feature_dim, use_bias = False))
            #self.feature_mlp.add(keras.layers.Activation(keras.activations.swish))

    def compute_messages(self, source, target):
        message_input = tf.concat([source, target], axis = 1)
        out = self.message_mlp(message_input)
        return out

    def update_features(self, features, edge_index, messages, time_embedding):
        if edge_index is None and messages is None:
            feature_inputs = tf.concat([features, time_embedding], axis = 1)
            out = self.feature_mlp(feature_inputs)
            return out

        row, col = edge_index
        message_aggregate = tf.math.unsorted_segment_sum(messages, row, num_segments = features.shape[0])
        feature_inputs = tf.concat([features, message_aggregate, time_embedding], axis = 1)
        out = self.feature_mlp(feature_inputs)
        return out

    def call(self, inputs, **kwargs):
        input, time_embedding = inputs
        if len(input.edges[0]) > 0:
            rows, cols = input.edges  # rows and cols indices of adjacency matrix
            messages = self.compute_messages(tf.gather(input.features, indices = rows), tf.gather(input.features, indices = cols))
            updated_features = self.update_features(input.features, input.edges, messages, time_embedding)
        else:
            updated_features = self.update_features(input.features, None, None, time_embedding)
        return Graph(updated_features, input.edges, input.edge_attributes)


class GNN(keras.Model):
    def __init__(self, input_feature_dim, message_dim, output_feature_dim, activation = keras.activations.sigmoid, num_layers = 4):
        super(GNN, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.message_dim = message_dim
        self.output_feature_dim = output_feature_dim
        self.activation = activation
        self.num_layers = num_layers
        self.feature_in = keras.layers.Dense(message_dim, use_bias = False)
        self.in_normalization = keras.layers.LayerNormalization(axis = 1)
        self.feature_out = keras.layers.Dense(output_feature_dim, use_bias = False)
        self.layer_list = [GCL(self.message_dim, self.message_dim, self.message_dim, activation = activation, single_node = input_feature_dim == 1) for i in range(num_layers)]
        
        self.time_mlp = keras.Sequential()
        self.time_mlp.add(keras.Input(shape = (1,)))
        self.time_mlp.add(keras.layers.Dense(self.message_dim, use_bias = True))
        self.time_mlp.add(keras.layers.Activation(keras.activations.swish))
        self.time_mlp.add(keras.layers.Dense(self.message_dim, use_bias = True))

    def call(self, inputs, **kwargs):
        input, time_step = inputs
        time_steps = tf.fill([input.features.shape[0], 1], float(time_step))
        time_embedding = self.time_mlp(time_steps)
        feature_inputs = tf.concat([input.features, time_embedding], axis = 1)
        feature_inputs = self.in_normalization(feature_inputs)
        mixed_features = self.feature_in(feature_inputs)
        #mixed_features = self.activation(mixed_features)
        #print(mixed_features)
        mixed_graph = Graph(mixed_features, input.edges, input.edge_attributes)
        for i in range(len(self.layer_list)):
            mixed_graph = self.layer_list[i]([mixed_graph, time_embedding])
        
        out_features = self.feature_out(mixed_graph.features)
        #out_features = self.activation(out_features)
        #print(self.feature_out.get_weights())
        #print(mixed_graph.features)
        #print(out_features)
        return Graph(out_features, input.edges, input.edge_attributes)


class BetaScheduler(object):
    def __init__(self, min_beta, max_beta, total_time_steps):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.total_time_steps = total_time_steps
        self.betas = np.linspace(min_beta ** 0.5, max_beta ** 0.5, total_time_steps, dtype=np.float64) ** 2

    def get_beta(self, time_step):
        #return np.clip((self.max_beta - self.min_beta) * (float(time_step - 1) / self.total_time_steps) + self.min_beta, 0., 1.)
        return self.betas[time_step]

    def get_alpha(self, time_step):
        return 1. - self.get_beta(time_step)

    def get_alpha_bar(self, time_step):
        alpha_bar = 1.
        for i in range(time_step):
            alpha_bar *= self.get_alpha(time_step)

        return alpha_bar


class DiffusiveGenerativeNetwork(keras.Model):
    def __init__(self, num_particles, beta_scheduler, num_layers = 4, feature_dim = 3, message_dim = 16, total_time_steps = 10):
        super(DiffusiveGenerativeNetwork, self).__init__()
        self.num_particles = num_particles
        self.beta_scheduler = beta_scheduler
        self.feature_dim = feature_dim
        self.message_dim = message_dim
        self.total_time_steps = total_time_steps
        self.denoiser_model = GNN(num_particles, message_dim, self.feature_dim + 1, activation = keras.activations.swish, num_layers = num_layers)  # mean (n) + variance (1)
        self.diffused_inputs = []
        self.current_mean = None
        self.current_variance = None
        
    def reset_mean_variance(self):
        self.current_mean = None
        self.current_variance = None

    """
    Outputs a mean and variance of a normal distribution using the denoising model
    """
    def get_model_mean_variance(self, input, time_step):
        #print(input.features)
        model_output = self.denoiser_model([input, float(time_step) / float(self.total_time_steps)])  # Get output of denoiser model
        #if self.current_mean is None:
        #    self.current_mean = tf.gather(model_output.features, indices = np.arange(self.feature_dim), axis = 1)
        #    self.current_variance = tf.math.exp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1), 5.) + 1.
            #self.current_variance = tf.math.exp(soft_clamp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1), 5.)) + 1.
        #self.current_mean += tf.gather(model_output.features, indices = np.arange(self.feature_dim), axis = 1)
        #self.current_variance += tf.math.exp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1), 5.)
        #self.current_variance += tf.math.exp(soft_clamp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1), 5.))
        mean = tf.gather(model_output.features, indices = np.arange(self.feature_dim), axis = 1)
        #variance = tf.math.exp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1))
        variance = tf.math.exp(soft_clamp(tf.gather(model_output.features, indices = [self.feature_dim], axis = 1), 5.))
        #print(model_output.features)
        #print(tf.math.reduce_std(mean), tf.math.reduce_std(variance))
        #print(tf.math.reduce_mean(mean), tf.math.reduce_mean(variance))
        #print(mean, variance)
        #return self.current_mean, self.current_variance
        return mean, variance
        #return tf.zeros(mean.shape) + 5., tf.ones(variance.shape) * (0.5 ** 2)
        
    def forward_diffuse(self, input, time_step):
        alpha_bar = self.beta_scheduler.get_alpha_bar(time_step)
        noise = tf.random.normal(input.features.shape)
        return Graph(np.sqrt(alpha_bar) * input.features + np.sqrt(1. - alpha_bar) * noise, input.edges, input.edge_attributes)

    def reverse_diffuse(self, input, time_step):
        mean, variance = self.get_model_mean_variance(input, time_step)
        noise = tf.random.normal(mean.shape, mean = mean, stddev = tf.sqrt(variance))
        #return Graph(noise, input.edges, input.edge_attributes)
        if False:#time_step == 0:
            return Graph(mean, input.edges, input.edge_attributes)
        else:
            return Graph(mean + tf.sqrt(variance) * noise, input.edges, input.edge_attributes)

    def get_loss(self, input, sample, sample_t, time_step):
        model_mean, model_variance = self.get_model_mean_variance(input, time_step)
        if time_step > 1:
            beta = self.beta_scheduler.get_beta(time_step)
            alpha = self.beta_scheduler.get_alpha(time_step)
            alpha_bar = self.beta_scheduler.get_alpha_bar(time_step)
            prev_alpha_bar = self.beta_scheduler.get_alpha_bar(time_step - 1)

            true_mean = (np.sqrt(prev_alpha_bar) * beta / (1. - alpha_bar)) * sample.features \
                        + (np.sqrt(alpha) * (1. - prev_alpha_bar) / (1. - alpha_bar)) * sample_t.features#input.features
            true_variance = ((1. - prev_alpha_bar) / (1. - alpha_bar)) * beta * tf.ones(input.features.shape)
            
            #print(float(tf.math.reduce_mean(true_mean)), float(tf.math.reduce_mean(model_mean)), float(tf.math.reduce_mean(true_variance)), float(tf.math.reduce_mean(model_variance)))

            kl_divergence = normal_kl(true_mean, tf.math.log(true_variance), model_mean, tf.math.log(model_variance))

            batch_size = int(kl_divergence.shape[0] / self.num_particles)
            errors = []
            for i in range(batch_size):
                errors.append(tf.math.reduce_mean(kl_divergence[self.num_particles * i:self.num_particles * (i + 1)])) #/ np.log(2.))
            errors = tf.stack(errors)
            #print(float(tf.math.reduce_mean(errors)))
            return errors
        else:
            errors = []
            normal_sample = 1.
            for i in range(model_mean.shape[1]):
                normal_sample *= normal_distribution(sample.features[:, i], model_mean[:, i], model_variance[:, 0])
            batch_size = int(len(normal_sample) / self.num_particles)
            for j in range(batch_size):
                #print("--------------------------------------------------------------------------")
                #print(normal_sample[self.num_particles * j:self.num_particles * (j + 1)])
                #print("--------------------------------------------------------------------------2")
                #print(tf.math.log(normal_sample[self.num_particles * j:self.num_particles * (j + 1)]) * -1.)
                errors.append(tf.math.reduce_mean(tf.math.log(tf.maximum(normal_sample[self.num_particles * j:self.num_particles * (j + 1)], 1e-12)) * -1.)) #/ np.log(2.))
            errors = tf.stack(errors)
            #print(float(tf.math.reduce_mean(errors)))
            return errors

    def call(self, input, **kwargs):
        self.reset_mean_variance()
        self.diffused_inputs = []
        self.diffused_inputs.append(input)
        for t in range(self.total_time_steps, 0, -1):
            self.diffused_inputs.insert(0, self.reverse_diffuse(self.diffused_inputs[0], t - 1))

        return self.diffused_inputs[0]

    def compute_loss(self, x = None, y = None, y_pred = None, sample_weight = None):
        self.reset_mean_variance()
        loss = None
        for t in range(self.total_time_steps, 0, -1):
            input = self.diffused_inputs[t - 1]
            if loss is None:
                if t == 1:
                    loss = self.get_loss(input, y, y, t - 1)
                else:
                    loss = self.get_loss(input, y, self.forward_diffuse(y, t - 1), t - 1)
            else:
                if t == 1:
                    loss += self.get_loss(input, y, y, t - 1)
                else:
                    loss += self.get_loss(input, y, self.forward_diffuse(y, t - 1), t - 1)

        return loss


if __name__ == '__main__':
    batch_size = 1000#2
    num_nodes = 4
    num_features = 3

    features = tf.random.normal([batch_size * num_nodes, num_features]) + 3.
    graph = Graph.generate_fully_connected_graph(features, num_nodes, batch_size)

    feature1 = list(map(lambda feature: feature[0], graph.features))

    beta_scheduler = BetaScheduler(0.01, 0.3, 100)
    #for i in range(100):
    #    graph.diffuse(0.05)

    #diffused_feature1 = list(map(lambda feature: feature[0], graph.features))

    gnn = GNN(input_feature_dim = num_features, message_dim = 32, output_feature_dim = num_features)
    #output_graph = gnn([graph, 1])
    #print(graph.features - output_graph.features)
    #gnn.summary()

    model = DiffusiveGenerativeNetwork(num_nodes, BetaScheduler(0.01, 0.3, 100), num_features, 100)
    model_output = model(graph)
    print(model.compute_loss(graph, graph, model_output))
    #print(model_output.features)

    model_feature1 = list(map(lambda feature: feature[0], model_output.features))

    plt.hist(feature1, 100, range = (-5, 8), alpha = 0.5)
    plt.hist(model_feature1, 100, range = (-5, 8), alpha = 0.5)
    plt.show()


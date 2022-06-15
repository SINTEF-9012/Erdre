#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creating deep learning model for estimating power from breathing.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras_tuner import HyperModel
from tensorflow.keras import layers, models, optimizers
from tensorflow.random import set_seed

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

import edward2 as ed

# from cyclemoid_pytorch.easteregg import CycleMoid
from edward2.tensorflow import constraints, initializers, random_variable, regularizers
from edward2.tensorflow.layers import utils

# tf.keras.utils.get_custom_objects()["cyclemoid"] = CycleMoid


def dnn(
    input_size,
    output_length=1,
    activation_function="relu",
    output_activation="linear",
    loss="mse",
    metrics="mse",
    n_layers=2,
    n_nodes=16,
    dropout=0.0,
    seed=2020,
):
    """Define a DNN model architecture using Keras.

    Args:
        input_size (int): Number of features.
        output_length (int): Number of output steps.
        activation_function (str): Activation function in hidden layers.
        output_activation (str): Activation function for outputs.
        loss (str): Loss to penalize during training.
        metrics (str): Metrics to evaluate model.
        n_layers (int): Number of hidden layers.
        n_nodes (int or list of int): Number of nodes in each layer. If int,
            all layers have the same number of nodes. If list, the length of
            the list must match the number of layers, and each integer of the
            list specifies the number of nodes in the corresponding layer.
        dropout (float or list of float): Dropout, either the same for all
            layers, or a list specifying dropout for each layer.
        seed (int): Random seed.

    Returns:
        model (keras model): Model to be trained.

    """

    tf.random.set_seed(seed)

    n_nodes = element2list(n_nodes, n_layers)
    dropout = element2list(dropout, n_layers)

    model = models.Sequential()

    model.add(
        layers.Dense(n_nodes[0], activation=activation_function, input_dim=input_size)
    )

    model.add(layers.Dropout(dropout[0]))

    for i in range(1, n_layers):
        model.add(layers.Dense(n_nodes[i], activation=activation_function))

        model.add(layers.Dropout(dropout[i]))

    model.add(layers.Dense(output_length, activation=output_activation))

    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def cnn(
    input_size_x,
    input_size_y,
    output_length=1,
    kernel_size=2,
    activation_function="relu",
    output_activation="linear",
    loss="mse",
    metrics="mse",
    n_layers=2,
    n_filters=16,
    maxpooling=False,
    maxpooling_size=4,
    n_dense_layers=1,
    n_nodes=16,
    dropout=0.0,
    seed=2020,
):
    """Define a CNN model architecture using Keras.

    Args:
        input_size_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_size_y (int): Number of features for each time step in the input data.
        output_length (int): Number of output steps.
        kernel_size (int): Size of kernel in CNN.
        activation_function (str): Activation function in hidden layers.
        output_activation: Activation function for outputs.
        loss (str): Loss to penalize during training.
        metrics (str): Metrics to evaluate model.
        n_layers (int): Number of hidden layers.
        n_filters (int or list of int): Number of filters in each layer. If int,
            all layers have the same number of filters. If list, the length of
            the list must match the number of layers, and each integer of the
            list specifies the number of filters in the corresponding layer.
        n_dense_layers (int): Number of dense layers after the convolutional
            layers.
        n_nodes (int or list of int): Number of nodes in each layer. If int,
            all layers have the same number of nodes. If list, the length of
            the list must match the number of layers, and each integer of the
            list specifies the number of nodes in the corresponding layer.
        maxpooling (bool): If True, add maxpooling after each Conv1D-layer.
        maxpooling_size (int): Size of maxpooling.
        dropout (float or list of float): Dropout, either the same for all
            layers, or a list specifying dropout for each layer.
        seed (int): Seed for random initialization of weights.

    Returns:
        model (keras model): Model to be trained.

    """

    tf.random.set_seed(seed)

    n_filters = element2list(n_filters, n_layers)
    n_nodes = element2list(n_nodes, n_dense_layers)
    dropout = element2list(dropout, n_layers + n_dense_layers)

    model = models.Sequential()

    model.add(
        layers.Conv1D(
            filters=n_filters[0],
            kernel_size=kernel_size,
            activation=activation_function,
            input_shape=(input_size_x, input_size_y),
            name="input_layer",
            padding="SAME",
        )
    )

    if maxpooling:
        model.add(layers.MaxPooling1D(pool_size=maxpooling_size, name="pool_0"))

    model.add(layers.Dropout(dropout[0]))

    for i in range(1, n_layers):
        model.add(
            layers.Conv1D(
                filters=n_filters[i],
                kernel_size=kernel_size,
                activation=activation_function,
                name=f"conv1d_{i}",
                padding="SAME",
            )
        )

        if maxpooling:
            model.add(layers.MaxPooling1D(pool_size=maxpooling_size, name=f"pool_{i}"))

        model.add(layers.Dropout(dropout[i]))

    model.add(layers.Flatten(name="flatten"))

    for i in range(n_dense_layers):
        model.add(
            layers.Dense(n_nodes[i], activation=activation_function, name=f"dense_{i}")
        )

        model.add(layers.Dropout(dropout[n_layers + i]))

    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )

    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def rnn(
    input_size_x,
    input_size_y,
    output_length=1,
    unit_type="lstm",
    activation_function="relu",
    output_activation="linear",
    loss="mse",
    metrics="mse",
    n_layers=2,
    n_units=16,
    n_dense_layers=1,
    n_nodes=16,
    dropout=0.1,
    seed=2020,
):
    """Define an RNN model architecture using Keras.

    Args:
        input_size_x (int): Number of time steps to include in each sample, i.e. how
            much history is matched with a given target.
        input_size_y (int): Number of features for each time step in the input data.
        output_length (int): Number of output steps.
        unit_type (str): Type of RNN-unit: 'lstm', 'rnn' or 'gru'.
        activation_function (str): Activation function in hidden layers.
            output_activation: Activation function for outputs.
            loss (str): Loss to penalize during training.
        metrics (str): Metrics to evaluate model.
        n_layers (int): Number of hidden layers.
        n_units (int or list of int): Number of units in each layer. If int,
            all layers have the same number of units. If list, the length of
            the list must match the number of layers, and each integer of the
            list specifies the number of units in the corresponding layer.
        n_dense_layers (int): Number of dense layers after the convolutional
            layers.
        n_nodes (int or list of int): Number of nodes in each layer. If int,
            all layers have the same number of nodes. If list, the length of
            the list must match the number of layers, and each integer of the
            list specifies the number of nodes in the corresponding layer.
        dropout (float or list of float): Dropout, either the same for all
            layers, or a list specifying dropout for each layer.
        seed (int): Seed for random initialization of weights.

    Returns:
        model (keras model): Model to be trained.

    """

    tf.random.set_seed(seed)

    n_units = element2list(n_units, n_layers)
    n_nodes = element2list(n_nodes, n_dense_layers)
    dropout = element2list(dropout, n_layers + n_dense_layers)

    return_sequences = True if n_layers > 1 else False

    if unit_type.lower() == "rnn":
        layer = getattr(layers, "SimpleRNN")
    elif unit_type.lower() == "gru":
        layer = getattr(layers, "GRU")
    elif unit_type.lower() == "lstm":
        layer = getattr(layers, "LSTM")
    else:
        layer = getattr(layers, "LSTM")

    model = models.Sequential()

    model.add(
        layer(
            n_units[0],
            input_shape=(input_size_x, input_size_y),
            return_sequences=return_sequences,
            name="rnn_0",
        )
    )

    model.add(layers.Dropout(dropout[0]))

    if return_sequences:
        for i in range(1, n_layers):
            model.add(
                layer(n_units[i], activation=activation_function, name=f"rnn_{i}")
            )

        model.add(layers.Dropout(dropout[i]))

    # Add dense layers
    for i in range(n_dense_layers):
        model.add(
            layers.Dense(n_nodes[i], activation=activation_function, name=f"dense_{i}")
        )

        model.add(layers.Dropout(dropout[n_layers + i]))

    # Output layer
    model.add(
        layers.Dense(output_length, activation=output_activation, name="output_layer")
    )

    # Compile model
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def cnndnn(input_x, input_y, n_forecast_hours, n_steps_out=1):
    """Define a model architecture that combines CNN and DNN.

    Parameters
    ----------
    input_x : int
        Number of time steps to include in each sample, i.e. how much history
        should be matched with a given target.
    input_y : int
        Number of features for each time step, in the input data.
    dense_x: int
        Number of features for the dense part of the network.
    n_steps_out : int
        Number of output steps.
    Returns
    -------
    model : Keras model
        Model to be trained.
    """
    kernel_size = 4

    input_hist = layers.Input(shape=(input_x, input_y))
    input_forecast = layers.Input(shape=((n_forecast_hours,)))

    c = layers.Conv1D(
        filters=64,
        kernel_size=kernel_size,
        activation=activation_function,
        input_shape=(input_x, input_y),
    )(input_hist)
    c = layers.Conv1D(
        filters=32, kernel_size=kernel_size, activation=activation_function
    )(c)
    c = layers.Flatten()(c)
    c = layers.Dense(128, activation=activation_function)(c)
    c = models.Model(inputs=input_hist, outputs=c)

    d = layers.Dense(256, input_dim=n_forecast_hours, activation=activation_function)(
        input_forecast
    )
    d = layers.Dense(128, activation=activation_function)(d)
    d = layers.Dense(64, activation=activation_function)(d)
    d = models.Model(inputs=input_forecast, outputs=d)

    combined = layers.concatenate([c.output, d.output])

    combined = layers.Dense(256, activation=activation_function)(combined)
    combined = layers.Dense(128, activation=activation_function)(combined)
    combined = layers.Dense(64, activation=activation_function)(combined)
    combined = layers.Dense(n_steps_out, activation="linear")(combined)

    model = models.Model(inputs=[c.input, d.input], outputs=combined)

    model.compile(optimizer="adam", loss="mae")

    return model


def bcnn(
    data_size,
    window_size,
    feature_size,
    batch_size,
    kernel_size=5,
    n_steps_out=2,
    classification=False,
    output_activation="linear",
):
    """Creates a Keras model using the temporal bayesian cnn architecture.
    We use the Flipout Monte Carlo estimator for the convolution and fully-connected layers:
    This enables lower variance stochastic gradients than naive reparameterization

     Args:
         data_size: (int )Number of training examples
         window_size: (int ) Number of historical sequence used as an input
         feature_size: (int) Number of features(sensors) used as an input
         batch_size: (int) Size of single batch used as an input
         '
         kernel_size: (int,default : 5) Size of kernel in CNN

         n_steps_out: (int,default : 2)  Number of output classes for classification.
         classification: (boolean, default: False). True if the model is used for classification tasts
         output_activation: (str or tf.nn.activation, default "linear")

     Returns: (model) Compiled Keras model.

    """

    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(
        data_size, dtype=tf.float32
    )
    inputs = layers.Input(shape=(window_size, feature_size))

    layer_1_outputs = tfp.layers.Convolution1DFlipout(
        32,
        kernel_size=kernel_size,
        padding="SAME",
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu,
        name="cnn1",
    )(inputs)
    batch_norm_1_outputs = tf.keras.layers.BatchNormalization(name="batch_norm1")(
        layer_1_outputs
    )
    layer_2_outputs = tfp.layers.Convolution1DFlipout(
        16,
        kernel_size=kernel_size,
        padding="SAME",
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu,
        name="cnn2",
    )(batch_norm_1_outputs)
    batch_norm_2_outputs = tf.keras.layers.BatchNormalization(name="batch_norm2")(
        layer_2_outputs
    )
    flatten_layer_outputs = tf.keras.layers.Flatten(name="flatten_layer")(
        batch_norm_2_outputs
    )
    layer_3_outputs = tfp.layers.DenseFlipout(
        32, kernel_divergence_fn=kl_divergence_function, name="dense1"
    )(flatten_layer_outputs)

    if classification:
        layer_4_outputs = tfp.layers.DenseFlipout(
            n_steps_out,
            kernel_divergence_fn=kl_divergence_function,
            name="dense2",
            activation=output_activation,
        )(layer_3_outputs)
        outputs = tfp.distributions.Categorical(
            logits=layer_4_outputs,
            probs=None,
            dtype=tf.int32,
            validate_args=False,
            allow_nan_stats=True,
            name="Categorical",
        )
    else:
        layer_4_outputs = tfp.layers.DenseFlipout(
            2,
            kernel_divergence_fn=kl_divergence_function,
            name="dense2",
            activation=output_activation,
        )(layer_3_outputs)
        loc = layer_4_outputs[..., :1]
        c = np.log(np.expm1(1.0))
        scale_diag = 1e-5 + tf.nn.softplus(
            c + layer_4_outputs[..., 1:]
        )  ##tf.nn.softplus(outputs[..., 1:]) + 1e-5
        outputs = tf.keras.layers.Concatenate(name="concatenate")([loc, scale_diag])
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :1], scale=t[..., 1:]),
                reinterpreted_batch_ndims=1,
            ),
            name="lambda_normal_dist_layer",
        )(outputs)
    model = models.Model(inputs=inputs, outputs=outputs, name="bvae")
    if classification:
        neg_log_likelihood = lambda x, rv_x: -tf.reduce_mean(
            input_tensor=rv_x.log_prob(x)
        )
    else:
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer, loss=neg_log_likelihood)

    return model


def brnn(data_size, window_size, feature_size, batch_size, hidden_size=5):
    """Creates a Keras model using the temporal  LSTM architecture based on edward2 library.
     We use the Flipout Monte Carlo estimator for the LSTM and fully-connected layers:
    This enables lower variance stochastic gradients than naive reparameterization

     Args:
         data_size: (int )Number of training examples
         window_size: (int ) Number of historical sequence used as an input
         feature_size: (int) Number of features(sensors) used as an input
         batch_size: (int) Size of single batch used as an input
         hidden_size: (int) Number of nodes in lstm hidden layer

     Returns: (model) Compiled Keras model.

    """

    inputs = layers.Input(shape=(window_size, feature_size))

    forward = layers.RNN(
        cell=ed.layers.LSTMCellFlipout(
            units=hidden_size,
            recurrent_dropout=0.1,
            dropout=0.1,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            recurrent_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            activation="relu",
        ),
        return_sequences=True,
    )
    backward = layers.RNN(
        cell=ed.layers.LSTMCellFlipout(
            units=hidden_size,
            recurrent_dropout=0.1,
            dropout=0.1,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            recurrent_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            activation="relu",
        ),
        return_sequences=True,
        go_backwards=True,
    )
    outputs = layers.Bidirectional(layer=forward, backward_layer=backward)(inputs)

    outputs = tf.keras.layers.Flatten()(outputs)

    outputs = ed.layers.DenseFlipout(
        units=hidden_size,
        kernel_regularizer=ed.regularizers.NormalKLDivergence(
            scale_factor=1.0 / data_size
        ),
        activation="relu",
    )(outputs)
    outputs = layers.Dense(2, activation=None)(outputs)

    loc = outputs[..., :1]
    c = np.log(np.expm1(1.0))
    scale_diag = 1e-5 + tf.nn.softplus(
        c + outputs[..., 1:]
    )  ##tf.nn.softplus(outputs[..., 1:]) + 1e-5
    outputs = tf.keras.layers.Concatenate()([loc, scale_diag])
    outputs = tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :1], scale=t[..., 1:]), reinterpreted_batch_ndims=1
        )
    )(outputs)
    model = models.Model(inputs=inputs, outputs=outputs, name="bvae")
    neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    kl = sum(model.losses) / data_size
    model.add_loss(lambda: kl)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer, loss=neg_log_likelihood)

    return model


def bcnn_edward(
    data_size,
    window_size,
    feature_size,
    kernel_size=5,
    n_steps_out=2,
    classification=False,
    output_activation="linear",
    learning_rate=0.001,
):
    """Creates a Keras model using the temporal cnn architecture.
    Args:
        output_activation:
        classification:

    Returns:
        model: Compiled Keras model.
    """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(
        data_size, dtype=tf.float32
    )
    inputs = layers.Input(shape=(window_size, feature_size))

    outputs = ed.layers.Conv1DFlipout(
        filters=32,
        kernel_size=kernel_size,
        kernel_regularizer=ed.regularizers.NormalKLDivergence(
            scale_factor=1.0 / data_size
        ),
        activation=tf.nn.relu,
        name="cnn1",
    )(inputs)
    # outputs=tf.keras.layers.BatchNormalization(name='batch_norm1')(outputs)

    outputs = ed.layers.Conv1DFlipout(
        filters=32,
        kernel_size=kernel_size,
        padding="SAME",
        kernel_regularizer=ed.regularizers.NormalKLDivergence(
            scale_factor=1.0 / data_size
        ),
        activation=tf.nn.relu,
        name="cnn2",
    )(outputs)

    # outputs = ed.layers.Conv1DFlipout(
    #     128, kernel_size=kernel_size, padding='SAME',
    #     kernel_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1. / data_size),
    #     activation=tf.nn.relu, name="cnn3")(outputs)
    outputs = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="SAME")(
        outputs
    )
    # outputs = ed.layers.Conv1DFlipout(
    #     8, kernel_size=kernen_size, padding='SAME',
    #     kernel_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1. / data_size),
    #     activation=tf.nn.relu, name="cnn3",kernel_constraint=kl_divergence_function)(outputs)
    # outputs=tf.keras.layers.BatchNormalization(name='batch_norm2')(outputs)
    outputs = tf.keras.layers.Flatten(name="flatten_layer")(outputs)
    outputs = ed.layers.DenseFlipout(
        units=32,
        kernel_regularizer=ed.regularizers.NormalKLDivergence(
            scale_factor=1.0 / data_size
        ),
        name="dense1",
    )(outputs)

    if classification:
        outputs = ed.layers.DenseFlipout(
            n_steps_out,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            name="dense-0",
            activation=output_activation,
        )(outputs)
        outputs = tfp.distributions.Categorical(
            logits=outputs,
            probs=None,
            dtype=tf.int32,
            validate_args=False,
            allow_nan_stats=True,
            name="Categorical",
        )
    else:
        outputs = ed.layers.DenseFlipout(
            units=2,
            kernel_regularizer=ed.regularizers.NormalKLDivergence(
                scale_factor=1.0 / data_size
            ),
            name="dense0",
            activation=output_activation,
        )(outputs)
        loc = outputs[..., :1]
        c = 0.04  # np.log(np.expm1(1.))
        scale_diag = 1e-5 + tf.nn.softplus(
            c * outputs[..., 1:]
        )  ##tf.nn.softplus(outputs[..., 1:]) + 1e-5
        outputs = tf.keras.layers.Concatenate(name="concatenate")([loc, scale_diag])
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :1], scale=t[..., 1:]),
                reinterpreted_batch_ndims=1,
            ),
            name="lambda_normal_dist_layer",
        )(outputs)
    model = Model(inputs=inputs, outputs=outputs, name="bvae")
    if classification:
        neg_log_likelihood = lambda x, rv_x: -tf.reduce_mean(
            input_tensor=rv_x.log_prob(x)
        )
    else:
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    kl = sum(model.losses) / data_size
    model.add_loss(lambda: kl)
    model.compile(
        optimizer,
        loss=neg_log_likelihood,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )  # keras.metrics.RootMeanSquaredError()

    return model


class SequentialHyperModel(HyperModel):
    def __init__(self, input_x, input_y=0, n_steps_out=1, loss="mse", metrics="mse"):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out
        self.loss = loss
        self.metrics = metrics

    def build(self, hp, seed=2020):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        print(self.loss)

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.Dense(
                units=hp.Int(
                    name="units", min_value=2, max_value=16, step=2, default=8
                ),
                input_dim=self.input_x,
                activation="relu",
                name="input_layer",
            )
        )

        for i in range(hp.Int("num_dense_layers", min_value=0, max_value=4, default=1)):
            model.add(
                layers.Dense(
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=2,
                        max_value=16,
                        step=2,
                        default=8,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)

        return model


class LSTMHyperModel(HyperModel):
    def __init__(self, input_x, input_y=0, n_steps_out=1, loss="mse", metrics="mse"):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out
        self.loss = loss
        self.metrics = metrics

    def build(self, hp, seed=2020, loss="mse", metrics="mse"):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.LSTM(
                hp.Int(
                    name="lstm_units", min_value=4, max_value=256, step=8, default=128
                ),
                input_shape=(self.input_x, self.input_y),
            )
        )  # , return_sequences=True))

        add_dropout = hp.Boolean(name="dropout", default=False)

        if add_dropout:
            model.add(
                layers.Dropout(
                    hp.Float("dropout_rate", min_value=0.1, max_value=0.9, step=0.3)
                )
            )

        for i in range(hp.Int("num_dense_layers", min_value=1, max_value=4, default=2)):
            model.add(
                layers.Dense(
                    # units=64,
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=16,
                        max_value=512,
                        step=16,
                        default=64,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)

        return model


class CNNHyperModel(HyperModel):
    def __init__(self, input_x, input_y, n_steps_out=1, loss="mse", metrics="mse"):
        """Define size of model.

        Args:
            input_x (int): Number of time steps to include in each sample, i.e. how
                much history is matched with a given target.
            input_y (int): Number of features for each time step in the input data.
            n_steps_out (int): Number of output steps.

        """

        self.input_x = input_x
        self.input_y = input_y
        self.n_steps_out = n_steps_out
        self.loss = loss
        self.metrics = metrics

    def build(self, hp, seed=2020, loss="mse", metrics="mse"):
        """Build model.

        Args:
            hp: HyperModel instance.
            seed (int): Seed for random initialization of weights.

        Returns:
            model (keras model): Model to be trained.

        """

        set_seed(seed)

        model = models.Sequential()

        model.add(
            layers.Conv1D(
                input_shape=(self.input_x, self.input_y),
                # filters=64,
                filters=hp.Int(
                    "filters", min_value=8, max_value=256, step=32, default=64
                ),
                # kernel_size=hp.Int(
                #     "kernel_size",
                #     min_value=2,
                #     max_value=6,
                #     step=2,
                #     default=4),
                kernel_size=2,
                activation="relu",
                name="input_layer",
                padding="same",
            )
        )

        for i in range(hp.Int("num_conv1d_layers", 1, 3, default=1)):
            model.add(
                layers.Conv1D(
                    # filters=64,
                    filters=hp.Int(
                        "filters_" + str(i),
                        min_value=8,
                        max_value=256,
                        step=32,
                        default=64,
                    ),
                    # kernel_size=hp.Int(
                    #     "kernel_size_" + str(i),
                    #     min_value=2,
                    #     max_value=6,
                    #     step=2,
                    #     default=4),
                    kernel_size=2,
                    activation="relu",
                    name=f"conv1d_{i}",
                )
            )

        # model.add(layers.MaxPooling1D(pool_size=2, name="pool_1"))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Flatten(name="flatten"))

        for i in range(hp.Int("num_dense_layers", min_value=1, max_value=8, default=2)):
            model.add(
                layers.Dense(
                    # units=64,
                    units=hp.Int(
                        "units_" + str(i),
                        min_value=16,
                        max_value=1024,
                        step=16,
                        default=64,
                    ),
                    activation="relu",
                    name=f"dense_{i}",
                )
            )

        model.add(
            layers.Dense(self.n_steps_out, activation="linear", name="output_layer")
        )
        model.compile(optimizer="adam", loss=self.loss, metrics=self.metrics)

        return model


def element2list(element, expected_length):
    """Take an element an produce a list.

    If the element already is a list of the correct length, nothing will
    change.

    Args:
        element (int, float or list)
        expected_length (int): The length of the list.

    Returns:
        element (list): List of elements with correct length.

    """

    if isinstance(element, int) or isinstance(element, float):
        element = [element] * expected_length
    elif isinstance(element, list):
        assert len(element) == expected_length

    return element

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import datetime
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context

anchor_images_path = "../left"
positive_images_path = "../right"

target_shape = (200, 200)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [anchor_images_path + '/' + f for f in os.listdir(anchor_images_path)]
)
positive_images = sorted(
    [positive_images_path + '/' + f for f in os.listdir(positive_images_path)]
)


image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)


# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

# Create dataset of all anchor, positive and negative
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        norm_anchor = anchor / tf.linalg.norm(anchor, axis=1, keepdims=1)
        norm_positive = positive / tf.linalg.norm(positive, axis=1, keepdims=1)
        norm_negative = negative / tf.linalg.norm(negative, axis=1, keepdims=1)

        ap_distance = tf.reduce_sum(tf.square(norm_anchor - norm_positive), -1)
        an_distance = tf.reduce_sum(tf.square(norm_anchor - norm_negative), -1)
        return (ap_distance, an_distance)


def get_model():
    # ResNet50
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    # Add last layers
    avgpool = layers.GlobalAveragePooling2D()(base_cnn.output)
    flatten = layers.Flatten()(avgpool)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    output = layers.Dense(256)(dense1)

    # Build model
    model = Model(base_cnn.input, output, name="model")

    # Freeze all weight till layer conv5_block1_out
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    # Inputs
    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        model(resnet.preprocess_input(anchor_input)),
        model(resnet.preprocess_input(positive_input)),
        model(resnet.preprocess_input(negative_input)),
    )

    # Siamese network
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        # an_distance >= ap_distance + 0.5
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


# Tensorboard callback
# tensorboard serve --logdir logs/ --port 8080
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

# Save model weights callback function
latest_checkpoint_path = "latest_weights/cp.ckpt"
latest_checkpoint_dir = os.path.dirname(latest_checkpoint_path)
latest_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filepath=latest_checkpoint_dir,
    save_weights_only=True,
    verbose=1)

best_checkpoint_path = "best_weights/cp.ckpt"
best_checkpoint_dir = os.path.dirname(best_checkpoint_path)
best_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filepath=best_checkpoint_dir,
    save_weights_only=True,
    verbose=1,
    mode='min',
    save_best_only=True)

siamese_network = get_model()
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.SGD(0.0001))

if len(sys.argv) > 1:
    siamese_model.load_weights(sys.argv[1])
    print('loaded weights')


# siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=[tensorboard_callback])

siamese_model.fit(train_dataset, epochs=60, validation_data=val_dataset, callbacks=[
                  latest_cp_callback, best_cp_callback, tensorboard_callback])

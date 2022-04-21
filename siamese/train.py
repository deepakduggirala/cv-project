import argparse
import json
import ssl
import datetime
import os
import math

from model import get_model
from triplet_loss import batch_all_triplet_loss
from data import get_dataset, get_ELEP_images_and_labels

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model


ssl._create_default_https_context = ssl._create_unverified_context


class SiameseModel(Model):
    def __init__(self, params, finetune):
        super().__init__()
        self.params = params
        self.finetune = finetune
        self.siamese_network = get_model(params, finetune)
        self.custom_loss = batch_all_triplet_loss
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            loss = self._compute_loss(images, labels)

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
        images, labels = data
        loss = self._compute_loss(images, labels)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, images, labels):
        embeddings = self.siamese_network(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        return self.custom_loss(labels, embeddings, self.params['margin'], self.params['squared'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--save_freq', default=20, type=int,
                        help="save model every 'save_freq' epochs")
    parser.add_argument('--params', default='hyperparameters/initial_run.json',
                        help="JSON file with parameters")
    parser.add_argument('--data_dir', default='../data/',
                        help="Directory containing the dataset")
    parser.add_argument('--log_dir', default='logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--restore_latest', default=False, action='store_true',
                        help="Restart the model from the last Checkpoint")
    parser.add_argument('--restore_best', default=False, action='store_true',
                        help="Restart the model from the best Checkpoint")
    parser.add_argument('--finetune', default=False, action='store_true',
                        help="unfreeze last layers of base model")
    args = parser.parse_args()

    print(args)

    with open(args.params, 'rb') as f:
        params = json.load(f)

    cache_files = {
        'train': 'ELP_train.cache',
        'val': 'ELP_val.cache'
    }
    train_ds, val_ds, N = get_dataset(get_ELEP_images_and_labels, params, args.data_dir, cache_files)

    # Tensorboard callback
    # tensorboard serve --logdir logs/ --port 8080
    log_dir = args.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    train_size = int(N * params['train_size'])
    STEPS_PER_EPOCH = math.ceil(train_size / params['batch_size'])

    # Save model weights callback function
    latest_checkpoint_path = "latest_model/cp.ckpt"
    latest_checkpoint_dir = os.path.dirname(latest_checkpoint_path)
    latest_cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=latest_checkpoint_dir,
        save_weights_only=False,
        verbose=1,
        save_freq=int(args.save_freq * STEPS_PER_EPOCH))

    best_checkpoint_path = "best_weights/cp.ckpt"
    best_checkpoint_dir = os.path.dirname(best_checkpoint_path)
    best_cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=best_checkpoint_dir,
        save_weights_only=True,
        verbose=1,
        mode='min',
        save_best_only=True)

    siamese_model = SiameseModel(params, args.finetune)
    siamese_model.compile(optimizer=optimizers.Adam(params['lr']))

    if args.restore_best:
        siamese_model.load_weights('best_weights')
        print('loaded weights')

    if args.restore_latest:
        siamese_model = load_model('latest_model', compile=False)
        siamese_model.compile(optimizer=optimizers.Adam(params['lr']))
        print('loaded model')

    input_shape = (None, params['image_size'], params['image_size'], 3)
    siamese_model.compute_output_shape(input_shape=input_shape)

    siamese_model.fit(train_ds,
                      epochs=args.epochs,
                      validation_data=val_ds,
                      callbacks=[latest_cp_callback, best_cp_callback, tensorboard_callback])

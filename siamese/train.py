import argparse
import json
import ssl
import datetime
import os
import math
from pathlib import Path

from model import get_model
from triplet_loss import batch_all_triplet_loss
from data import get_dataset, get_ELEP_images_and_labels
from metrics import get_kernel_mask, val

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
        self.val_rate_tracker = metrics.Mean(name="VAL")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            loss, embeddings = self._compute_loss(images, labels)
            loss = loss + sum(self.siamese_network.losses)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "VAL": self.val_rate_tracker.result()}

    def test_step(self, data):
        images, labels = data
        loss, embeddings = self._compute_loss(images, labels)

        # compute Validation Rate metric
        K, mask = get_kernel_mask(labels.numpy(), embeddings.numpy())
        d = self.params['margin'] if self.params['squared'] else self.params['margin']**2
        val_rate = val(K, mask, d, include_diag=True)
        self.val_rate_tracker.update_state(val_rate)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "VAL": self.val_rate_tracker.result()}

    def _compute_loss(self, images, labels):
        embeddings = self.siamese_network(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        return self.custom_loss(labels, embeddings, self.params['margin'], self.params['squared']), embeddings


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
    parser.add_argument('--restore_latest', default=False,
                        help="Restart the model from the last Checkpoint")
    parser.add_argument('--restore_best', default=False,
                        help="Restart the model from the best Checkpoint")
    parser.add_argument('--finetune', default=False, action='store_true',
                        help="unfreeze last layers of base model")
    args = parser.parse_args()

    print(args)

    with open(args.params, 'rb') as f:
        params = json.load(f)

    print(params)

    cache_files = {
        'train': str(Path(args.data_dir) / 'train.cache'),
        'val': str(Path(args.data_dir) / 'val.cache')
    }
    # train_ds, val_ds, N = get_dataset(get_ELEP_images_and_labels, params, args.data_dir, cache_files)
    train_ds, N_train = get_dataset(get_ELEP_images_and_labels, params, args.data_dir, 'train', cache_files)
    val_ds, N_val = get_dataset(get_ELEP_images_and_labels, params, args.data_dir, 'val', cache_files)

    RUN_DATETIME_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Tensorboard callback
    # tensorboard serve --logdir logs/ --port 8080
    log_dir = str(Path(args.log_dir) / RUN_DATETIME_STR)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    # STEPS_PER_EPOCH = math.ceil(N_train / params['batch_size']['train'])

    # # Save model weights callback function
    # filepath = Path('latest_models') / RUN_DATETIME_STR / 'model.ckpt'
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # latest_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     monitor='val_loss',
    #     filepath=str(filepath),
    #     save_weights_only=False,
    #     verbose=1,
    #     save_freq=int(args.save_freq * STEPS_PER_EPOCH))

    filepath = Path('best_weights') / RUN_DATETIME_STR / 'weights.ckpt'
    filepath.parent.mkdir(parents=True, exist_ok=True)
    best_cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=str(filepath),
        save_weights_only=True,
        verbose=1,
        mode='min',
        save_best_only=True)

    siamese_model = SiameseModel(params, args.finetune)
    siamese_model.compile(optimizer=optimizers.Adam(params['lr']), run_eagerly=True)

    if args.restore_best:
        weights_path = str(Path(args.restore_best) / 'weights.ckpt')
        siamese_model.load_weights(weights_path)
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
                      callbacks=[best_cp_callback, tensorboard_callback])

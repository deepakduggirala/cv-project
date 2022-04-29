import argparse
import json
import ssl
import datetime
import os
import math
from pathlib import Path
import numpy as np

from model import get_model, enable_finetune
from triplet_loss import batch_all_triplet_loss, val, far, batch_hard_triplet_loss, adapted_triplet_loss
from data import get_dataset, get_ELEP_images_and_labels, get_zoo_elephants_images_and_labels

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model


ssl._create_default_https_context = ssl._create_unverified_context


class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0):
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            val_ds, validation_set_name = validation_set

            results = self.model.evaluate(val_ds,
                                          verbose=self.verbose,
                                          return_dict=True)

            for metric, result in results.items():
                valuename = validation_set_name + '_' + metric
                logs[valuename] = result


class SiameseModel(Model):
    def __init__(self, params, finetune):
        super().__init__()
        self.params = params
        self.finetune = finetune
        self.siamese_network = get_model(params, finetune)
        self.custom_loss = batch_all_triplet_loss
        self.val_metric = val
        self.far_metric = far
        self.loss_tracker = metrics.Mean(name="loss")
        self.val_rate_tracker = metrics.Mean(name="VAL")
        self.far_rate_tracker = metrics.Mean(name="FAR")

        if self.params['triplet_strategy'] == "batch_all":
            self.custom_loss = batch_all_triplet_loss

        elif self.params['triplet_strategy'] == "batch_hard":
            self.custom_loss = batch_hard_triplet_loss

        elif self.params['triplet_strategy'] == "batch_adaptive":
            self.custom_loss = adapted_triplet_loss

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

        val_rate = self.val_metric(labels, embeddings, d=self.params['metrics_d'],  squared=self.params['squared'])
        self.val_rate_tracker.update_state(val_rate)

        far_rate = self.far_metric(labels, embeddings, d=self.params['metrics_d'],  squared=self.params['squared'])
        self.far_rate_tracker.update_state(far_rate)

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "VAL": self.val_rate_tracker.result(), "FAR": self.far_rate_tracker.result()}

    def test_step(self, data):
        images, labels = data
        loss, embeddings = self._compute_loss(images, labels)

        # compute Validation Rate metric
        val_rate = self.val_metric(labels, embeddings, d=self.params['metrics_d'],  squared=self.params['squared'])
        self.val_rate_tracker.update_state(val_rate)

        far_rate = self.far_metric(labels, embeddings, d=self.params['metrics_d'],  squared=self.params['squared'])
        self.far_rate_tracker.update_state(far_rate)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "VAL": self.val_rate_tracker.result(), "FAR": self.far_rate_tracker.result()}

    def _compute_loss(self, images, labels):
        embeddings = self.siamese_network(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        return self.custom_loss(labels, embeddings, self.params['margin'], self.params['squared']), embeddings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--epochs2', default=0, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--save_freq', default=20, type=int,
                        help="save model every 'save_freq' epochs")
    parser.add_argument('--params', default='hyperparameters/init.json',
                        help="JSON file with parameters")
    parser.add_argument('--data_dir', default='../data/',
                        help="Directory containing the dataset")
    parser.add_argument('--additional_data_dir', default=False,
                        help="Directory containing the additional dataset for validation")
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

    rand_string = ''.join(np.random.choice([chr(i) for i in range(48, 58)] + [chr(i) for i in range(97, 123)], 5))
    RUN_DATETIME_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + rand_string
    print('\n', RUN_DATETIME_STR, '\n')

    cache_files = {
        'train': str(Path(args.data_dir) / 'train.cache'),
        'val': str(Path(args.data_dir) / 'val.cache')
    }

    train_ds, N_train, _ = get_dataset(get_ELEP_images_and_labels,
                                       params,
                                       str(Path(args.data_dir)/'train'),
                                       augment=True,
                                       cache_file=cache_files['train'],
                                       shuffle=True,
                                       batch_size=params['batch_size']['train'])

    val_ds, _, _ = get_dataset(get_ELEP_images_and_labels,
                               params,
                               str(Path(args.data_dir)/'val'),
                               augment=False,
                               cache_file=cache_files['val'],
                               shuffle=False,
                               batch_size=params['batch_size']['val'])

    if args.additional_data_dir:
        val_2_ds, _, _ = get_dataset(get_zoo_elephants_images_and_labels,
                                    params,
                                    str(Path(args.additional_data_dir)),
                                    augment=False,
                                    cache_file=str(Path(args.additional_data_dir) / 'val.cache'),
                                    shuffle=False,
                                    batch_size=params['batch_size']['val'])
        additional_val_cb = AdditionalValidationSets([(val_2_ds, 'val_2')], verbose=0)

    # train_ds = train_ds.take(1)
    # val_ds = val_ds.take(1)
    # val_2_ds = val_2_ds.take(1)

    # Tensorboard callback
    # tensorboard serve --logdir logs/ --port 8080
    log_dir = str(Path(args.log_dir) / RUN_DATETIME_STR)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    STEPS_PER_EPOCH = math.ceil(N_train / params['batch_size']['train'])

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

    filepath = Path('latest_weights') / RUN_DATETIME_STR / 'weights.ckpt'
    filepath.parent.mkdir(parents=True, exist_ok=True)
    latest_cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=str(filepath),
        save_weights_only=True,
        verbose=1,
        save_freq=int(args.save_freq * STEPS_PER_EPOCH))

    

    # Create and compile model
    siamese_model = SiameseModel(params, args.finetune)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        params['lr'],
        decay_steps=params['decay_steps'],
        decay_rate=params['decay_rate'],
        staircase=True)
    # siamese_model.compile(optimizer=optimizers.SGD(learning_rate=lr_schedule))
    siamese_model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule))

    if args.restore_best:
        weights_path = str(Path(args.restore_best) / 'weights.ckpt')
        siamese_model.load_weights(weights_path)
        print('loaded best weights')

    if args.restore_latest:
        weights_path = str(Path(args.restore_latest) / 'weights.ckpt')
        siamese_model.load_weights(weights_path)
        print('loaded latest weights')

    # if args.restore_latest:
    #     siamese_model = load_model('latest_model', compile=False)
    #     siamese_model.compile(optimizer=optimizers.Adam(params['lr']))
    #     print('loaded model')

    input_shape = (None, params['image_size'], params['image_size'], 3)
    siamese_model.compute_output_shape(input_shape=input_shape)

    if args.additional_data_dir:
      callbacks = [additional_val_cb, latest_cp_callback, best_cp_callback, tensorboard_callback]
    else:
      callbacks = [latest_cp_callback, best_cp_callback, tensorboard_callback]

    siamese_model.fit(train_ds,
                      epochs=args.epochs,
                      validation_data=val_ds,
                      callbacks=callbacks)

    if args.epochs2:
        base_model = siamese_model.siamese_network.layers[1]
        enable_finetune(params, base_model)

        # siamese_model.compile(optimizer=optimizers.Adam(learning_rate=params['lr']))
        siamese_model.fit(train_ds,
                          epochs=args.epochs2,
                          validation_data=val_ds,
                          callbacks=callbacks)

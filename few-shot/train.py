import argparse
import json
import ssl
import datetime
import os
import math
from pathlib import Path
import numpy as np
import time


import tensorflow as tf
from tensorflow.keras import optimizers

from sklearn.preprocessing import OneHotEncoder

from model import FewShotModel
from data import get_dataset, get_zoo_elephants_images_and_labels, get_support_and_query_sets, get_ELEP_images_and_labels


ssl._create_default_https_context = ssl._create_unverified_context


def get_support_class_means(preds, categories, support_labels):
    d_out = categories.shape[0]
    d_in = preds.shape[1]
    class_means = np.zeros((d_out, d_in))
    for i, c in enumerate(categories):
        mask = np.array(support_labels) == c
        class_means[i, :] = np.mean(preds[mask, :], axis=0)
    return class_means.astype(np.float32).T


def get_w_init(params, base_model, support_image_paths, support_labels, categories):

    support_ds, _, _ = get_dataset(support_image_paths, support_labels,
                                   params,
                                   augment=False,
                                   cache_file=None,
                                   shuffle=False,
                                   batch_size=32)

    preds = base_model.predict(support_ds, verbose=True)
    preds = preds/np.linalg.norm(preds, axis=1, keepdims=1)

    return get_support_class_means(preds, categories, support_labels)


def my_loss_fn(y_true, y_pred, C=0.1):
    # tf.print(y_pred.shape)
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    entropy_loss = -tf.reduce_sum(y_pred * tf.math.log(y_pred), 1)
    return cross_entropy_loss + C * entropy_loss


def make_callbacks(args, params, N_train):
    rand_string = ''.join(np.random.choice([chr(i) for i in range(48, 58)] + [chr(i) for i in range(97, 123)], 5))
    RUN_DATETIME_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + rand_string
    print('\n', RUN_DATETIME_STR, '\n')

    # Tensorboard callback
    # tensorboard serve --logdir logs/ --port 8080
    log_dir = str(Path(args.log_dir) / RUN_DATETIME_STR)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    # filepath = Path('best_weights') / RUN_DATETIME_STR / 'weights.ckpt'
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # best_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     monitor='val_loss',
    #     filepath=str(filepath),
    #     save_weights_only=True,
    #     verbose=1,
    #     mode='min',
    #     save_best_only=True)

    # STEPS_PER_EPOCH = math.ceil(N_train / params['batch_size']['train'])
    # filepath = Path('latest_weights') / RUN_DATETIME_STR / 'weights.ckpt'
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # latest_cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     monitor='val_loss',
    #     filepath=str(filepath),
    #     save_weights_only=True,
    #     verbose=1,
    #     save_freq=int(args.save_freq * STEPS_PER_EPOCH))

    # return [latest_cp_callback, best_cp_callback, tensorboard_callback]
    return [tensorboard_callback]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--epochs2', default=0, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--n_support', default=5, type=int,
                        help="number of images of each class in the support set")
    parser.add_argument('--params', default='hyperparameters/init.json',
                        help="JSON file with parameters")
    parser.add_argument('--data_dir', default='../data/',
                        help="Directory containing the dataset")
    parser.add_argument('--additional_data_dir', default='../data/',
                        help="Directory containing the additional dataset for validation")
    parser.add_argument('--log_dir', default='logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--restore_latest', default=False,
                        help="Restart the model from the last Checkpoint")
    parser.add_argument('--restore_best', default=False,
                        help="Restart the model from the best Checkpoint")
    parser.add_argument('--finetune', default=False, action='store_true',
                        help="unfreeze last layers of base model")
    parser.add_argument('--do_not_augment', default=False, action='store_true',
                        help="use data aumentation")
    args = parser.parse_args()

    print(args)

    with open(args.params, 'rb') as f:
        params = json.load(f)

    print(params)

    image_paths, image_labels = get_zoo_elephants_images_and_labels(args.data_dir)
    support_image_paths, support_labels, query_image_paths, query_labels = get_support_and_query_sets(
        image_paths, image_labels, args.n_support)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    support_labels_enc = enc.fit_transform(np.array(support_labels).reshape(-1, 1))
    query_labels_enc = enc.transform(np.array(query_labels).reshape(-1, 1))

    # cache_files = {
    #     'train': str(Path(args.data_dir) / 'few_shot_train.cache'),
    #     'val': str(Path(args.data_dir) / 'few_shot_val.cache'),
    # }

    train_ds, N_train, _ = get_dataset(support_image_paths, support_labels_enc,
                                       params,
                                       augment=True,
                                       cache_file=None,
                                       shuffle=True,
                                       batch_size=params['batch_size']['train'])

    val_ds, _, _ = get_dataset(query_image_paths, query_labels_enc,
                               params,
                               augment=False,
                               cache_file=None,
                               shuffle=False,
                               batch_size=params['batch_size']['val'])

    # train_ds = train_ds.take(1)
    # val_ds = val_ds.take(1)

    # Create and compile model
    model_cnt = FewShotModel(params)

    w_init = get_w_init(params, model_cnt.base_model, support_image_paths,
                        support_labels, categories=enc.categories_[0])
    few_shot_model = model_cnt.get_model(w_init)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     params['lr'],
    #     decay_steps=params['decay_steps'],
    #     decay_rate=params['decay_rate'],
    #     staircase=True)
    # siamese_model.compile(optimizer=optimizers.SGD(learning_rate=lr_schedule))

    top_3_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
    top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    few_shot_model.compile(
        optimizer=optimizers.Adam(learning_rate=params['lr']),
        loss=my_loss_fn,
        metrics=['accuracy', top_3_acc, top_5_acc])

    result = few_shot_model.evaluate(val_ds)
    print('before finetuning', result)

    if args.restore_best:
        weights_path = str(Path(args.restore_best) / 'weights.ckpt')
        few_shot_model.load_weights(weights_path)
        print('loaded best weights')

    if args.restore_latest:
        weights_path = str(Path(args.restore_latest) / 'weights.ckpt')
        few_shot_model.load_weights(weights_path)
        print('loaded latest weights')

    input_shape = (None, params['image_size'], params['image_size'], 3)
    few_shot_model.compute_output_shape(input_shape=input_shape)

    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    callbacks = make_callbacks(args, params, N_train)

    few_shot_model.fit(train_ds,
                       epochs=args.epochs,
                       validation_data=val_ds,
                       callbacks=callbacks)

    # if args.epochs2:
    #     base_model = few_shot_model.layers[1]
    #     enable_finetune(params, base_model)

    #     # siamese_model.compile(optimizer=optimizers.Adam(learning_rate=params['lr']))
    #     few_shot_model.fit(train_ds,
    #                        epochs=args.epochs2,
    #                        validation_data=val_ds,
    #                        callbacks=callbacks)

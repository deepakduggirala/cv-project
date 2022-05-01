import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from pathlib import Path
import numpy as np


def preprocess_image(image, image_size, augment=True, model_preprocess=True):
    if augment:
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, 0.2)
        # image = tf.image.random_contrast(image, 0.5, 2.0)
        image = tf.image.random_saturation(image, 0.75, 1.25)
        image = tf.image.random_hue(image, 0.05)
        # image = tf.image.random_jpeg_quality(image, 20, 100)
    if model_preprocess:
        image = preprocess_input(image)
    return image


def parse_image_function(image_path, image_size, resize_pad=False):
    # print('reading', image_path)
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    if not resize_pad:
        image = tf.image.resize(image, [image_size, image_size])
    else:
        image = tf.image.resize_with_pad(image, target_height=image_size, target_width=image_size)
    # image = preprocess_image(image, image_size, augment)
    return image


def get_ELEP_images_and_labels(dir_path):
    root_dir = Path(dir_path)
    image_paths = [p for p in sorted(root_dir.iterdir()) if p.suffix in ['.jpg']]
    image_paths_str = [str(p) for p in image_paths]
    image_labels = [img_path.name.split('_')[0] for img_path in image_paths]
    return image_paths_str, image_labels


def get_zoo_elephants_images_and_labels(dir_path):
    root_dir = Path(dir_path)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    x = [(str(img_path), c.name) for c in class_dirs for img_path in c.iterdir() if img_path.suffix in ['.png']]
    images_paths, image_labels = list(zip(*x))
    return list(images_paths), list(image_labels)


def get_dataset(f, params, dir_path,
                augment=None, cache_file=None, model_preprocess=True,
                shuffle=True, batch_size=32):

    image_paths, image_labels = f(dir_path)
    N = len(image_labels)

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(lambda x, y: (parse_image_function(
        x, params['image_size'], resize_pad=params['resize_pad']), y), num_parallel_calls=AUTOTUNE)

    if cache_file:
        dataset = dataset.cache(cache_file)

    dataset = dataset.map(lambda x, y: (
        preprocess_image(x, params['image_size'], augment=augment, model_preprocess=model_preprocess), y),
        num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=N)

    if batch_size:
        dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset, N, np.array(image_labels)


def get_eval_dataset(
        f, params, dir_path, cache_file=None, batch_size=32):
    image_paths, image_labels = f(dir_path)
    N = len(image_labels)

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: parse_image_function(
        x, params['image_size'], resize_pad=params['resize_pad']), num_parallel_calls=tf.data.AUTOTUNE)

    if cache_file:
        dataset = dataset.cache(cache_file)

    dataset = dataset.map(lambda x: (preprocess_image(
        x, params['image_size'], augment=False)), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset, np.array(image_labels)

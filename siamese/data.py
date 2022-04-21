import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from pathlib import Path


def preprocess_image(image, image_size):

    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.random_flip_left_right(image)
    image = preprocess_input(image)
    return image


def parse_image_function(image_path, label, image_size):

    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image, image_size)
    return image, label


def get_ELEP_images_and_labels(dir_path):
    root_dir = Path(dir_path)
    image_paths = [str(p) for p in root_dir.iterdir()]
    image_labels = [img_path.name.split('_')[0] for img_path in root_dir.iterdir()]
    return image_paths, image_labels


def get_zoo_elephants_images_and_labels(dir_path):
    root_dir = Path(dir_path)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    x = [(str(img_path), c.name) for c in class_dirs for img_path in c.iterdir() if img_path.suffix in ['.png']]
    images_paths, image_labels = list(zip(*x))
    return list(images_paths), list(image_labels)


def get_dataset(f, params, dir_path, cache_files=None):

    image_paths, image_labels = f(dir_path)
    N = len(image_labels)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.shuffle(buffer_size=N)

    train_ds = dataset.take(round(N * params['train_size']))
    val_ds = dataset.skip(round(N * params['train_size']))

    if cache_files:
        train_ds = train_ds.cache(cache_files['train'])
        val_ds = val_ds.cache(cache_files['val'])
    else:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.shuffle(buffer_size=N, reshuffle_each_iteration=True)
    # val_ds   = val_ds.shuffle(buffer_size=N, reshuffle_each_iteration=True)

    train_ds = train_ds.map(lambda x, y: parse_image_function(x, y, params['image_size']))
    train_ds = train_ds.batch(params['batch_size']).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: parse_image_function(x, y, params['image_size']))
    val_ds = val_ds.batch(params['val_batch_size']).prefetch(AUTOTUNE)

    return train_ds, val_ds, N

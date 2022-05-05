import numpy as np
from collections import Counter
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def shuffle(n):
    x = np.arange(n, dtype=np.int32)
    np.random.shuffle(x)
    return x


def get_support_and_query_sets(image_paths, image_labels, n_support, seed=99):
    support_images = []
    support_labels = []

    query_images = []
    query_labels = []

    np.random.seed(seed)
    counts = Counter(image_labels)
    shuffled_idxs = {c: shuffle(count) for c, count in counts.items()}

    for c, idxs in shuffled_idxs.items():
        s_idxs = idxs[:n_support]
        q_idxs = idxs[n_support:]

        mask = np.array(image_labels) == c
        c_image_labels = np.array(image_labels)[mask]
        c_image_paths = np.array(image_paths)[mask]

        support_images.extend(c_image_paths[s_idxs])
        support_labels.extend(c_image_labels[s_idxs])
        query_images.extend(c_image_paths[q_idxs])
        query_labels.extend(c_image_labels[q_idxs])

    return support_images, support_labels, query_images, query_labels


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

    x = [(str(img_path), c.name) for c in class_dirs for img_path in sorted(c.iterdir()) if img_path.suffix in ['.png']]
    images_paths, image_labels = list(zip(*x))
    return list(images_paths), list(image_labels)


def get_dataset(image_paths, image_labels, params,
                augment=None, cache_file=None, model_preprocess=True,
                shuffle=True, batch_size=32):
    N = len(image_labels)

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(lambda x, y: (parse_image_function(
        x, params['image_size'], resize_pad=params['resize_pad']), y), num_parallel_calls=AUTOTUNE)

    if cache_file:
        dataset = dataset.cache(cache_file)
    else:
        dataset = dataset.cache()
        

    dataset = dataset.map(lambda x, y: (
        preprocess_image(x, params['image_size'], augment=augment, model_preprocess=model_preprocess), y),
        num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=N)

    if batch_size:
        dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset, N, image_labels


def get_embeddings(image_paths, image_labels, params, base_model, n_repeat=1, cache_file=None):
    ds_aug, _, _ = get_dataset(image_paths, image_labels,
                               params,
                               augment=True,
                               cache_file=cache_file,
                               shuffle=False,
                               batch_size=32)
    ds_aug = ds_aug.repeat(n_repeat)
    embeddings_aug = base_model.predict(ds_aug, verbose=True)

    ls = np.array(image_labels)

    return embeddings_aug, np.hstack([ls]*n_repeat)


def load_embeddings(embs_path, labels_path):
    embs = np.load('embeddings.npy')
    ls = np.load('image_labels.npy')

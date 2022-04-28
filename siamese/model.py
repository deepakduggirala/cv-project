import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# class face_model(tf.keras.Model):

#     def __init__(self, params):
#         super().__init__()
#         img_size = (params.image_size, params.image_size, 3)
#         self.base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=img_size)
#         self.base_model.trainable = False
#         self.flatten = tf.keras.layers.Flatten()
#         self.embedding_layer = tf.keras.layers.Dense(units=params.embedding_size)

#     def call(self, images):
#         x = self.base_model(images)
#         x = self.flatten(x)
#         x = self.embedding_layer(x)
#         return x

def get_model_init(params, finetune=False):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(
        params['image_size'], params['image_size'], 3), pooling='avg')

    if finetune:
        base_model = unfreeze_resnet(base_model, from_layer="conv5_block2_out")
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(params['image_size'], params['image_size'], 3))
    x = base_model(inputs, training=False)
    embedding_layer = tf.keras.layers.Dense(
        units=params['embedding_size'],
        kernel_regularizer=tf.keras.regularizers.L2(params['dense_l2_reg_c']))(x)
    return tf.keras.Model(inputs, embedding_layer)


def get_model_dw(params, finetune=False):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(
        params['image_size'], params['image_size'], 3))

    if finetune:
        base_model = unfreeze_resnet(base_model, from_layer="conv5_block2_out")
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(params['image_size'], params['image_size'], 3))
    x = base_model(inputs, training=False)
    if params['use_avg_pool']:
        flatten = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(8, 8), activation='relu')(x)
        flatten = tf.keras.layers.Flatten()(dw_conv)
    dropout1 = tf.keras.layers.Dropout(rate=params['dropout1_rate'])(flatten)

    if params['embedding_size'] < 256:
        dense1 = tf.keras.layers.Dense(units=512, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(rate=params['dropout2_rate'])(dense1)
        embedding_layer = tf.keras.layers.Dense(units=params['embedding_size'])(dropout2)
    else:
        embedding_layer = tf.keras.layers.Dense(units=params['embedding_size'])(dropout1)

    return tf.keras.Model(inputs, embedding_layer)


def get_model_dw2(params, finetune=False):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(
        params['image_size'], params['image_size'], 3))

    if finetune:
        base_model = unfreeze_resnet(base_model, from_layer="conv5_block2_out")
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(params['image_size'], params['image_size'], 3))
    x = base_model(inputs, training=False)
    dropout1 = tf.keras.layers.Dropout(rate=params['dropout1_rate'])(x)

    conv1x1 = tf.keras.layers.Conv2D(kernel_size=1, filters=256, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=params['dropout2_rate'])(conv1x1)

    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(8, 8))(dropout2)
    flatten = tf.keras.layers.Flatten()(dw_conv)
    dropout3 = tf.keras.layers.Dropout(rate=params['dropout2_rate'])(flatten)

    embedding_layer = tf.keras.layers.Dense(
        units=params['embedding_size'],
        kernel_regularizer=tf.keras.regularizers.L2(params['dense_l2_reg_c']))(dropout3)
    return tf.keras.Model(inputs, embedding_layer)


def inception_model(params, finetune=False):
    s = params['image_size']
    d = params['embedding_size']
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(s, s, 3))

    if finetune:
        base_model = unfreeze_inception(base_model)
    else:
        for layer in base_model.layers:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(s, s, 3))
    x = base_model(inputs, training=False)
    flatten = tf.keras.layers.GlobalAveragePooling2D()(x)
    embedding_layer = tf.keras.layers.Dense(units=d, kernel_regularizer=tf.keras.regularizers.L2(
        params['dense_l2_reg_c']))(flatten)
    return tf.keras.Model(inputs, embedding_layer)


def inception_model_dw(params, finetune=False):
    s = params['image_size']
    d = params['embedding_size']
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(s, s, 3))

    if finetune:
        base_model = unfreeze_inception(base_model)
    else:
        for layer in base_model.layers:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(s, s, 3))
    x = base_model(inputs, training=False)
    dropout1 = tf.keras.layers.Dropout(rate=params['dropout1_rate'])(x)

    conv1x1 = tf.keras.layers.Conv2D(kernel_size=1, filters=256, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=params['dropout2_rate'])(conv1x1)

    dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(6, 6))(dropout2)
    flatten = tf.keras.layers.Flatten()(dw_conv)
    dropout3 = tf.keras.layers.Dropout(rate=params['dropout2_rate'])(flatten)

    embedding_layer = tf.keras.layers.Dense(
        units=d,
        kernel_regularizer=tf.keras.regularizers.L2(params['dense_l2_reg_c']))(dropout3)
    return tf.keras.Model(inputs, embedding_layer)


def get_model(params, finetune=False):
    if params['name'] == 'init':
        return get_model_init(params, finetune)
    elif params['name'] == 'depthwise':
        return get_model_dw(params, finetune)
    elif params['name'] == 'depthwise2':
        return get_model_dw2(params, finetune)
    elif params['name'] == 'inceptionv3':
        return inception_model(params, finetune)
    elif params['name'] == 'inceptionv3_dw':
        return inception_model_dw(params, finetune)
    else:
        return None


def unfreeze_resnet(base_model, from_layer='conv5_block2_out'):
    # Freeze all weight till layer conv5_block1_out
    base_model.trainable = True
    trainable = False
    for layer in base_model.layers:
        if layer.name == from_layer:
            trainable = True
        layer.trainable = trainable
    return base_model


def unfreeze_inception(base_model):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True
    return base_model


def enable_finetune(params, base_model, from_layer='conv5_block2_out'):
    if params['name'] == 'init' or 'depthwise' or 'depthwise2':
        return unfreeze_resnet(base_model, from_layer)
    elif params['name'] == 'inceptionv3' or 'inceptionv3_dw':
        return unfreeze_inception(base_model)
    else:
        return None

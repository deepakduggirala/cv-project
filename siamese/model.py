import os
import tensorflow as tf

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

def get_model(params, finetune=False):
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights="imagenet",
                                                  input_shape=(params['image_size'], params['image_size'], 3),
                                                  pooling='avg')

    if finetune:
        # Freeze all weight till layer conv5_block1_out
        trainable = False
        for layer in base_model.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=(params['image_size'], params['image_size'], 3))
    x = base_model(inputs, training=False)
    embedding_layer = tf.keras.layers.Dense(
        units=params['embedding_size'],
        kernel_regularizer=tf.keras.regularizers.L2(params['dense_l2_reg_c']))(x)
    return tf.keras.Model(inputs, embedding_layer)

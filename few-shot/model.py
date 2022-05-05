import tensorflow as tf


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, w_init, name='custom_layer'):
        super(CustomLayer, self).__init__(name=name)
        units = w_init.shape[1]
        self.w = tf.Variable(
            initial_value=w_init,
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        embeddings = tf.math.l2_normalize(inputs, axis=1, epsilon=1e-10)  # b x 2048, w: 2048 x 17
        t = tf.matmul(embeddings, self.w)                                # b x 17
        t = tf.math.l2_normalize(t, axis=1, epsilon=1e-10)
        z = t + self.b
        return tf.keras.activations.softmax(z, axis=-1)


class FewShotModel():
    def __init__(self, params, use_base_model=True):
        self.params = params
        self.use_base_model = use_base_model
        self.base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(
            params['image_size'], params['image_size'], 3), pooling='avg')

    def get_model(self, w_init):
        if self.use_base_model:
            inputs = tf.keras.Input(shape=(self.params['image_size'], self.params['image_size'], 3))

            x = self.base_model(inputs, training=False)

            # dense1 = tf.keras.layers.Dense(units=17, activation='softmax', name='dense1')(x)
            custom_layer = CustomLayer(w_init=w_init, name='custom_layer')(x)
            model = tf.keras.Model(inputs, custom_layer)
            return model
        else:
            inputs = tf.keras.Input(shape=(2048,))
            custom_layer = CustomLayer(w_init=w_init, name='custom_layer')(x)
            model = tf.keras.Model(inputs, custom_layer)
            return model

import tensorflow as tf

class DecomNet(tf.keras.Model):
    def __init__(self, kernel_size=3, num_layers=5, channels=5, input_shape=(None, 400, 600, 3), **kwargs):
        super().__init__(kwargs)
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.image_shape = input_shape

    def build(self, inputs):
        self.l1 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=None, input_shape=self.image_shape)
        self.mids = [tf.keras.layers.Conv2D(
            4, self.kernel_size, padding='same', activation=tf.keras.activations.relu) for _ in range(self.num_layers)]
        self.out = tf.keras.layers.Conv2D(
            4, self.kernel_size, padding='same', activation=None)

    def call(self, inputs):
        input_max = tf.reduce_max(inputs, axis=3, keepdims=True)
        input_im = tf.concat([input_max,inputs], axis=3)
        x = self.l1(input_im)
        for i in range(self.num_layers):
            x = self.mids[i](x)
            # x=tf.keras.layers.Conv2D(4,self.kernel_size,padding='same',activation=tf.keras.activations.relu)(x)
        # x=tf.keras.layers.Conv2D(4,self.kernel_size,padding='same',activation=None)(x)
        x = self.out(x)
        R = tf.keras.activations.sigmoid(x[:, :, :, :3])
        L = tf.keras.activations.sigmoid(x[:, :, :, 3:4])
        return R, L

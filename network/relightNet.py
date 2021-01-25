import tensorflow as tf


class RelightNet(tf.keras.Model):
    def __init__(self, kernel_size=3, num_layers=5, channels=5, input_shape=(None, 400, 600, 3), **kwargs):
        super().__init__(kwargs)
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.image_shape = input_shape

    def build(self, inputs):
        self.conv0 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=None, input_shape=self.image_shape)

        self.conv1 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu, strides=2)
        self.deconv1 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu)
        self.deconv2 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu)
        self.deconv3 = tf.keras.layers.Conv2D(
            self.channels, self.kernel_size, padding='same', activation=tf.keras.activations.relu)

        self.fusion = tf.keras.layers.Conv2D(
            self.channels, 1, padding='same', activation=None)

        self.out = tf.keras.layers.Conv2D(
            1, 3, padding='same', activation=None)

    def call(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        up1 = tf.image.resize(
            conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]), "nearest")
        deconv1 = self.deconv1(up1)+conv2
        up2 = tf.image.resize(
            deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]), "nearest")
        deconv2 = self.deconv2(up2)+conv1
        up3 = tf.image.resize(
            deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = self.deconv3(up3)+conv0
        x11 = tf.image.resize(
            deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), "nearest")
        x12 = tf.image.resize(
            deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]), "nearest")
        x13 = self.fusion(tf.concat([x11, x12, deconv3], 3))
        out = self.out(x13)
        # print(tf.shape(out))
        return out

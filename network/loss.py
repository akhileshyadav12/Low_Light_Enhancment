
import tensorflow as tf
def concat(layers):
    return tf.concat(layers, axis=3)
# %%


def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant(
        [[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    x = tf.nn.conv2d(tf.expand_dims(
        input_tensor[:, :, :, 0], -1), kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.abs(x)


def ave_gradient(input_tensor, direction):
    return tf.nn.avg_pool2d(gradient(input_tensor, direction), ksize=3, strides=1, padding='SAME')


def smooth(input_I, input_R):
    #print(tf.shape(input_R))
    input_R = tf.image.rgb_to_grayscale(input_R)
    return tf.reduce_mean(gradient(input_I, "x") * tf.exp(-10 * ave_gradient(input_R, "x")) + gradient(input_I, "y") * tf.exp(-10 * ave_gradient(input_R, "y")))


def Decom_loss(R_low, I_low, s_low, R_normal, I_normal, s_normal, lamda_rec=0.1, lamda_ir=0.1, lamda_is=0.1):

    s_ll = tf.reduce_mean(tf.abs(tf.multiply(R_low, I_low)-s_low))
    s_nn = tf.reduce_mean(tf.abs(tf.multiply(R_normal, I_normal)-s_normal))
    recon_loss_mutal_high = tf.reduce_min(
        tf.abs(tf.multiply(R_low, I_normal)-s_normal))
    recon_loss_mutal_low = tf.reduce_mean(
        tf.abs(tf.multiply(R_normal, I_low)-s_low))
    equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_normal))

    Ismooth_loss_low = smooth(I_low, R_low)
    Ismooth_loss_high = smooth(I_normal, R_normal)
    loss_Decom = s_ll+s_nn + 0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + \
        0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + 0.01 * equal_R_loss

    return loss_Decom

def Relight_loss(s_normal,R_low,I_delta):
    I_delta_3=tf.concat([I_delta,I_delta,I_delta],3)
    relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - s_normal))
    Ismooth_loss_delta = smooth(I_delta, R_low)

    return relight_loss + 3 * Ismooth_loss_delta

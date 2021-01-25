import numpy as np
from PIL import Image
import io
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(x, r, l, x_):
    #figure = plt.figure(figsize=(10,10))
    n = len(x)
    figure, axs = plt.subplots(
        n, 4, sharex=True, sharey=True, figsize=(20, 20))
    plt.tight_layout()
    for j in range(n):
        # Start next subplot.
        axs[j % n][0].imshow(x[j].astype(np.uint8))
        axs[j % n][1].imshow(r[j].numpy())
        axs[j % n][2].imshow(l[j,:,:,0].numpy(), 'gray')
        axs[j % n][3].imshow(x_[j].numpy())
    return figure


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

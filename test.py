# %%
# %%
import os
import tensorflow as tf
import numpy as np
from PIL import Image

from datetime import datetime
import matplotlib.pyplot as plt
from utils.datageneretor import DataGenerator
from network.loss import Decom_loss, Relight_loss
from network.decomNet import DecomNet
from network.relightNet import RelightNet
from utils.utils import plot_to_image, image_grid

def test(low_img):
    decomNet = DecomNet()
    relightNet = RelightNet()
    optm = tf.keras.optimizers.Adam(learning_rate=0.0003)

    
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optm, model=decomNet)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="checkpoint/DecomNet", max_to_keep=5)
    status1 = checkpoint.restore(manager.latest_checkpoint)

    checkpoint1 = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optm, model=relightNet)
    manager1 = tf.train.CheckpointManager(
        checkpoint1, directory="checkpoint/RelightNet", max_to_keep=5)
    status2 = checkpoint1.restore(manager1.latest_checkpoint)
    
    R,I=decomNet(low_img)
    I_delt=relightNet(R*I)
    return R,I,I_delt
def predict(tfdg):
    for imgs,_,name in tfdg:
        r,i,I_delta=test(imgs)
        for k in range(len(r)):
            tf.keras.preprocessing.image.save_img(
                f"output/enhacned_{name[k]}", r[k]*I_delta[k], file_format=None, scale=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental.set_memory_growth('GPU:0',True)
tfdg_train = DataGenerator(
    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=4,testing=True)
#tfdg_val = DataGenerator(
#    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=2)
# %%
predict(tfdg_train)

#%%
"""x,y=next(iter(tfdg_val))
#%%
R,I,I_delt=test(x)


# %%
plt.imshow(R[0])
# %%
plt.imshow(I[0],"gray")
# %%
plt.imshow(I_delt[0],"gray")
# %%
plt.imshow(R[0]*I[0])
# %%
plt.imshow(R[0]*I_delt[0])
"""
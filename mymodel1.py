# %%
import os
import tensorflow as tf
import numpy as np
from PIL import Image

from datetime import datetime
import matplotlib.pyplot as plt
# %%
from datageneretor import DataGenerator
from loss import Decom_loss, Relight_loss
from decomNet import DecomNet
from relightNet import RelightNet
from utils import plot_to_image, image_grid

#%%
def train_step_decom(decomNet, train_gen, optimizer, Decom_loss, cur_loss=0):
    for imgs_low, imgs_normal in train_gen:
        # print(imgs_low)
        with tf.GradientTape() as tape:
            R_normal, I_normal = decomNet(imgs_normal)
            R_low, I_low = decomNet(imgs_low)
            I_normal_3 = tf.concat([I_normal, I_normal, I_normal], axis=3)
            I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
            loss_ = Decom_loss(R_low, I_low_3, imgs_low,
                               R_normal, I_normal_3, imgs_normal)

        grad = tape.gradient(loss_, decomNet.trainable_variables)
        optimizer.apply_gradients(zip(grad, decomNet.trainable_variables))
        cur_loss += loss_
    return cur_loss


def val_step_decom(decomNet, val_gen, Decom_loss, val_loss=0):
    for imgs_low, imgs_normal in val_gen:

        R_normal, I_normal = decomNet(imgs_normal)
        R_low, I_low = decomNet(imgs_low)
        I_normal_3 = tf.concat([I_normal, I_normal, I_normal], 3)
        I_low_3 = tf.concat([I_low, I_low, I_low], 3)
        loss_ = Decom_loss(R_low, I_low_3, imgs_low,
                           R_normal, I_normal_3, imgs_normal)
        val_loss += loss_
    return val_loss


def train_from_generator(tfdg_train, tfdg_val, test_img, epochs):
    decomNet = DecomNet()
    optm = tf.keras.optimizers.Adam(learning_rate=0.0003)
    logdir = "logs/Decom_Net/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_file_filewriter = tf.summary.create_file_writer(logdir + "/scalar")
    img_file_filewriter = tf.summary.create_file_writer(logdir+"/image")
    # tf.profiler.experimental.start(logdir)
    # tf.debugging.experimental.enable_dump_debug_info(
    #    logdir,
    #   tensor_debug_mode="FULL_HEALTH",
    #    circular_buffer_size=-1)
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optm, model=decomNet)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="decomNet", max_to_keep=5)
    status = checkpoint.restore(manager.latest_checkpoint)
    print(status)
    for epoch in range(epochs):
        train_loss = 0
        train_loss = train_step_decom(decomNet, tfdg_train,
                                      optm, Decom_loss, train_loss)

        #val_loss = 0
        #val_loss = val_step_decom(decomNet, tfdg_val, Decom_loss, val_loss)
        tfdg_train.on_epoch_end()
        #tfdg_val.on_epoch_end()
        with loss_file_filewriter.as_default():
            tf.summary.scalar("train_loss", data=train_loss, step=epoch)
            #tf.summary.scalar("val_loss", data=val_loss, step=epoch)

        # Prepare the plot
        '''R_, L_ = decomNet(test_img)
        x_ = R_*L_
        figure = image_grid(test_img, R_, L_, x_)
        # Convert to image and log
        with img_file_filewriter.as_default():
            tf.summary.image("output", plot_to_image(figure), step=epoch)
        '''
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 2 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(
                int(checkpoint.step), save_path))
            print("loss {:1.2f}".format(train_loss))
            # decomNet.save("saved_model/v001.h5")

    # tf.profiler.experimental.stop()

    return

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%%
# tf.config.experimental.set_memory_growth('GPU:0',True)
tfdg_train = DataGenerator(
    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=4)
tfdg_val = DataGenerator(
    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=2)
x, y = next(iter(tfdg_val))
#%%
train_from_generator(tfdg_train, tfdg_val, x, 2)
'''
# %%
mod=DecomNet()
# %%
r,l=mod(x)
# %%
plt.imshow(r[0])

# %%
plt.imshow(l[0],"gray")
# %%
plt.imshow((r*l)[0])
# %%
mod.save_weights("new_weight")
#%%

#mod.load_weights("decomNet")
checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), model=mod)
manager = tf.train.CheckpointManager(
        checkpoint, directory="decomNet", max_to_keep=5)
#%%
status = checkpoint.restore(manager.latest_checkpoint)
    
#%%
# Tested with TF 2.0.0, Linux Ubuntu 18.04, Python 3.7.3, TF installed from binary

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#%%
# create model and optimizer and checkpoint
model = tf.keras.models.Sequential([tf.keras.layers.Dense(5)])
opt = tf.keras.optimizers.RMSprop()
checkpoint_dir = 'ckpts'
ckpt = tf.train.Checkpoint( model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
#%%

# train with one example
example_x = tf.constant([[1.]])
example_y = tf.constant([[1.,2.,3.,4.,5.]])
model.compile(loss="mean_squared_error", optimizer=opt)
#%%
model.fit(example_x, example_y, epochs=1)
#%%
save_path = manager.save()
ckpt.write("/tmp/ckpt")
model.save_weights("myweight")
print("Saved checkpoint: {}".format(save_path))
#%%
# ========== restart from scratch but restore from checkpoint
model = tf.keras.models.Sequential([tf.keras.layers.Dense(5)])
opt = tf.keras.optimizers.RMSprop()
checkpoint_dir = 'ckpts'

ckpt = tf.train.Checkpoint( model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
print('restoring...')
status = ckpt.restore(manager.latest_checkpoint)
# assert_consumed() fails with:
    # AssertionError: Unresolved object in checkpoint (root).opt.iter: attributes {
    #   name: "VARIABLE_VALUE"
    #   full_name: "Adam/iter"
    #   checkpoint_key: "opt/iter/.ATTRIBUTES/VARIABLE_VALUE"
status.assert_consumed()
#ckpt.read("/tmp/ckpt").assert_consumed()

# %%
nm=tf.keras.models.Sequential([tf.keras.layers.Dense(5)])
# %%
# %%
nm.load_weights("myweight")'''
# %%
import os
import tensorflow as tf
from network.decomNet import DecomNet
from network.relightNet import RelightNet
from datetime import datetime
import matplotlib.pyplot as plt
# %%
from network.loss import Decom_loss, Relight_loss



def train_step_relightnet(decomNet, relightNet, train_gen, optimizer, Decom_loss, Relight_loss, cur_loss=0):
    for imgs_low, imgs_normal in train_gen:
        # print(imgs_low)
        with tf.GradientTape() as tape:
            # R_normal, I_normal = decomNet(imgs_normal)d_pro/RetinexNet/data/LOLdataset/eval15", batch_size=d_pro/RetinexNet/data/LOLdataset/eval15", batch_size=2)2)
            #I_normal_3 = tf.concat([I_normal, I_normal, I_normal],axis=3)

            R_low, I_low = decomNet(imgs_low)
            I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
            #loss_1 = Decom_loss(R_low, I_low_3, imgs_low,R_normal, I_normal_3, imgs_normal)
            s_pred = R_low*I_low_3
            I_delta = relightNet(s_pred)
            loss_2 = Relight_loss(imgs_normal, R_low, I_delta)  # +loss_1
        grad = tape.gradient(
            loss_2, decomNet.trainable_variables+relightNet.trainable_variables)
        optimizer.apply_gradients(
            zip(grad, decomNet.trainable_variables+relightNet.trainable_variables))
        cur_loss += loss_2
    return cur_loss


def val_step_relightnet(decomNet, relightNet, val_gen, decomLoss, relightLoss, val_loss=0):
    for imgs_low, imgs_normal in val_gen:
        R_normal, I_normal = decomNet(imgs_normal)
        R_low, I_low = decomNet(imgs_low)
        I_normal_3 = tf.concat([I_normal, I_normal, I_normal], 3)
        I_low_3 = tf.concat([I_low, I_low, I_low], 3)
        loss_1 = Decom_loss(R_low, I_low_3, imgs_low,
                            R_normal, I_normal_3, imgs_normal)
        I_delta = relightNet(R_low*I_low_3)
        loss_2 = relightLoss(imgs_normal, R_low, I_delta)  # +loss_1
        val_loss += loss_1+loss_2
    return val_loss


def train_from_generator_relight(tfdg_train, tfdg_val, test_img, epochs,lr=0.001):
    decomNet = DecomNet()
    relightNet = RelightNet()
    #logdir = "logs/DecomNet/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/RelightNet/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    optm = tf.keras.optimizers.Adam(learning_rate=lr)
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
        checkpoint, directory="checkpoint/DecomNet", max_to_keep=5)
    #status = checkpoint.restore(manager.latest_checkpoint)

    checkpoint1 = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optm, model=relightNet)
    manager1 = tf.train.CheckpointManager(
        checkpoint1, directory="checkpoint/RelightNet", max_to_keep=5)
    #status = checkpoint1.restore(manager1.latest_checkpoint)
    for epoch in range(epochs):
        train_loss = 0
        train_loss = train_step_relightnet(decomNet, relightNet, tfdg_train,
                                           optm, Decom_loss, Relight_loss, train_loss)

        #val_loss = 0
        #val_loss = val_step_relightnet(decomNet, relightNet, tfdg_val, Decom_loss, Relight_loss, val_loss)

        '''R_,I_=decomNet(test_img)
        I_delta=relightNet(R_*tf.concat([I_,I_,I_],3))
        images_R = np.reshape(R_[0:3].numpy(), (-1,600,400,3))
        images_L=np.reshape(I_[0:3].numpy(), (-1,600,400,1))
        #image=np.reshape(imgs_low[0:3].astype(np.uint8),(-1,600,400,3))
        tf.summary.image('Img',test_img.astype(np.uint8),max_outputs=10,step=epoch)
        tf.summary.image("R", images_R, max_outputs=10, step=epoch)
        tf.summary.image("L",images_L,max_outputs=10,step=epoch)
        tf.summary.scalar('loss', data=loss, step=epoch)
        '''
        tfdg_train.on_epoch_end()
        # tfdg_val.on_epoch_end()
        with loss_file_filewriter.as_default():
            tf.summary.scalar("train_loss", data=train_loss, step=epoch)
            #tf.summary.scalar("val_loss", data=val_loss, step=epoch)
        print(epoch, train_loss)
        # Prepare the plot
        '''R_, L_ = decomNet(test_img)
        x_ = R_*L_
        I_delta = relightNet(x_)

        figure = image_grid(test_img, R_, L_, x_)
        # Convert to image and log
        with img_file_filewriter.as_default():
            tf.summary.image("output", plot_to_image(figure), step=epoch)
        '''
        checkpoint.step.assign_add(1)
        checkpoint1.step.assign_add(1)
        if int(checkpoint.step) % 2 == 0:
            save_path = manager.save()
            save_path = manager1.save()

            print("Saved checkpoint for step {} and {}: {}".format(
                int(checkpoint.step), int(checkpoint1.step), save_path))
            print("loss {:1.2f}".format(train_loss))
            # decomNet.save("saved_model/v001.h5")

    # tf.profiler.experimental.stop()

    return


# %%
if __name__ == '__main__':

    from utils.datageneretor import DataGenerator

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # tf.config.experimental.set_memory_growth('GPU:0',True)
    tfdg_train = DataGenerator(
        "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/our485", batch_size=4)
    #tfdg_val = DataGenerator(    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=2)
    #x, y = next(iter(tfdg_val))
    train_from_generator_relight(tfdg_train, None, None, 2)
# %%

# %%



    #tfdg_val = DataGenerator(    "/home/akhilesh/Documents/sem3/video/mid_pro/RetinexNet/data/LOLdataset/eval15", batch_size=2)
    #x, y = next(iter(tfdg_val))

# %%

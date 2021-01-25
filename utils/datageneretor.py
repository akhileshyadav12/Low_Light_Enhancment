# %%
import numpy as np
from tensorflow import keras
import os
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, path, batch_size=4, dim=(400, 600), n_channels=3, shuffle=True,testing=False):
        'Initialization'
        print(path)
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = os.listdir(os.path.join(path, "high"))
        self.id = range(len(self.list_ids))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.testing=testing
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.id) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.id[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))
        names=[None]*self.batch_size
        # Generate data
        for i, ID in enumerate(IDs_temp):
            # Store sample
            #X[i,] = np.asarray(Image.open(os.path.join(self.path,"low/"+self.list_ids[ID])),dtype=np.float32)
            #Y[i,] = np.asarray(Image.open(os.path.join(self.path,"high/"+self.list_ids[ID])),dtype=np.float32)
            X[i, ] = np.asarray(Image.open(os.path.join(
                self.path, "low/"+self.list_ids[ID])), dtype=np.float32)
            Y[i, ] = np.asarray(Image.open(os.path.join(
                self.path, "high/"+self.list_ids[ID])), dtype=np.float32)
            names[i]=self.list_ids[ID]
        if self.testing:
            return X,Y,names

        return X, Y


# %%
enumerate# %%
if __name__ == "__main__":
    tfdg = DataGenerator("data/LOLdataset/our485")

    print(next(iter(tfdg)))
# %%

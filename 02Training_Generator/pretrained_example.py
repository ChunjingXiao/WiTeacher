#https://blog.csdn.net/weixin_41943311/article/details/100539707
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import glob
import random
from scipy import io
import matplotlib
PREFIX = 'M'
import matplotlib.pyplot as plt
TIMES_LOOP = 200

def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    # Load pre-trained network.
    Model = 'cache/network-snapshot-010000.pkl'
    model_file = glob.glob(Model)
    if len(model_file) == 1:
        model_file = open(model_file[0], "rb")
    else:
        raise Exception('Failed to find the model')

    _G, _D, Gs = pickle.load(model_file)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run. Generator的瞬时快照。主要用于恢复以前的训练运行。
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run. Discriminator的瞬时快照。主要用于恢复以前的训练运行。
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot. Generator的长期平均值。产生比瞬时快照更高质量的结果。
    # Print network details.
    # onehot1 = np.zeros((1, 10), dtype=np.float32)
    # onehot1[np.arange(1), 7] = 1.0
    image_data = []
    label = []
    feature = io.loadmat('datasets/signfi/signfi_data_.mat')
    data_ = feature['signfi_data_']

    data = np.zeros((500, 200, 30, 3), dtype=np.float64)
    for i in range(500): #i=0:249
        data[i, :, :, :] = data_[:, :, :, i]
    del data_

    data = np.transpose(data, axes=[0, 3, 1, 2])
    datapad = np.pad(data, [(0, 0), (0, 0), (0, 56), (0, 226)], 'constant', constant_values=0)

    mats = datapad.astype(np.float32)

    feature1 = io.loadmat('datasets/signfi/signfi_label_.mat')
    label_1 = feature1['signfi_label_']

    label_ = label_1.T
    del label_1

    labels = np.zeros(data.shape[0], dtype=np.uint8)
    for i in range(data.shape[0]):
        labels[i] = label_[0][i]

    # Print network details.
    Gs.print_layers()
    lbes = [1]
    a3 = 1501
    for i, lbe in enumerate(lbes):

        lables_index = np.array(np.where(labels == lbe))
        onehot1 = np.zeros((1, 1), dtype=np.float32)
        onehot1[np.arange(1), i] = 1.0
        for j in range(10000):
            # Pick latent vector.
            SEED = random.randint(0, 10000)
            rnd = np.random.RandomState(SEED)
            latents = rnd.randn(1, Gs.input_shape[1])

            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, onehot1, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
            # images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)


            #Generate and Save image.
            a1 = images[0, :200, -30:, :] #(200,30,3)
            matplotlib.image.imsave('./datasets/signfi_gen/sign_' + str(lbe) + '/' + str(a3) + '_sign_' + str(lbe) + '.png', a1.astype(np.uint8))
            a3 = a3 + 1
            del a1



if __name__ == "__main__":
    main()
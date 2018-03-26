from skimage.transform import resize

import tensorflow as tf
import numpy as np

import re
import requests
import os
import tarfile


CHECKPOINT_FILE = 'vgg_19.ckpt'


def get_output_tensor_of_layer(layer_name):
    if re.match(r'conv\d_\d', layer_name) is not None:
        layer_name += '/Relu'
    elif re.match(r'pool\d', layer_name) is not None:
        layer_name += '/AvgPool'

    op = next((op for op in tf.get_default_graph().get_operations() if layer_name in op.name), None)

    if op is None:
        raise ValueError('Invalid layer name: %s' % layer_name)

    return op.outputs[0]


def get_bounded_shape(image, max_size):
    """get a shape with largest dimension <= max_size while preserving aspect ratio"""
    h, w, _ = image.shape
    aspect_ratio = w / h
    if h >= w:
        h = min(max_size, h)
        image_shape = (h, int(h * aspect_ratio))
    else:
        w = min(max_size, w)
        image_shape = (int(w / aspect_ratio), w)
    return image_shape


def preprocess_image(image, image_shape):
    if image.shape != image_shape:
        image = resize(image, image_shape, order=3, mode='constant', preserve_range=True)
    return np.expand_dims(image.astype(np.float32), axis=0)


def postprocess_image(image):
    image = np.squeeze(np.clip(image, 0.0, 255.0).astype(np.uint8))
    return image


def maybe_download_pretrained_model():
    temp_file = 'weights.tar.gz'
    url = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
    if not os.path.exists(CHECKPOINT_FILE):
        print('Downloading pretrained model...', end='', flush=True)
        response = requests.get(url)
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        tar = tarfile.open(temp_file, 'r:gz')
        tar.extractall()
        tar.close()
        os.remove(temp_file)
        print('done.')


def maybe_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        for f in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, f))

from skimage import io
from transfer_cnn import TransferCNN

import tensorflow as tf
import numpy as np

import argparse
import utils
import os

parser = argparse.ArgumentParser(description='Transfer style')

parser.add_argument('--hparams', type=str, help='Comma seperated list of \'name=value\' pairs', default='')
parser.add_argument('--content_image', type=argparse.FileType('r'))
parser.add_argument('--style_image', type=argparse.FileType('r'))
parser.add_argument('--generated_image', type=argparse.FileType('w'), default='generated_image.jpg')
parser.add_argument('--should_log', action='store_true')
parser.add_argument('--logs_dir', default='./logs')

args = parser.parse_args()

hparams = tf.contrib.training.HParams(
    learning_rate=1.0,
    content_layer_names=['conv4_2'],
    style_layer_names=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    num_epochs=2000,
    content_weight=5e2,
    style_weight=1,
    variation_weight=0.2,
    max_size=750,
    noise_ratio=0.6
)

hparams.parse(args.hparams)

content_image = io.imread(args.content_image.name)
bounded_shape = utils.get_bounded_shape(content_image, hparams.max_size)
style_image = io.imread(args.style_image.name)

content_image = utils.preprocess_image(content_image, bounded_shape)
style_image = utils.preprocess_image(style_image, bounded_shape)

graph = tf.Graph()

with tf.Session(graph=graph) as session:
    transferCNN = TransferCNN(
        session,
        content_layer_names=hparams.content_layer_names,
        style_layer_names=hparams.style_layer_names,
        image_shape=bounded_shape,
        content_weight=hparams.content_weight,
        style_weight=hparams.style_weight,
        variation_weight=hparams.variation_weight,
        learning_rate=hparams.learning_rate
    )

    transferCNN.x = content_image
    # ground truth content comes from content image
    transferCNN.gt_contents = session.run(transferCNN.contents())

    transferCNN.x = style_image
    # ground truth style comes from style image
    transferCNN.gt_styles = session.run(transferCNN.styles())

    transferCNN.build()

    summary_writer = None
    if args.should_log:
        current_logs_dir = os.path.join(args.logs_dir, hparams.to_json())
        utils.maybe_mkdir(current_logs_dir)
        summary_writer = tf.summary.FileWriter(current_logs_dir, graph)

    # initialize
    transferCNN.x = (1 - hparams.noise_ratio) * content_image +\
        hparams.noise_ratio * np.random.uniform(-20.0, 20.0, content_image.shape)

    for i in range(hparams.num_epochs):
        _, loss = session.run([transferCNN.optimize(), transferCNN.loss()])

        if i % 50 == 0 or i == hparams.num_epochs - 1:
            print("step: %i, loss: %.4e" % (i + 1, loss))
            if args.should_log:
                summary_writer.add_summary(session.run(transferCNN.summaries()), i + 1)
    if args.should_log:
        summary_writer.close()

    generated_image = utils.postprocess_image(session.run(transferCNN.x))
    io.imsave(args.generated_image.name, generated_image)

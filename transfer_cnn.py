from functools import wraps

import tensorflow as tf

import utils

conv = tf.layers.conv2d
pool = tf.layers.average_pooling2d


def evaluated_op(func):
    cache = '_' + func.__name__

    @wraps(func)
    def evaluated_cached_op(self):
        if not hasattr(self, cache):
            setattr(self, cache, func(self))
        return getattr(self, cache)

    return evaluated_cached_op


class TransferCNN:
    def __init__(self,
                 session,
                 content_layer_names=['conv4_2'],
                 style_layer_names=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                 image_shape=[224, 224],
                 content_weight=1.0,
                 style_weight=1e2,
                 variation_weight=1e-3,
                 learning_rate=0.1):

        self._content_layer_names = content_layer_names
        self._style_layer_names = style_layer_names
        self._session = session
        self._content_weight = content_weight
        self._style_weight = style_weight
        self._variation_weight = variation_weight
        self._learning_rate = learning_rate

        self.gt_contents = None
        self.gt_styles = None

        # VGG-19 with average pooling
        with tf.variable_scope('transfer_cnn', reuse=tf.AUTO_REUSE):
            net = self._x = tf.get_variable(name='x', shape=[1, image_shape[0], image_shape[1], 3])
            mean = tf.get_variable('mean_rgb', 3)
            net = net - tf.reshape(mean, [1, 1, 1, 3])

            with tf.variable_scope('conv1'):
                net = conv(net, 64, [3, 3], padding='same', activation=tf.nn.relu, name='conv1_1')
                net = conv(net, 64, [3, 3], padding='same', activation=tf.nn.relu, name='conv1_2')

            net = pool(net, [2, 2], [2, 2], name='pool1')

            with tf.variable_scope('conv2'):
                net = conv(net, 128, [3, 3], padding='same', activation=tf.nn.relu, name='conv2_1')
                net = conv(net, 128, [3, 3], padding='same', activation=tf.nn.relu, name='conv2_2')

            net = pool(net, [2, 2], [2, 2], name='pool2')

            with tf.variable_scope('conv3'):
                net = conv(net, 256, [3, 3], padding='same', activation=tf.nn.relu, name='conv3_1')
                net = conv(net, 256, [3, 3], padding='same', activation=tf.nn.relu, name='conv3_2')
                net = conv(net, 256, [3, 3], padding='same', activation=tf.nn.relu, name='conv3_3')
                net = conv(net, 256, [3, 3], padding='same', activation=tf.nn.relu, name='conv3_4')

            net = pool(net, [2, 2], [2, 2], name='pool3')

            with tf.variable_scope('conv4'):
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv4_1')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv4_2')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv4_3')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv4_4')

            net = pool(net, [2, 2], [2, 2], name='pool4')

            with tf.variable_scope('conv5'):
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv5_1')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv5_2')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv5_3')
                net = conv(net, 512, [3, 3], padding='same', activation=tf.nn.relu, name='conv5_4')

            net = pool(net, [2, 2], [2, 2], name='pool5')

            self._initialize_parameters()

    @property
    @evaluated_op
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        # set input `x` to `value` by running the assign op
        assign_op = self._x.assign(value)
        self._session.run(assign_op)

    @evaluated_op
    def contents(self):
        # content tensors
        return list(map(utils.get_output_tensor_of_layer, self._content_layer_names))

    @evaluated_op
    def styles(self):
        # style tensors
        return list(
            map(lambda x: TransferCNN._gram_matrix(utils.get_output_tensor_of_layer(x)), self._style_layer_names))

    @evaluated_op
    def loss(self):
        with tf.name_scope('loss'):
            loss = 0.0

            # content loss
            if self.gt_contents is None:
                raise ValueError('gt_contents is None')
            content_loss = tf.reduce_mean(tf.stack(list(
                map(lambda x: tf.losses.mean_squared_error(*x, loss_collection=None),
                    zip(self.gt_contents, self.contents())))), name='content_loss')
            tf.losses.add_loss(content_loss)
            loss = tf.add(loss, self._content_weight * content_loss)

            # style loss
            if self.gt_styles is None:
                raise ValueError('gt_styles is None')
            style_loss = tf.reduce_mean(tf.stack(list(
                map(lambda x: tf.losses.mean_squared_error(*x, loss_collection=None),
                    zip(self.gt_styles, self.styles())))), name='style_loss')
            tf.losses.add_loss(style_loss)
            loss = tf.add(loss, self._style_weight * style_loss)

            # total variation loss
            variation_loss = tf.image.total_variation(self._x)
            variation_loss = tf.identity(variation_loss[0], name='variation_loss')
            tf.losses.add_loss(variation_loss)

            loss = tf.add(loss, self._variation_weight * variation_loss, name='total_loss')
            tf.losses.add_loss(loss)

            return loss

    @evaluated_op
    def optimize(self):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            update_op = optimizer.minimize(self.loss(), var_list=[self._x])
            self._session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'optimizer')))
            return update_op

    @evaluated_op
    def summaries(self):
        with tf.name_scope('summaries'):
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                tf.summary.scalar(loss.name[:-2], loss)
            tf.summary.image('generated_image', tf.clip_by_value(self.x, 0.0, 255.0))
            return tf.summary.merge_all()

    def build(self):
        # Build rest of the graph for synthesizing a new image
        self.loss()
        self.optimize()
        self.summaries()

    def _initialize_parameters(self):
        utils.maybe_download_pretrained_model()

        reader = tf.train.NewCheckpointReader(utils.CHECKPOINT_FILE)
        var_to_shape_map = reader.get_variable_to_shape_map()
        var_dict = dict()

        for name_in_ckpt in var_to_shape_map:
            var_name = name_in_ckpt.split('/', 1)[-1].replace('weights', 'kernel').replace('biases', 'bias')
            if 'conv' in var_name:
                var_dict[name_in_ckpt] = tf.get_variable(var_name)
            elif 'mean_rgb' in var_name:
                var_dict[name_in_ckpt] = tf.get_variable(var_name)

        saver = tf.train.Saver(var_dict)
        saver.restore(self._session, utils.CHECKPOINT_FILE)

        print('Initialized TransferCNN weights.')

    @staticmethod
    def _gram_matrix(x):
        with tf.control_dependencies([tf.assert_equal(len(x.shape), 4),
                                      tf.assert_equal(x.shape[0], 1)]):
            _, h, w, c = x.shape.as_list()
            x = tf.reshape(x, [h * w, c])
            gram = tf.matmul(x, x, transpose_a=True) / (h * w)
            return gram

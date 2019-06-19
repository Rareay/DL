from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def model_inputs(image_width, image_height, image_channels, z_dim):
    ## Real imag
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')

    ## input z

    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    ## Learning rate
    learning_rate = tf.placeholder(tf.float32, name='lr')

    return inputs_real, inputs_z, learning_rate


def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function

    ## scope here

    with tf.variable_scope('discriminator', reuse=reuse):
        alpha = 0.2  ### leak relu coeff

        # drop out probability
        keep_prob = 0.8

        # input layer 28 * 28 * color channel
        x1 = tf.layers.conv2d(images, 128, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        ## No batch norm here
        ## leak relu here / alpha = 0.2
        relu1 = tf.maximum(alpha * x1, x1)
        # applied drop out here
        drop1 = tf.nn.dropout(relu1, keep_prob=keep_prob)
        # 14 * 14 * 128

        # Layer 2
        x2 = tf.layers.conv2d(drop1, 256, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        ## employ batch norm here
        bn2 = tf.layers.batch_normalization(x2, training=True)
        ## leak relu
        relu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.nn.dropout(relu2, keep_prob=keep_prob)

        # 7 * 7 * 256

        # Layer3
        x3 = tf.layers.conv2d(drop2, 512, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        drop3 = tf.nn.dropout(relu3, keep_prob=keep_prob)
        # 4 * 4 * 512

        # Output
        # Flatten
        flatten = tf.reshape(relu3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        # activation
        out = tf.nn.sigmoid(logits)

    return out, logits


def bottleneck(inputs, filter_out, is_training):
    momentum = 0.9
    if inputs.shape[3] == filter_out:
        residual = tf.layers.conv2d_transpose(inputs, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual, training=is_training)
        residual = tf.nn.relu(tf.contrib.layers.batch_norm(residual, is_training=is_training, decay=momentum))
        residual = tf.layers.conv2d_transpose(residual, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual, training=is_training)
    else:
        residual = tf.layers.conv2d_transpose(inputs, kernel_size=3, filters=filter_out, strides=2, padding='same')
        residual = tf.layers.batch_normalization(residual, training=is_training)
        residual = tf.nn.relu(tf.contrib.layers.batch_norm(residual, is_training=is_training, decay=momentum))
        residual = tf.layers.conv2d_transpose(residual, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual, training=is_training)
        inputs = tf.layers.conv2d_transpose(inputs, kernel_size=1, filters=filter_out, strides=2, padding='same')
    out = tf.add(inputs, residual)
    return out


def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function

    with tf.variable_scope('generator', reuse=not is_train):
        # First Fully connect layer
        print("**************************z =", z)
        x0 = tf.layers.dense(z, 4 * 4 * 512)
        # Reshape
        x0 = tf.reshape(x0, (-1, 4, 4, 512))
        # Use the batch norm
        bn0 = tf.layers.batch_normalization(x0, training=is_train)
        # Leak relu
        relu0 = tf.nn.relu(bn0)
        # 4 * 4 * 512

        # Conv transpose here
        x1 = tf.layers.conv2d_transpose(relu0, 256, 4, strides=1, padding='valid')
        bn1 = tf.layers.batch_normalization(x1, training=is_train)
        relu1 = tf.nn.relu(bn1)
        # 7 * 7 * 256

        x2 = tf.layers.conv2d_transpose(relu1, 128, 3, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=is_train)
        relu2 = tf.nn.relu(bn2)
        # 14 * 14 * 128

        # Last cov
        logits = tf.layers.conv2d_transpose(relu2, out_channel_dim, 3, strides=2, padding='same')
        ## without batch norm here
        out = tf.tanh(logits)

        return out


def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    print("<<<<<<<<<<<input_z.shape", input_z.shape)
    g_model = generator(input_z, out_channel_dim, is_train=True)

    print("<<<<<<<<<<<input_real.shape", input_real.shape)
    d_model_real, d_logits_real = discriminator(input_real, reuse=False)

    print("<<<<<<<<<<<g_model.shape", g_model.shape)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    print("<<<<<<<<<<<", d_model_fake.shape, d_logits_fake.shape)

    ## add smooth here

    smooth = 0.1
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth)))

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss, g_model


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')
def get_batches(batch_size):
    mnist_dataset = mnist.train.images[0:40000]
    mnist_dataset = np.array(mnist_dataset)
    print(">>>>>>>>>>>> get_batches.shape=",mnist_dataset.shape)
    mnist_dataset = np.reshape(mnist_dataset,(10000,4,784))
    #mnist_dataset = mnist.train.next_batch(batch_size=batch_size)  # 样本和标签
    print(">>>>>>>>>>>> get_batches.shape=",mnist_dataset.shape)
    mnist_dataset = (mnist_dataset - 0.5)/2
    return mnist_dataset


# 定义一个辅助函数，用于将多张图片以网格状拼在一起显示
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    #img_n = images.shape[3]
    n_plots = int(np.sqrt(images.shape[0]))
    m = np.ones((images.shape[1] * n_plots, images.shape[2] * n_plots, 1)) * 0.5
    #m = np.ones((images.shape[1] * n_plots, images.shape[2] * n_plots, img_n)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img
                #m[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w, 0:img_n] = this_img
    return m


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    losses = []
    samples = []

    # add
    print('>>>>>>>>>>>>>>>>>>>>>>>>>',data_shape[1])
    print('>>>>>>>>>>>>>>>>>>>>>>>>>',data_shape[2])
    print('>>>>>>>>>>>>>>>>>>>>>>>>>',data_shape[3])

    input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>',input_real.shape, input_z.shape, data_shape[-1])

    d_loss, g_loss, g = model_loss(input_real, input_z, data_shape[-1])
    print('>>>>>>>>>>>>>>>>>>>>>>>>>')

    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>')

    steps = 10000

    batch_images = get_batches(batch_size)[0]
    print("***************************",batch_images.shape)
    temp = batch_images.reshape(4,28,28,1)
    temp = montage(temp)
    temp = temp.reshape(28*2,28*2)
    plt.axis('off')
    plt.imshow(temp, cmap='gray')
    plt.show()
    #batch_images = get_batches(batch_size)[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            #for batch_images in get_batches:
            #for batch_images in get_batches(batch_size):
            for i in range(10000):
                # TODO: Train Model
                steps -= 1
                print("第%d次训练" %(steps))

                # Reshape the image and pass to Discriminator
                batch_images = batch_images.reshape(batch_size,
                                                    data_shape[1],
                                                    data_shape[2],
                                                    data_shape[3])
                # Rescale the data to -1 and 1
                batch_images = (batch_images - 0.5)*2
                #batch_images = batch_images * 2


                # Sample the noise
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                ## Run optimizer
                _ = sess.run(d_opt, feed_dict={input_real: batch_images,
                                               input_z: batch_z,
                                               lr: learning_rate
                                               })
                _ = sess.run(g_opt, feed_dict={input_real: batch_images,
                                               input_z: batch_z,
                                               lr: learning_rate})

                if steps % 10 == 0:
                    train_loss_d = d_loss.eval({input_real: batch_images, input_z: batch_z})
                    train_loss_g = g_loss.eval({input_real: batch_images, input_z: batch_z})

                    losses.append((train_loss_d, train_loss_g))

                    print("Epoch {}/{}...".format(epoch_i + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % 10 == 0:
                    print("显示图片")
                    gen = sess.run(g, feed_dict={input_z:batch_z,})
                    gen = (gen + 1) / 2
                    print(">>>>>>>>>>>>>>>>>> g_model.shape =", gen.shape)
                    imgs = [img[:, :] for img in gen]
                    gen_imgs = montage(imgs)
                    gen_imgs = gen_imgs.reshape(28*2,28*2)
                    plt.axis('off')
                    print(">>>>>>>>>>>>>>>>>> gen_imgs.shape =", gen_imgs.shape)
                    plt.imshow(gen_imgs, cmap='gray')
                    plt.savefig(os.path.join('samples/', 'sample_%d.jpg' % steps))
                    #plt.show()
                    samples.append(gen_imgs)

batch_size = 4
z_dim = 100
learning_rate = 0.001
beta1 = 0.5
epochs = 20

#mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
data_shape = np.array([0,28,28,1,1])
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, None)
    #train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
    #      mnist_dataset.shape, mnist_dataset.image_mode)

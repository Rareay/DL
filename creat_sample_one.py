# 导入库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import imageio  # 读取图片
import cv2 as cv

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

# 获取数据
class DATA():
    def __init__(self, samples_path, size, batch_size=1):
        self.__batch_size = batch_size
        self.__batch_count = 0
        self.cycel = 0
        if not os.listdir(samples_path):
            print("文件夹为空，获取数据失败！")
        for one_sample in os.listdir(samples_path):
            sample = samples_path + '/' + one_sample
            img = Image.open(sample)
            # img = img.resize((32, 32), Image.ANTIALIAS)
            img = img.resize((size[0], size[1]), Image.ANTIALIAS)
            pic = np.array(img)
            picture = pic.flatten()
            try:
                self.__data = np.vstack([self.__data, picture])
            except Exception:
                self.__data = picture
                if pic.shape[2] != size[2]:
                    print("样本图片通道数与指定通道数不一致！")
                    return
        print("从%s中成功获取数据！" % (samples_path))

    def get(self):
        if self.__batch_size*(self.__batch_count+1) > self.__data.shape[0]:
            self.__batch_count = 0
            self.cycel += 1
        output = self.__data[self.__batch_size * self.__batch_count:self.__batch_size * (self.__batch_count + 1)]
        self.__batch_count += 1
        return output


class GAN():
    OUTPUT_DIR = 'samples'  # 输出文件
    TRAIN_NUM = 600

    def __init__(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.mkdir(self.OUTPUT_DIR)
        self.train_num = tf.Variable(0)
        self.train_num = tf.assign(self.train_num, self.train_num+1, name='train_num')
        self.saver = tf.train.Saver() # 需要放在函数外面

    def __lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def __sigmoid_cross_entropy_with_logits(self, x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    # 判别器
    def discriminator(self, image, reuse=None):
        momentum = 0.9
        with tf.variable_scope('discriminator', reuse=reuse):  # 划分区域
            h0 = self.__lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))  # 输入一张图片

            h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
            h1 = self.__lrelu(
                tf.contrib.layers.batch_norm(h1, is_training=self.is_training, decay=momentum))  # BN在激活函数之前

            h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
            h2 = self.__lrelu(tf.contrib.layers.batch_norm(h2, is_training=self.is_training, decay=momentum))

            h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
            h3 = self.__lrelu(tf.contrib.layers.batch_norm(h3, is_training=self.is_training, decay=momentum))

            h4 = tf.contrib.layers.flatten(h3)  # 全连接前的处理，第一维度，二三四维度相乘为一个向量
            h4 = tf.layers.dense(h4, units=1)  # 全连接层
            return tf.nn.sigmoid(h4), h4

    # 生成器
    def generator(self, z):
        momentum = 0.9
        with tf.variable_scope('generator', reuse=None):
            d = 3
            h0 = tf.layers.dense(z, units=d * d * 512)
            h0 = tf.reshape(h0, shape=[-1, d, d, 512])
            h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=self.is_training, decay=momentum))

            h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
            h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=self.is_training, decay=momentum))

            h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
            h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=self.is_training, decay=momentum))

            h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
            h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=self.is_training, decay=momentum))

            h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=1, padding='valid',
                                            activation=tf.nn.tanh, name='g')
            return h4

    # 优化器
    def optimizer(self):
        # 定义损失函数，虽然是两个部分，但是参数是共享的
        self.g = self.generator(self.noise)
        self.d_real, self.d_real_logits = self.discriminator(self.X)  # 真实样本
        self.d_fake, self.d_fake_logits = self.discriminator(self.g, reuse=True)  # 参数共享

        self.vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        self.loss_d_real = tf.reduce_mean(
            self.__sigmoid_cross_entropy_with_logits(self.d_real_logits, tf.ones_like(self.d_real)))
        self.loss_d_fake = tf.reduce_mean(
            self.__sigmoid_cross_entropy_with_logits(self.d_fake_logits, tf.zeros_like(self.d_fake)))  # 真实数据和假的数据
        self.loss_g = tf.reduce_mean(
            self.__sigmoid_cross_entropy_with_logits(self.d_fake_logits, tf.ones_like(self.d_fake)))  # 生成器的损失函数
        self.loss_d = self.loss_d_real + self.loss_d_fake  # 判别器的损失函数

        # 定义优化函数注意损失函数需要和可调参数对应上
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 先执行BN参数更新，根据每一批的均值和方差去估计整个数据的均值和方差
            self.optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss_d,
                                                                                                var_list=self.vars_d)
            self.optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss_g,
                                                                                                var_list=self.vars_g)

    # 定义一个辅助函数，用于将多张图片以网格状拼在一起显示
    def montage(self, images):
        if isinstance(images, list):
            images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        img_n = images.shape[3]
        n_plots = int(np.sqrt(images.shape[0]))
        m = np.ones((images.shape[1] * n_plots, images.shape[2] * n_plots, img_n)) * 0.5
        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    m[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w, 0:img_n] = this_img
        if m.shape[2] == 1:
            m = m.reshape(m.shape[0], m.shape[1])
        return m

    # 训练
    def train(self, sample_path, sample_size, batch_size, z_dim, train_num=100):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, sample_size[0], sample_size[1], sample_size[2]], name='X')
        self.noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.optimizer()
        samples = DATA(samples_path=sample_path, size=sample_size, batch_size=batch_size)
        with tf.Session() as sess:
            try:
                MODE_META = self.OUTPUT_DIR + '/model/my-model.meta'
                print(MODE_META)
                MODE_DIR = self.OUTPUT_DIR + '/model'
                print(MODE_DIR)
                saver2 = tf.train.import_meta_graph(MODE_META)
                saver2.restore(sess, tf.train.latest_checkpoint(MODE_DIR))
                print("加载模型...")
            except Exception:
                print("加载模型出错!")
                print("重新开始训练模型...")
                sess.run(tf.global_variables_initializer())

            sess.run(tf.global_variables_initializer())
            z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
            loss = {'d': [], 'g': []}

            i = sess.run(self.train_num)
            while i < train_num:
                print("第%d次训练" % (i))
                n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
                batch = (samples.get()/255 - 0.5) * 2
                batch = np.reshape(batch, [-1, sample_size[0], sample_size[1] , sample_size[2]])
                batch_show = (self.montage(batch) + 1)/2
                plt.axis('off')
                plt.imshow(batch_show, cmap='gray')
                #plt.savefig(os.path.join(self.OUTPUT_DIR, 'sample.jpg'))
                #plt.show()
                batch = (batch - 0.5) * 2
                d_ls, g_ls = sess.run([self.loss_d, self.loss_g],
                                      feed_dict={self.X: batch, self.noise: n, self.is_training: True})  # 训练中
                loss['d'].append(d_ls)
                loss['g'].append(g_ls)

                sess.run(self.optimizer_d, feed_dict={self.X: batch, self.noise: n, self.is_training: True})  # 优化
                sess.run(self.optimizer_g, feed_dict={self.X: batch, self.noise: n, self.is_training: True})
                sess.run(self.optimizer_g, feed_dict={self.X: batch, self.noise: n, self.is_training: True})

                if i % 10 == 0:
                    print(i, d_ls, g_ls)  # 输出损失函数
                    gen_imgs = sess.run(self.g,
                                        feed_dict={self.noise: z_samples, self.is_training: False})  # 用生成器生成图片，噪音保持不变
                    gen_imgs = (gen_imgs + 1) / 2
                    gen_imgs = self.montage(gen_imgs)
                    plt.axis('off')
                    plt.imshow(gen_imgs, cmap='gray')
                    plt.savefig(os.path.join(self.OUTPUT_DIR, 'sample_%d.jpg' % i))
                    # plt.show()
                if i % 10 == 0:
                    MODE_SAVE = self.OUTPUT_DIR + '/model/my-model'
                    self.saver.save(sess, MODE_SAVE)
                    #saver.save(sess, "model/my-model")
                sess.run(self.train_num)
                i += 1


if __name__ == '__main__':
    gan = GAN()
    gan.train(sample_path='./picture', sample_size=[28,28,3], batch_size=4, z_dim=20, train_num=1000)

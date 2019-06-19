# 导入库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio  # 读取图片

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

batch_size = 4  # 迭代样本数
z_dim = 100  # 噪音的维度100
OUTPUT_DIR = 'samples'  # 输出文件
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')



from PIL import Image
def get_pic():
    for pic in os.listdir(r'./picture'):
        pic_path = './picture/' + pic
        img = Image.open(pic_path)
        img = img.resize((28, 28), Image.ANTIALIAS)
        picture = np.array(img)
        picture = picture.flatten()
        try:
            data = np.vstack([data, picture])
            #data = np.stack((data, picture), axis=0)
        except Exception:
            data = picture
    return data


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

# 判别器
def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):  # 划分区域
        print(">>>>image.shape =",image.shape)
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))  # 输入一张图片
        print(">>>>h0.shape =",h0)

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))  # BN在激活函数之前
        print(">>>>h1.shape =",h1)

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        print(">>>>h2.shape =",h2)

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        print(">>>>h3.shape =",h3)

        h4 = tf.contrib.layers.flatten(h3)  # 全连接前的处理，第一维度，二三四维度相乘为一个向量
        print(">>>>h4.shape =",h4)
        h4 = tf.layers.dense(h4, units=1)  # 全连接层
        print(">>>>h4.shape =",h4)
        return tf.nn.sigmoid(h4), h4


# 生成器
def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 3
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))
        print(">>>>>> h0.shape =", h0)

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))
        print(">>>>>> h1.shape =", h1)

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        print(">>>>>> h2.shape =", h2)

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        print(">>>>>> h3.shape =", h3)

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=1, padding='valid', activation=tf.nn.tanh,
                                        name='g')
        print(">>>>>> h4.shape =", h4)
        return h4


# 定义损失函数，虽然是两个部分，但是参数是共享的
g = generator(noise)
d_real, d_real_logits = discriminator(X)  # 真实样本
d_fake, d_fake_logits = discriminator(g, reuse=True)  # 参数共享

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))  # 真实数据和假的数据
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))  # 生成器的损失函数
loss_d = loss_d_real + loss_d_fake  # 判别器的损失函数

# 定义优化函数注意损失函数需要和可调参数对应上
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):  # 先执行BN参数更新，根据每一批的均值和方差去估计整个数据的均值和方差
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)


# 定义一个辅助函数，用于将多张图片以网格状拼在一起显示
def montage(images):
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
    return m


# 开始训练模型，G更新两次
sess = tf.Session()
sess.run(tf.global_variables_initializer())
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
# 随机生成的一个样本
samples = []
loss = {'d': [], 'g': []}
batch = mnist.train.next_batch(batch_size=batch_size)[0]  # 样本和标签
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch.shape =", batch.shape)
batch = get_pic()
batch = batch / 255
print(batch.shape)
batch = np.reshape(batch, [-1, 28, 28, 3])
batch = (batch - 0.5) * 2  # 【-1，1】
for i in range(600):
    print(i)
    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})  # 训练中
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)

    sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})  # 优化
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})

    if i % 25 == 0:
        print(i, d_ls, g_ls)  # 输出损失函数
        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})  # 用生成器生成图片，噪音保持不变
        gen_imgs = (gen_imgs + 1) / 2
        imgs = [img[:, :, 0:3] for img in gen_imgs]
        gen_imgs = montage(imgs)
        plt.axis('off')
        plt.imshow(gen_imgs, cmap='gray')
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i))
        #plt.show()
        samples.append(gen_imgs)

plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig('Loss.png')
plt.show()
imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)
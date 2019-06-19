# 样本尺寸32x32，训练模型保存在 ./samples/sample3/ 文件夹下

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# 导入数据
import os
from PIL import Image
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''

def get_pic():
    for pic in os.listdir(r'./picture'):
        pic_path = './picture/' + pic
        img = Image.open(pic_path)
        img = img.resize((32, 32), Image.ANTIALIAS)
        picture = np.array(img)
        picture = picture.flatten()
        try:
            data = np.vstack([data, picture])
            #data = np.stack((data, picture), axis=0)
        except Exception:
            data = picture
    return data

batch_size = 4  # 迭代样本数
z_dim = 4  # 噪音的维度
X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

train_num = tf.Variable(0)
train_num = tf.assign(train_num, train_num+1, name='train_num')

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


# 判别器
def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    print("--------------")
    print(image.shape)#32x32
    with tf.variable_scope('discriminator', reuse=reuse):  # 划分区域
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))  # 输入一张图片
        #                         输入数据，卷积核的高和宽，卷积核个数，步长
        print(h0.shape)#16
        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=132, strides=2, padding='same')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))  # BN在激活函数之前
        #                                       输入，是否处于训练模式，       衰减系数
        print(h1.shape)#8

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))
        print(h2.shape)#4

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))
        print(h3.shape)#2

        h7 = tf.contrib.layers.flatten(h3)  # 全连接前的处理，第一维度，二三四维度相乘为一个向量
        h7 = tf.layers.dense(h7, units=1)  # 全连接层
        print("h7.shape =",h7.shape)#2
        print("--------------")
        return tf.nn.sigmoid(h7), h7

def bottleneck(inputs, filter_out):
    momentum = 0.9
    if inputs.shape[3] == filter_out:
        residual = tf.layers.conv2d_transpose(inputs, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual)
        residual = tf.nn.relu(tf.contrib.layers.batch_norm(residual, is_training=is_training, decay=momentum))
        residual = tf.layers.conv2d_transpose(residual, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual)
    else:
        residual = tf.layers.conv2d_transpose(inputs, kernel_size=3, filters=filter_out, strides=2, padding='same')
        residual = tf.layers.batch_normalization(residual)
        residual = tf.nn.relu(tf.contrib.layers.batch_norm(residual, is_training=is_training, decay=momentum))
        residual = tf.layers.conv2d_transpose(residual, kernel_size=3, filters=filter_out, strides=1, padding='same')
        residual = tf.layers.batch_normalization(residual)
        inputs = tf.layers.conv2d_transpose(inputs, kernel_size=1, filters=filter_out, strides=2, padding='same')
    out = tf.add(inputs, residual)
    return out



# 生成器
def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 2
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))
        print(">>>>>>> h0.shape =", h0.shape)#(4,20,20,512)

        h1 = bottleneck(h0, 512)
        print(">>>>>>> h1.shape =", h1.shape)#(4,20,20,512)

        h2 = bottleneck(h1, 512)
        print(">>>>>>> h2.shape =", h2.shape)#(4,20,20,512)

        h3 = bottleneck(h2, 256)
        print(">>>>>>> h3.shape =", h3.shape)#(4,40,40,256)

        h4 = bottleneck(h3, 256)
        print(">>>>>>> h4.shape =", h4.shape)#(4,40,40,256)

        h5 = bottleneck(h4, 132)
        print(">>>>>>> h5.shape =", h5.shape)#(4,80,80,132)

        h6 = bottleneck(h5, 132)
        print(">>>>>>> h6.shape =", h6.shape)#(4,80,80,132)

        h7 = bottleneck(h6, 64)
        print(">>>>>>> h7.shape =", h7.shape)#(4,160,160,64)

        h8 = bottleneck(h7, 64)
        print(">>>>>>> h8.shape =", h8.shape)#(4,160,160,64)

        h9 = bottleneck(h8, 32)
        print(">>>>>>> h9.shape =", h9.shape)#(4,320,320,32)

        h10 = bottleneck(h9, 32)
        print(">>>>>>> h10.shape =", h10.shape)#(4,320,320,32)

        out = tf.layers.conv2d_transpose(h10, kernel_size=3, filters=3, strides=1,
                                         padding='same', activation=tf.nn.tanh, name='g')
        print(">>>>>>> out.shape =", out.shape)#(4,32,32,3)
        return out



# 定义损失函数，虽然是两个部分，但是参数是共享的
print("--0--")
g = generator(noise)
print("--1--")
d_real, d_real_logits = discriminator(X)  # 真实样本
print("--2--")
print(g.shape)
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

saver = tf.train.Saver() # 需要放在函数外面
def train_mode(SAMPLES_NUM, TRAIN_NUM = 100): #SAMPLES：第几个样本  # TRAIN_NUM：对这个样本训练的总数
    OUTPUT_DIR = './samples/'+'sample'+ str(SAMPLES_NUM)  # 输出文件
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    ################# 获取训练样本 ##################
    batch = get_pic()
    batch = np.reshape(batch, [-1, 32, 32, 3])/255
    batch = (batch - 0.5) * 2  # [-1，1]
    print(">>>>>>>>>>> batch.shape = ", batch.shape)
    #temp = (np.reshape(batch, [32, 32, 3]))
    temp = (batch + 1)/2
    temp = montage(temp)
    #img = Image.fromarray(np.uint8(temp)).convert('RGB')
    #img.save(os.path.join(OUTPUT_DIR, 'sample.jpg'))
    #img.show()
    #batch = (batch/255 - 0.5) * 2  # [-1，1]
    plt.axis('off')
    plt.imshow(temp, cmap='gray')
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample.jpg'))
    #plt.show()
    #batch = np.reshape(batch, [-1, 32, 32, 3])

    # 参数
    z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    print(">>>>>>>>>>>> z_samples.shape = ", z_samples.shape)
    #samples = []
    loss = {'d': [], 'g': []}


    with tf.Session() as sess:
        try:
            MODE_META = OUTPUT_DIR + '/model/my-model.meta'
            print(MODE_META)
            MODE_DIR = OUTPUT_DIR + '/model'
            print(MODE_DIR)
            saver2 = tf.train.import_meta_graph(MODE_META)
            saver2.restore(sess, tf.train.latest_checkpoint(MODE_DIR))
            print("加载模型...")
        except Exception:
            print("加载模型出错!")
            print("重新开始训练模型...")
            sess.run(tf.global_variables_initializer())

        i = sess.run(train_num)
        while i < TRAIN_NUM:
            i = i + 1
            print("第",i,"次训练")
            n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

            for count in range(3):
                sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})  # 优化
            for count in range(1):
                sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})

            if i % 10 == 0:
                d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})  # 训练中
                loss['d'].append(d_ls)
                loss['g'].append(g_ls)
                print(i, d_ls, g_ls)  # 输出损失函数
                gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})  # 用生成器生成图片，噪音保持不变
                gen_imgs = (gen_imgs + 1) / 2 * 255
                #gen_imgs = (gen_imgs + 1) / 2
                print(">>>>>>>> gen_imgs.shape =",gen_imgs.shape)
                imgs = [img[:, :, 0:3] for img in gen_imgs]
                gen_imgs = montage(imgs)
                #gen_imgs = gen_imgs[0] - gen_imgs[1]
                #gen_imgs = np.reshape(gen_imgs, [-1, 32,32,3])
                #img = Image.fromarray(np.uint8(gen_imgs)).convert('RGB')
                #img.save(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i))
                #plt.axis('off')
                #plt.imshow(gen_imgs, cmap='gray')
                #plt.savefig(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i))
                cv2.imwrite(OUTPUT_DIR+'/sample_%d.jpg'%i, gen_imgs)
                #plt.show()
                #samples.append(gen_imgs)
            MODE_SAVE = OUTPUT_DIR + '/model/my-model'
            saver.save(sess, MODE_SAVE)
            #saver.save(sess, "model/my-model")
            sess.run(train_num)
        plt.plot(loss['d'], label='Discriminator')
        plt.plot(loss['g'], label='Generator')
        plt.legend(loc='upper right')
        plt.savefig('Loss.png')
        plt.show()
        #imageio.mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=5)

if __name__ == '__main__':
    pic_num = 3
    train_mode(pic_num, 3000)
# -*-coding:utf-8-*-
import os
import numpy as np
import tensorflow as tf
from numpy import *
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'train/'
test_image_dir = 'ceshi'
batch = []
labels = []
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
for each in classes:
	labels.append(each)

with tf.Session() as sess:
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        # 载入VGG16模型
        vgg.build(input_)
    files = os.listdir(test_image_dir)
    for ii,file in enumerate(files,1):
    	img = utils.load_image(os.path.join(test_image_dir, file))
    	batch.append(img.reshape((1, 224, 224, 3)))
    	if ii == len(files):
    		images = np.concatenate(batch)
    		feed_dict = {input_: images}
    		codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

# 输入数据的维度
inputs_ = tf.placeholder(tf.float32, shape=[None, codes_batch.shape[1]])
# 标签数据的维度
labels_ = tf.placeholder(tf.int64, shape=[None, 92])

# 加入一个256维的全连接的层
fc = tf.contrib.layers.fully_connected(inputs_, 256)
# 加入一个5维的全连接层
logits = tf.contrib.layers.fully_connected(fc, 92, activation_fn=None)
# 得到最后的预测分布
predicted = tf.nn.softmax(logits)


with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, "beicun/checkpoints/flowers.ckpt")
	feed = {inputs_: codes_batch}
	result = sess.run(predicted, feed_dict=feed)
	m = argmax(result, 1)
	for i in range(12):
		a = int(m[i])
        c = labels[a]
        print (c)
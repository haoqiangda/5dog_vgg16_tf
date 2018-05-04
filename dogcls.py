# -*-coding:utf-8-*-
import os
import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = '/home/../dogImages/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir+each)]

batch_size = 10
codes_list = []
labels = []
# batch数组用来临时存储图片数据
batch = []
codes = None 

with tf.Session() as sess:
	vgg = vgg16.Vgg16()
	input_ = tf.placeholder(tf.float32,[None,224,224,3])
	with tf.name_scope("content_vgg"):
		vgg.build(input_)
	# 对每个不同种类的dog分别用VGG16计算特征值
	for each in classes:
		print("Starting {} images".format(each))
		class_path = data_dir+each
		files = os.listdir(class_path)
		for ii,file in enumerate(files,1):
			# 载入图片并放入batch数组中
			img = utils.load_image(os.path.join(class_path,file))  # resize to 224, 224
			batch.append(img.reshape(1,224,224,3))
			labels.append(each)
			# 如果图片数量到了batch_size则开始具体的运算
			if ii % batch_size ==0 or ii == len(files):
				images = np.concatenate(batch)
				feed_dict = {input_:images}
				# 计算特征值
				codes_batch = sess.run(vgg.relu6,feed_dict=feed_dict)
				# 将结果放入到codes数组中
				if codes is None:
					codes = codes_batch
				else:
					codes = np.concatenate((codes,codes_batch))
				# 清空数组准备下一个batch的计算
				batch = []
				print('{}images processed'.format(ii))

with open('codes','w') as f:
	codes.tofile(f)
import csv
with open('labels','w') as f:
	writer = csv.writer(f,delimiter='\n')
	writer.writerow(labels)

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)

# StratifiedShuffleSplit 方法来进行分层随机划分。假设我们使用训练集：验证集：测试集 = 8:1:1
from sklearn.model_selection import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
train_idx,val_idx = next(ss.split(codes,labels))
half_val_len = int(len(val_idx)/2)
val_idx,test_idx = val_idx[:half_val_len],val_idx[half_val_len:]

train_x,train_y = codes[train_idx],labels_vecs[train_idx]
val_x,val_y = codes[val_idx],labels_vecs[val_idx]
test_x,test_y = codes[test_idx],labels_vecs[test_idx]

print("Train shapes(x,y)",train_x.shape,train_y.shape)
print("Validation shapes(x,y)",val_x.shape,val_y.shape)
print("Test shapes(x,y)",test_x.shape,test_y.shape)

#start to train the networks 
inputs_ = tf.placeholder(tf.float32,shape=[None,codes.shape[1]])
labels_ = tf.placeholder(tf.int64,shape=[None,labels_vecs.shape[1]])

#add the extra networks 
fc = tf.contrib.layers.fully_connected(inputs_,256)
logits = tf.contrib.layers.fully_connected(fc,labels_vecs.shape[1],activation_fn=None)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_,logits=logits)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer().minimize(cost)
predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted,1),tf.argmax(labels_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#为了方便把数据分成一个个 batch 以降低内存的使用，还可以再用一个函数专门用来生成 batch。
def get_batches(x,y,n_batches=10):
	batch_size = len(x)//n_batches
	for ii in range(0,n_batches*batch_size,batch_size):
		if ii!=(n_batches-1)*batch_size:
			X,Y = x[ii:ii+batch_size],y[ii:ii+batch_size]
		else:
			X,Y = x[ii:],y[ii:]
		yield X,Y
#NOW We can to train 
epochs = 20
iteration = 0
# 保存模型的保存器
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in range(epochs):
		for x,y in get_batches(train_x,train_y):
			feed = {inputs_:x,
					labels_:y}
			# 训练模型
			loss,_ = sess.run([cost,optimizer],feed_dict=feed)
			print("Epoch:{}/{}".format(e+1,epochs),
				  "Iteration:{}".format(iteration),
				  "Training loss:{:.5f}".format(loss))
			iteration+=1
		if e>=0:
			feed = {inputs_:val_x,
					labels_:val_y}
			val_acc = sess.run(accuracy,feed_dict=feed)
			print("Epoch:{}/{}".format(e,epochs),
				  "Iteration:{}".format(iteration),
				  "Validation Acc:{:.5f}".format(val_acc))
	# 保存模型		
	saver.save(sess,"checkpoints/dogs.ckpt")
#测试模型效果
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    
    feed = {inputs_: test_x,
            labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))
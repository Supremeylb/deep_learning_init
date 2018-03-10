#two inputs and one output ,no hidden
import tensorflow as tf 
import numpy as np 

batch_size = 8

x = tf.placeholder(tf.float32,shape(None,2),name = "x_input")
y_ = tf.placeholder(tf.float32,shape(None,1),name="y_output")

#feedward
w = tf.Variable(tf.random.normal(2,1),stddev=1,seed=1)
b = tf.constant(np.random.randn(0,1))
y = np.dot(x,w)+b 

#cost function
loss_more = 10
loss_less = 1
cost_func = tf.reduce_sum(tf.select(tf.greater(y,y_),loss_more(y-y_),loss_less(y_-y)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	for i in range(5000):
		sess.run(train_step,feed_dict = {x：，y_:})
# set the learning rate
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay（0.1，global_step,100,
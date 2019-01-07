import tensorflow as tf
import numpy as np

X = tf.random_normal(shape=[3,120,6], dtype=tf.float32)
X = tf.reshape(X, [-1,120, 6])
print(X)
# keep_prob = tf.placeholder(tf.float32,'keep_prob')
def get_a_cell():
    cell = tf.nn.rnn_cell.BasicRNNCell(10)#此处的10和最终输出的维数有关，相当于每个词输出向量的维数,在tensorflow中默认每一层细胞的个数和输出维数相当
    drop = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
    return drop
# cell = tf.nn.rnn_cell.BasicLSTMCell(10)  # 也可以换成别的，比如GRUCell，BasicRNNCell等等
# cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
lstm_multi = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(4)])
state = lstm_multi.zero_state(3, tf.float32)
output, states = tf.nn.dynamic_rnn(lstm_multi, X, initial_state=state)#states是每一层lstm层最后一个step的输出
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(output.get_shape())
    print(sess.run(states))
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold

data = pd.read_csv('train.csv')

features = data.iloc[:,:-1]
target = data.iloc[:,-1]

samples_num, features_num = features.shape
hl1 = 1000
hl2 = 1000

outputl = 1
batch = 100

x = tf.placeholder('float', [None, features_num])
y = tf.placeholder('float')


def nnmodel(data):
    weighted_input_layer = {'weights': tf.Variable(tf.matrix_diag(tf.random_normal([features_num])))}

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([features_num, hl1])),
                      'biases': tf.Variable(tf.random_normal([hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([hl1, hl2])),
                      'biases': tf.Variable(tf.random_normal([hl2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hl2, outputl])),
                    'biases': tf.Variable(tf.random_normal([outputl]))}

    wl = tf.matmul(data, weighted_input_layer['weights'])
    wl = tf.nn.relu(wl)

    l1 = tf.add(tf.matmul(wl, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output


def train(x):
    prediction = nnmodel(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 15
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print('Total Epochs:', epochs)


        for epoch in range(epochs):
            epoch_loss = 0
            kf = KFold(n_splits=int(samples_num / batch), random_state=None, shuffle=True)

            for _, sample_index in kf.split(features):
                epoch_x = features.loc[sample_index]
                epoch_y = target.loc[sample_index]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # graph = tf.get_default_graph()
            # print(sess.run(graph.get_operation_by_name('Variable_2')))
            print('Epoch', epoch, 'loss:', epoch_loss)

        summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval(feed_dict={x: features, y: target}, session=sess))


train(x)
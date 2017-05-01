import tensorflow as tf
import tensorflow.contrib.layers as layers
import math

class DECONV_RNN(object):
    """
    An RNN to deconvolve Fluorescent traces of transcriptional activity
    """

    def __init__(self, max_length, batch_size, num_input_classes, num_output_classes, num_rnn_neurons, num_rnn_layers, conv_filter_sizes, rnn_input_size, pool_kernel_sizes, deprecated):
        # Define Placeholders
        with tf.name_scope("inputs"):
            self.input_x = tf.placeholder(tf.float32, [None, max_length, num_input_classes], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, max_length], name="input_y")
            #self.full_labels = tf.placeholder(tf.float32, [None, max_length, num_output_classes], name="input_y_full")
            self.seq_lengths = tf.placeholder(tf.int32, None, name = "seq_lengths")
            self.dropout = tf.placeholder(tf.float32, None, name = "dropout")

            if deprecated:
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_rnn_neurons, state_is_tuple=True)  # Or LSTMCell(num_rnn_neurons)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_rnn_layers)

            if not deprecated:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_rnn_neurons, state_is_tuple=True)  # Or LSTMCell(num_rnn_neurons)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cell = tf.contrib.rnn.MultiRNNCell([cell] * num_rnn_layers)

        with tf.name_scope("Conv"):
            # Create a convolution + maxpool layer for each filter size
            input = tf.reshape(self.input_x,[-1, max_length, num_input_classes,1])
            #width = num_input_classes
            for i, filter_size in enumerate(conv_filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % i):
                    # Convolution Layer

                    filter_shape = tf.cast(filter_size,tf.int32)
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_" + str(i))

                    b = tf.Variable(tf.constant(0.1, shape=[filter_size[3]]), name="b_" + str(i))

                    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME", name="conv_" + str(i))

                    # Apply nonlinearity
                    h = tf.nn.relu(conv + b, name="relu_" + str(i))
                    pool_kernel = pool_kernel_sizes[i]

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, pool_kernel[0], pool_kernel[1], 1],
                        strides=[1, pool_kernel[0], pool_kernel[1], 1],
                        padding='SAME',
                        name="pool_" + str(i))

                    #width = tf.cast(tf.ceil(tf.cast(width,tf.float32)/tf.cast(pool_kernel[1],tf.float32)),tf.int32)*tf.cast(filter_size[1],tf.int32)
                    input = pooled

        with tf.name_scope("LSTM"):
            input_shape = input.get_shape()
            rnn_input = tf.reshape(input,[batch_size, max_length, input_shape[-1].value])
            rnn_input.set_shape([None, None, input_shape[-1].value])
            #"Unfold" Network
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw = cell,
                dtype=tf.float32,
                sequence_length=self.seq_lengths,
                inputs=rnn_input)

        outputs_fw, outputs_bw = outputs

        with tf.name_scope("softmax"):

            # Output layer weights
            W_fw = tf.Variable(tf.truncated_normal(
                shape=[tf.cast(num_rnn_neurons,tf.int32), tf.cast(num_output_classes,tf.int32)],
                stddev=0.1),
                name="FW_Weights")

            b_fw = tf.Variable(tf.constant(0.1, shape=[num_output_classes]), name="b_fw")
            rnn_outputs_flat_fw = tf.reshape(outputs_fw, [-1, num_rnn_neurons])
            logits_flat_fw = tf.nn.xw_plus_b(rnn_outputs_flat_fw, W_fw, b_fw, name="logits_fw")

            W_bw = tf.Variable(tf.truncated_normal(
            shape=[tf.cast(num_rnn_neurons,tf.int32), tf.cast(num_output_classes,tf.int32)],
            stddev=0.1),
            name="BW_Weights")

            b_bw = tf.Variable(tf.constant(0.1, shape=[num_output_classes]), name="b_bw")
            rnn_outputs_flat_bw = tf.reshape(outputs_bw, [-1, num_rnn_neurons])
            logits_flat_bw = tf.nn.xw_plus_b(rnn_outputs_flat_bw, W_bw, b_bw, name="logits_fw")

            logits_flat = tf.add(logits_flat_bw, logits_flat_fw, name = "combined_logits")
            self.probs_flat = tf.nn.softmax(logits_flat, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probs_flat, labels=tf.reshape(tf.cast(self.input_y,tf.int32),[-1])),name="loss")

        with tf.name_scope("performance"):

            self.prediction = tf.reshape(self.probs_flat, [-1, tf.cast(max_length,tf.int32), tf.cast(num_output_classes,tf.int32)])
            if deprecated:
                self.accuracy = tf.reduce_mean(1.0 - tf.abs(tf.cast(tf.argmax(self.prediction,dimension=2),tf.float32) - self.input_y) / tf.cast(num_output_classes-1, tf.float32))
            else:
                self.accuracy = tf.reduce_mean(1.0 - tf.abs(tf.cast(tf.argmax(self.prediction,axis=2),tf.float32) - self.input_y) / tf.cast(num_output_classes-1, tf.float32))
            #self.accuracy_cat = 1.0 - tf.reduce_mean(tf.equal(tf.cast(tf.argmax(self.prediction,axis=2),tf.float32),self.input_y))

            if deprecated:
                probs_vec = tf.cast(tf.argmax(self.probs_flat,dimension=1), tf.float32)
            else:
                probs_vec = tf.cast(tf.argmax(self.probs_flat, axis=1), tf.float32)

            y_flat = tf.reshape(self.input_y, [-1])

            class_avg_pred = tf.reduce_mean(probs_vec)
            class_avg_true = tf.reduce_mean(y_flat)

            pred_diff = probs_vec - class_avg_pred
            true_diff = y_flat - class_avg_true

            denominator = tf.sqrt(tf.reduce_sum(tf.square(pred_diff))) * tf.sqrt(tf.reduce_sum(tf.square(true_diff) ))        # Accuracy per time step

            if deprecated:
                self.correlation = tf.reduce_mean(tf.div(tf.reduce_sum(pred_diff * true_diff), denominator))
            else:
                self.correlation = tf.reduce_mean(tf.divide(tf.reduce_sum(pred_diff * true_diff), denominator))


import tensorflow as tf
import tensorflow.contrib.layers as layers

class DECONV_RNN(object):
    """
    An RNN to deconvolve Fluorescent traces of transcriptional activity
    """

    def __init__(self, max_length, num_input_classes, num_output_classes, num_neurons, num_layers):
        # Define Placeholders
        with tf.name_scope("inputs"):
            self.input_x = tf.placeholder(tf.float32, [None, max_length, num_input_classes], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None, max_length], name="input_y")
            #self.full_labels = tf.placeholder(tf.float32, [None, max_length, num_output_classes], name="input_y_full")
            self.seq_lengths = tf.placeholder(tf.int32, None, name = "dropout")
            self.dropout = tf.placeholder(tf.float32, None, name = "dropout")
            total_length = tf.reduce_sum(self.seq_lengths)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, state_is_tuple=True)  # Or LSTMCell(num_neurons)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        #"Unfold" Network
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw = cell,
            dtype=tf.float32,
            sequence_length=self.seq_lengths,
            inputs=self.input_x)

        outputs_fw, outputs_bw = outputs

        with tf.name_scope("softmax"):

            # Output layer weights
            W_fw = tf.Variable(tf.truncated_normal(
                shape=[num_neurons, num_output_classes],
                stddev=0.1),
                name="FW_Weights")

            b_fw = tf.Variable(tf.constant(0.1, shape=[num_output_classes]), name="b_fw")
            rnn_outputs_flat_fw = tf.reshape(outputs_fw, [-1, num_neurons])
            logits_flat_fw = tf.nn.xw_plus_b(rnn_outputs_flat_fw, W_fw, b_fw, name="logits_fw")

            W_bw = tf.Variable(tf.truncated_normal(
            shape=[num_neurons, num_output_classes],
            stddev=0.1),
            name="BW_Weights")

            b_bw = tf.Variable(tf.constant(0.1, shape=[num_output_classes]), name="b_bw")
            rnn_outputs_flat_bw = tf.reshape(outputs_bw, [-1, num_neurons])
            logits_flat_bw = tf.nn.xw_plus_b(rnn_outputs_flat_bw, W_bw, b_bw, name="logits_fw")

            logits_flat = tf.add(logits_flat_bw, logits_flat_fw, name = "combined_logits")
            self.probs_flat = tf.nn.softmax(logits_flat, name="predictions")
            logits_full = tf.reshape(logits_flat, [-1,max_length,num_output_classes], name="logits_full")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits_full), name = "cross")

        with tf.name_scope("performance"):

            self.prediction = tf.reshape(self.probs_flat, [-1, max_length, num_output_classes])
            #self.accuracy = tf.subtract(1.0, tf.divide(self.prediction,self.full_labels))),2.0 * tf.cast(total_length,tf.float32), name="acc_per_timestep"))


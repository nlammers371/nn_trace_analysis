from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import os
import csv
import sys
import datetime
import time
from time import gmtime, strftime
from utilities.generate_traces import generate_traces_gillespie
from models.deconv_rnn import DECONV_RNN
import tensorflow as tf
import math

#DEFAULT Training Parameters
#--------------------------------------------------------------------------------------------
#Hyperparameters
batch_size =  10
num_test_steps = 2500
record_every = 1
evaluate_every = 1
test_type = "evaluate"
deprecated = 0


#Trace Parameters
trace_length = 500
uniform_trace_lengths = 1
memory = 20
switch_low = 3
switch_high = 12
noise_scale = .025
alpha = 6.0
fluo_scale = 201
init_scale = int((fluo_scale-1)/memory + 1)

soft_placement = True
log_placement = False

#Paths
read_dir = os.path.join( 'output/train')
folder_name = "train_015_2017-04-29_06:15:51"
write_dir = os.path.join( 'output/')

checkpoint_full_path = os.path.join(read_dir,folder_name)

# Generate batches for training and testing
# ============================================================================================

init_time = time.time()

print("Generating Training and Testing Data...")


eval_batches = generate_traces_gillespie(memory = memory,
                                                 length = trace_length,
                                                 input_size = fluo_scale,
                                                 batch_size=batch_size,
                                                 num_steps=num_test_steps,
                                                 alpha=alpha,
                                                 switch_low=switch_low,
                                                 switch_high = switch_high,
                                                 noise_scale=noise_scale
                                                 )




start_time = time.time()
checkpoint_file = tf.train.latest_checkpoint(os.path.join(checkpoint_full_path, "checkpoints/"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=soft_placement,
        log_device_placement=log_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("inputs/input_x").outputs[0]

        seq_length_vec = graph.get_operation_by_name("inputs/seq_lengths").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("inputs/dropout").outputs[0]

        dropout_keep_prob_1 = graph.get_operation_by_name("inputs/dropout_1").outputs[0]

        probs_flat = graph.get_operation_by_name("softmax/predictions").outputs[0]

        # Collect the predictions here

        for k, batch in enumerate(eval_batches):
            x_inputs, y_labels, seq_lengths, int_labels, int_inputs = batch
            id_list = range(k*batch_size,(k+1)*batch_size)

            batch_predictions = sess.run(probs_flat,{input_x: x_inputs, dropout_keep_prob: 1.0, dropout_keep_prob_1: 1.0, seq_length_vec: seq_lengths})

            bp_array = np.asarray(batch_predictions)

            #Take most likely class as the prediction
            bp_max = np.argmax(bp_array, axis=1)

            #Take weighted class average as prediction
            class_array = np.tile(np.arange(init_scale,[len(bp_max[0]),1]))
            bp_mean = np.mean(bp_array*class_array,axis=1)

            inputs_flat = np.reshape(int_inputs,(-1))
            labels_flat = np.reshape(int_labels,(-1))

            if ~os.path.isdir(os.path.join(checkpoint_full_path,"evaluate")):
                os.makedirs(os.path.join(checkpoint_full_path,"evaluate"))
            if k==0:
                write = 'wb'
            else:
                write = 'a'

            with open(os.path.join(checkpoint_full_path,"evaluate","test_predictions.csv"),write) as outfile:
                writer = csv.writer(outfile)
                for i in xrange(len(inputs_flat)):
                    row = [id_list[i%trace_length], inputs_flat[i], labels_flat[i], bp_max[i], bp_mean[i]]
                    writer.writerow(row)

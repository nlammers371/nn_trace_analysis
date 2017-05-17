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
from utilities.generate_traces import generate_traces_gill_r_mat
from models.deconv_rnn import DECONV_RNN
import tensorflow as tf
import math

#DEFAULT Training Parameters
#--------------------------------------------------------------------------------------------
#Hyperparameters
batch_size =  1
num_test_steps = 10
record_every = 1
evaluate_every = 10
test_type = "evaluate"
deprecated = 1


#Trace Parameters
trace_length = 500
uniform_trace_lengths = 1
memory = 40
switch_low = 3
switch_high = 12
noise_scale = .0125
alpha = 11.7
fluo_scale = 1001
init_scale = int((fluo_scale-1)/memory + 1)

soft_placement = True
log_placement = False

#R = np.array([[-0.011, 0.01, 0,.024],    [.011,-.018,.013,0],     [0,0.007,-.013,.023],     [0,.001, 0,-0.047]]) * 10.2
#v = np.array([0,3,6,10])

R = np.array([[-0.01145, 0.0095, 0],    [.01145,-0.0180,.044],     [0,0.0085,-.044]]) * 10.2
v = np.array([0,4,8])

#R = np.array([[-.012, .024],[.012,-.024]]) * 5.1
#v = np.array([0,7])

#Paths
test_name = "eve2_3state_ap41"
read_dir = os.path.join( 'output/train')
folder_name = "train_016_2017-04-30_02:08:02"
write_dir = os.path.join( 'output/')

checkpoint_full_path = os.path.join(read_dir,folder_name)

# Generate batches for training and testing
# ============================================================================================

init_time = time.time()

print("Generating Training and Testing Data...")


eval_batches = generate_traces_gill_r_mat(memory = memory,
                                                 length = trace_length,
                                                 input_size = fluo_scale,
                                                 batch_size=batch_size,
                                                 num_steps=num_test_steps,
                                                 alpha=alpha,
                                                 switch_low=switch_low,
                                                 r_mat = R,
                                                 v = v,
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

        seq_length_vec = graph.get_operation_by_name("inputs/dropout").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("inputs/dropout_1").outputs[0]

        #dropout_keep_prob_1 = graph.get_operation_by_name("inputs/dropout_2").outputs[0]

        probs_flat = graph.get_operation_by_name("softmax/predictions").outputs[0]

        # Collect the predictions here

        for k, batch in enumerate(eval_batches):
            print("Processing Batch " + str(k+1) + " of " + str(num_test_steps))
            x_inputs, y_labels, seq_lengths, _, int_inputs, conv_labels = batch
            id_list = range(k*batch_size,(k+1)*batch_size)

            batch_predictions = sess.run(probs_flat,{input_x: x_inputs, dropout_keep_prob: 1.0, seq_length_vec: seq_lengths})

            bp_array = np.asarray(batch_predictions)
            bp_array = np.reshape(bp_array,(-1,init_scale))
            #Take most likely class as the prediction
            bp_max = np.argmax(bp_array, axis=1)

            #Take weighted class average as prediction
            class_array = np.tile(np.arange(init_scale),[len(bp_max),1])
            bp_mean = np.sum(bp_array*class_array,axis=1)

            inputs_flat = np.reshape(int_inputs,(-1))
            labels_flat = np.reshape(conv_labels,(-1))

            if not os.path.isdir(os.path.join(checkpoint_full_path,"evaluate")):
                os.makedirs(os.path.join(checkpoint_full_path,"evaluate"))
            if k==0:
                write = 'wb'
            else:
                write = 'a'

            with open(os.path.join(checkpoint_full_path,"evaluate", test_name + ".csv"),write) as outfile:
                writer = csv.writer(outfile)
                for i in xrange(len(inputs_flat)):
                    row = [id_list[int(i/trace_length)], inputs_flat[i], labels_flat[i], bp_max[i], bp_mean[i]]
                    writer.writerow(row)


            with open(os.path.join(checkpoint_full_path,"evaluate", test_name + "full_probs.csv"),write) as outfile:
                writer = csv.writer(outfile)
                for i in xrange(len(inputs_flat)):
                    row = [id_list[int(i/trace_length)], inputs_flat[i], labels_flat[i], bp_max[i], bp_mean[i]]
                    writer.writerow(row)
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
from generate_traces import generate_traces_unconstrained
from deconv_rnn import DECONV_RNN
import tensorflow as tf

#DEFAULT Training Parameters
#--------------------------------------------------------------------------------------------
#Hyperparameters
batch_size =  25
num_training_steps = 2000
record_every = 25
evaluate_every = 25
test_type = "check"

#Architecture Parameters
num_layers = 4
num_neurons = 100
dropout_keep_prob = .8

#Trace Parameters
trace_length = 500
uniform_trace_lengths = 0
memory = 5
fluo_scale = 501

#Paths
write_dir = os.path.join( 'output/')

# Generate batches for training and testing
# ============================================================================================

init_time = time.time()

print("Generating Training and Testing Data...")

num_tests_total = int(num_training_steps  / evaluate_every) + 1

training_batches = generate_traces_unconstrained(memory = memory,
                                                 length = trace_length,
                                                 input_size = fluo_scale,
                                                 batch_size=batch_size,
                                                 num_steps=num_training_steps,
                                                 )


testing_batches = generate_traces_unconstrained(memory = memory,
                                                 length = trace_length,
                                                 input_size = fluo_scale,
                                                 batch_size=batch_size,
                                                 num_steps=num_tests_total,
                                                 )


init_time = time.time() - init_time
print("Data Loading and Batching Time: " + str(init_time))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session()
    with sess.as_default():
        rnn = DECONV_RNN(
            max_length = trace_length,
            num_neurons = num_neurons,
            num_layers = num_layers,
            num_input_classes = fluo_scale,
            num_output_classes= int(fluo_scale/memory)
            )

    # ============================================================================================
    # Open Interactive TF Session
    sess = tf.InteractiveSession()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(rnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Generate Write Directory for Training and Save Performance Metrics and (optionally) validation set
    # ============================================================================================
    test_outpath = os.path.join(write_dir, test_type)
    if not os.path.isdir(test_outpath):
        os.makedirs(test_outpath)
    # Output directory for models and summaries
    last_run = 0
    for file in os.listdir(test_outpath):
        if file.find(str(test_type) + "_") > -1:
            sub_char = str(test_type)+"_|_2017.*"
            run_num = int(re.sub(sub_char, "", file))
            last_run = max(last_run, run_num)
    run = last_run + 1
    run_num = '0'*(3-len(str(run))) + str(run)
    out_dir = os.path.join(write_dir, test_type, test_type + "_" + run_num + "_" + re.sub("\s","_",strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    print("Writing to {}\n".format(out_dir))

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.histogram("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)
    # Summaries for cross_entropy and accuracy
    loss_summary = tf.summary.scalar("cross_entropy", rnn.loss)
    acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge_all()
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge_all()
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    def train_step(x_batch, y_batch, int_labels):
        """
        A single training step
        """
        feed_dict = {
            rnn.input_x: x_batch,
            rnn.input_y: y_batch,
            rnn.dropout: dropout_keep_prob
        }
        _, step, summaries, cross_entropy, class_scores = sess.run(
            [train_op, global_step, dev_summary_op, rnn.loss, rnn.probs_flat],
            feed_dict)
        if current_step % record_every == 0:
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, int_labels, writer=dev_summary_writer):

        #Evaluates model on a dev set

        feed_dict = {
            rnn.input_x: x_batch,
            rnn.input_y: y_batch,
            rnn.dropout: 1.0
        }

        step, summaries, cross_entropy, class_scores, accuracy, correlation_vec = sess.run(
            [global_step, dev_summary_op, rnn.loss, rnn.probs_flat, rnn.accuracy],
            feed_dict)

        if writer:
            writer.add_summary(summaries, step)

        time_str = datetime.datetime.now().isoformat()

        print("{}: step {}, of {}, cross_entropy {:g}, accuracy {:g}".format(time_str, current_step, num_training_steps, cross_entropy, accuracy))

        if not os.path.exists(os.path.join(dev_summary_dir, "csv")):
            os.mkdir(os.path.join(dev_summary_dir, "csv"))

        dev_csv = open(os.path.join(dev_summary_dir, "csv", "acc_loss.csv"), "a")
        write = csv.writer(dev_csv)
        row = [time_str, current_step, num_training_steps, cross_entropy, accuracy]
        for corr in correlation_vec:
            row.append(corr)
        write.writerow(row)
        dev_csv.close()

        if current_step % evaluate_every * 50 == 0:
            with open(os.path.join(dev_summary_dir, "csv", "score_details.csv"),"a") as dev_score_csv:
                write = csv.writer(dev_score_csv)
                for i in xrange(len(y_batch[0])):
                    scores = class_scores[i,:]
                    answers =  y_batch[0][i]
                    row = [current_step]
                    for i in xrange(int(fluo_scale/memory)):
                        cell = scores[i]
                        row.append(cell)
                    for j in xrange(int(fluo_scale/memory)):
                        cell = answers[j]
                        row.append(cell)
                    write.writerow(row)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())
    #Train Model
    # ===========================================================================================
    tf.global_variables_initializer().run()

    # Train
    current_step = 0
    abs_start = time.time()
    start = time.time()
    for batch in training_batches:
        x_inputs, y_labels, int_labels = batch
        train_step(x_inputs, y_labels)

        if current_step % evaluate_every == 0:
            x_inputs, y_labels, int_lables = next(testing_batches)
            dev_step(x_inputs, y_labels, int_labels, writer=dev_summary_writer)

            print("")
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
            print("")
            cycle_time = time.time() - start
            abs_time = time.time() - abs_start
            avg_time = float(abs_time) / float(current_step + 1)
            projected_time = avg_time * float(num_training_steps - current_step - 1)
            if current_step != 0:
                print("prev time {:g}, total duration {:g}, avg time per step {:g}, projected time remaining {:g}".format(cycle_time, abs_time, avg_time, projected_time))
            start = time.time()

        current_step += 1

    saver.save(sess, checkpoint_prefix, global_step=current_step)
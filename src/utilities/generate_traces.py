import numpy as np
from matplotlib import pyplot as plt
import math

#Simple generator function to generate traces for training
#For now, assume that number of discrete states and system "memory" are known
#Question: can an RNN correctly infer loading rates and trajectories
#           if trained on traces generated using arbitrary A and v matrices?

def generate_traces_prob_mat(num_states, memory, length, input_size, batch_size):
    #define convolution kernel
    kernel = np.ones((1,memory))[0]
    input_list = []
    label_list = []
    for b in xrange(batch_size):
        #Set scale of inputs
        label_size = input_size/memory
        #Define random emissions vec (define as emissions per unit time)
        v = np.random.randint(0,label_size,(num_states,1))
        # define random transition matrix of proper form
        A_raw = np.random.random((num_states,num_states))
        A = np.array(A_raw / np.sum(A_raw,axis=0)[np.newaxis,:])
        trajectory = np.zeros((label_size,T), dtype='int')
        v_list = []
        v_list.append(v[np.random.randint(0,num_states)])
        trajectory[v_list[0],0] = 1

        for s in xrange(1,T):
            v_new = v[np.random.choice(num_states, p=A[:, trajectory[0, s - 1]])][0]
            v_list.append(v_new)
            trajectory[v_new,s] = 1

        F_series = np.convolve(kernel,np.array(v_list),mode='same')
        full_input = np.zeros((input_size,T))
        for f in xrange(T):
            full_input[int(F_series[f]),f] = 1

        input_list.append(np.ndarray.tolist(full_input))
        label_list.append(np.ndarray.tolist(trajectory))

    yield(input_list, label_list)


#This function defines the hard problem--an arbitrary number of emission states of an arbitrary magnitude
#Stay in discrete space for now.

def generate_traces_unconstrained(memory, length, input_size, batch_size, num_steps):
    #define convolution kernel
    kernel = np.ones((1,memory))[0]
    for step in xrange(num_steps):
        input_list = []
        label_list = []
        int_label_list = []
        int_input_list = []
        for b in xrange(batch_size):
            #Set scale of inputs
            label_size = int((input_size-1)/memory) + 1
            # define random transition matrix of proper form
            trajectory = np.zeros((length,label_size), dtype='int')
            v_list = []
            v_list.append(np.random.randint(0,label_size))
            trajectory[0,v_list[0]] = 1

            for s in xrange(1,length):
                v_new = np.random.randint(0,label_size)
                v_list.append(v_new)
                trajectory[s,v_new] = 1

            F_series = np.convolve(kernel,np.array(v_list),mode='same')
            full_input = np.zeros((length,input_size))
            for f in xrange(length):
                full_input[f,int(F_series[f])] = 1

            input_list.append(np.ndarray.tolist(full_input))
            label_list.append(np.ndarray.tolist(trajectory))
            int_label_list.append(v_list)
            int_input_list.append(F_series)

        seq_lengths = [length]*batch_size
        yield(input_list, label_list, seq_lengths, int_label_list, int_input_list)


if __name__ == "__main__":

    # num states
    K = 3
    # memory
    w = 1
    # Fix trace length for now
    T = 50
    # Input magnitude
    F = 2
    # Number of traces per batch
    batch_size = 1

    batches = generate_traces_unconstrained(w,T,F,batch_size,1)

    for batch in batches:
        input_list, label_list, seq_lengths, label_ints, input_ints = batch
        plt.plot(np.array(input_ints[0]))
        plt.plot(np.array(label_ints[0]))
        plt.show()
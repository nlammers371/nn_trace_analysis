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

def generate_traces_unconstrained(memory, length, input_size, batch_size, num_steps, noise_scale =.05, alpha=1.0):
    #define convolution kernel
    if alpha > 0:
        alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) + 1 * (i >= alpha) for i in xrange(memory)]
        #alpha_vec = np.array(alpha_vec[::-1])
    else:
        alpha_vec = np.array([1]*memory)

    kernel = np.ones((1,memory))[0]*alpha_vec
    print(kernel)
    for step in xrange(num_steps):
        input_list = []
        label_list = []
        int_label_list = []
        int_input_list = []
        for b in xrange(batch_size):
            # Set scale of inputs
            label_size = int((input_size - 1) / memory) + 1
            v_size = np.random.randint(2,label_size)
            v_choices = np.random.randint(0,label_size,size=v_size)
            # define random transition matrix of proper form
            trajectory = np.zeros((length,label_size), dtype='int')
            v_list = []
            v_list.append(np.random.choice(v_choices))
            trajectory[0,v_list[0]] = 1
            #Generate promoter trajectory
            for s in xrange(1,length):
                v_new = np.random.choice(v_choices)
                v_list.append(v_new)
                trajectory[s,v_new] = 1
            v_list = np.array(v_list)
            v_list.astype('float')
            #Convolve with kernel to generate compound signal
            F_series = np.floor(np.convolve(kernel,np.array(v_list),mode='full'))
            F_series = F_series[0:length]
            #Apply noise
            noise_vec = np.random.randn(length)*int(noise_scale*input_size)
            F_noised = F_series + noise_vec
            full_input = np.zeros((length, input_size))
            for f in xrange(length):
                full_input[f,max(0,min(input_size-1, int(F_noised[f])))] = 1

            input_list.append(np.ndarray.tolist(full_input))
            label_list.append(np.ndarray.tolist(trajectory))
            int_label_list.append(v_list.tolist())
            int_input_list.append(F_series.tolist())

        seq_lengths = [length]*batch_size
        yield(input_list, label_list, seq_lengths, int_label_list, int_input_list)


if __name__ == "__main__":

    # memory
    w = 10
    # Fix trace length for now
    T = 100
    # Input magnitude
    F = 201
    # Number of traces per batch
    batch_size = 1

    batches = generate_traces_unconstrained(w,T,F,batch_size,1,noise_scale =.5, alpha=10.0)

    for batch in batches:
        input_list, label_list, seq_lengths, label_ints, input_ints = batch
        print(input_ints)
        print(label_ints)
        plt.plot(np.array(input_ints[0]))
        plt.plot(np.array(label_ints[0]))
        plt.show()

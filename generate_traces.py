import numpy as np


#Simple generator function to generate traces for training
#For now, assume that number of discrete states and system "memory" are known
#Question: can an RNN correctly infer loading rates and trajectories
#           if trained on traces generated using arbitrary A and v matrices?

def generate_traces(num_states, memory, length, input_size, batch_size):
    #define convolution kernel
    kernel = np.ones((1,memory))
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

        F_series = np.convolve(kernel[0],np.array(v_list),mode='same')
        full_input = np.zeros((input_size,T))
        for f in xrange(T):
            full_input[int(F_series[f]),f] = 1

        input_list.append(np.ndarray.tolist(full_input))
        label_list.append(np.ndarray.tolist(trajectory))

    yield(input_list, label_list)

if __name__ == "__main__":

    # num states
    K = 3
    # memory
    w = 5
    # Fix trace length for now
    T = 500
    # Input magnitude
    F = 100
    # Number of traces per batch
    batch_size = 10

    batches = generate_traces(K,w,T,F,batch_size)

    for batch in batches:
        input_list, label_list = batch;

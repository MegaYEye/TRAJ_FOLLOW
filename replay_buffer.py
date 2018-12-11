import random
import numpy as np
class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):

        data = (state, action, reward, next_state, done)
        
        # add data to storage
        if len(self._storage)>=self._maxsize:
            self._storage.pop(0)
               
        self._storage.append(data)
        
    def sample(self, batch_size):

        #idxes = <randomly generate batch_size integers to be used as indexes of samples>
        idxes=np.random.choice(len(self._storage), batch_size)
        
        # collect <s,a,r,s',done> for each index

        data=[self._storage[i] for i in idxes]
        
        states, action, reward, next_states, is_done=zip(*list(data))
        
    
        return np.array(states), np.array(action), np.array(reward), np.array(next_states), np.array(is_done)


from collections import defaultdict
import random, math
import numpy as np
import pickle
from functools import partial
import h5py
import tensorflow as tf
import keras
import keras.layers as L
from keras.models import load_model,Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, Input
class A2C_Online:
    def __init__(self, discount, get_legal_actions,state_dim,action_dim):
        self.get_legal_actions = get_legal_actions
        #self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        # self._qvalues = defaultdict(partial(defaultdict,int))
        self.discount = discount
        self.state_dim=state_dim
        self.action_dim=action_dim

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        keras.backend.set_session(self.sess)
        self.network=self.create_network()
        self.sess.run(tf.global_variables_initializer())

    def symbolic_step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""
        logits, state_value = self.network(state_t)
        state_value = state_value[:, 0]
        
        assert tf.is_numeric_tensor(state_value) and state_value.shape.ndims == 1, \
            "please return 1D tf tensor of state values [you got %s]" % repr(state_value)
        assert tf.is_numeric_tensor(logits) and logits.shape.ndims == 2, \
            "please return 2d tf tensor of logits [you got %s]" % repr(logits)
        # hint: if you triggered state_values assert with your shape being [None, 1], 
        # just select [:, 0]-th element of state values as new state values
        
        return (logits, state_value)

    
    def create_network(self):

        inputs = Input(shape=self.state_dim)
        x = Dense(128, activation='relu')(inputs)
        logits = Dense(self.action_dim, activation='linear')(x)
        state_value = Dense(1, activation='linear')(x)
        self.network = Model(inputs=inputs, outputs=[logits, state_value])

        # let's create a network for approximate q-learning following guidelines above
        # <YOUR CODE: stack more layers!!!1 >
        self.state_t = tf.placeholder('float32', [None,] + list(self.state_dim))
        self.states_ph = tf.placeholder('float32', shape=(None,) + self.state_dim,name="st")
        self.actions_ph = tf.placeholder('int32', shape=[None],name="ac")
        self.rewards_ph = tf.placeholder('float32', shape=[None],name="r")
        self.next_states_ph = tf.placeholder('float32', shape=(None,) + self.state_dim,name="nst")
        self.is_done_ph = tf.placeholder('bool', shape=[None],name="done")

        self.agent_outputs = self.symbolic_step(self.state_t)
        logits, state_values = self.symbolic_step(self.states_ph)
        next_logits, next_state_values = self.symbolic_step(self.next_states_ph)
        #next_state_values = next_state_values * (1 - self.is_done_ph)
        probs = tf.nn.softmax(logits)            # [n_envs, n_actions]
        logprobs = tf.nn.log_softmax(logits)     # [n_envs, n_actions]
        logp_actions = tf.reduce_sum(logprobs * tf.one_hot(self.actions_ph, self.action_dim), axis=-1) # [n_envs,]



        gamma = self.discount

        advantage = self.rewards_ph + gamma*next_state_values - state_values

        assert advantage.shape.ndims == 1, "please compute advantage for each sample, vector of shape [n_envs,]"

        # compute policy entropy given logits_seq. Mind the "-" sign!
        entropy =  -tf.reduce_sum(probs * logprobs, 1, name="entropy")

        assert entropy.shape.ndims == 1, "please compute pointwise entropy vector of shape [n_envs,] "

        self.actor_loss =  - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * tf.reduce_mean(entropy)

        # compute target state values using temporal difference formula. Use rewards_ph and next_step_values
        target_state_values = self.rewards_ph+gamma*next_state_values

        self.critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2 )

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.actor_loss + self.critic_loss)

        self.network.summary()



        return self.network


    def update(self, state, action, reward, next_state):

        # print(state.shape[0])

        self.sess.run([self.train_step],{
            self.states_ph: state, self.actions_ph: action, self.rewards_ph: reward, 
            self.next_states_ph: next_state, self.is_done_ph: np.array([False ]* state.shape[0])
        })
     


    def get_action(self, state):

        """pick actions given numeric agent outputs (np arrays)"""

        logits, state_values = self.network.predict(state)
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])

    def save_param(self, path):
        # self.network.save(path)
        try:
            saver = tf.train.Saver()
            saver.save(self.sess, "./"+path)
        except:
            print("save failure")
            
    def load_param(self, path):
        try:
            # self.network=self.create_network()
            saver = tf.train.Saver()
            saver.restore(self.sess, "./"+path)
        except:
     
            print("load param failure! let's start from a empty q table!")
            #self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
            self.network = self.create_network()

    def test(self):
        # state=np.array([0 for i in range(21)])
        x=np.array([[1 for i in range(2)]])
        x2=np.array([[2 for i in range(2)]])
        self.load_param("aaa.ckpt")
        for i in range(500):
            self.update( state=x, action=[1], reward=[20], next_state=x2)
        print(self.get_action(x)[0])
        self.save_param("aaa.ckpt")
        # print(self.get_best_action(x))

if __name__ == '__main__':
    agent=A2C_Online(discount=0.9, get_legal_actions=lambda s: range(9),state_dim=(2,),action_dim=5)
    agent.test()
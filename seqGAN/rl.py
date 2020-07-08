import numpy as np
from seqGAN.models import Generator
from utils import *
class Agent:
    def __init__(self,generate_samples,maxLen,c_e,g_lr=1e-3,eps=0.0):
        self.B = generate_samples
        self.maxLen = maxLen
        self.g_lr = g_lr
        self.c_e = c_e
        self.eps = eps
        self.generator = Generator(self.B,self.maxLen,self.c_e,self.g_lr)

    def act(self,state,c_input,deterministic=False):
        """
        :param state:[0.....t-1] action
        :param c_input:  emtion input
        :param deterministic:
        :return: t's action
        """
        state = state.reshape([self.B,-1])
        return self._act_on_word(state,c_input,deterministic=deterministic)

    def _act_on_word(self,state,c_input,deterministic=False):
        word = state[:,-1].reshape([-1,1])
        is_PAD = word == pad_id

        is_EOS = word == end_id
        is_end = is_PAD.astype(np.int) + is_EOS.astype(np.int)
        is_end = 1 - is_end
        is_end = np.reshape(is_end,(self.B,1))
        if(np.random.rand()<=self.eps):
            action = np.random.randint(low=0,high=dict_V,size=(self.B,1))
        elif not deterministic:
            prob = self.generator.predict(state,c_input)
            prob = np.reshape(prob,[self.B,-1])
            action = self.generator.sampling_word(prob).reshape([self.B,1])
        else:
            prob = self.generator.predict(state,c_input)
            action = np.argmax(prob,axis=-1).reshape([self.B,1])
        return action * is_end

    def save(self,path):
        self.generator.save(path)

    def load(self,path):
        self.generator.load(path)


class Environment:
    def __init__(self,discriminator,data_generator,g_beta,batch_size,maxLen=128,n_sample = 16,posRate=0.5):
        self.discriminator = discriminator
        self.data_generator = data_generator
        self.B = batch_size
        self.g_beta = g_beta
        self.maxLen = maxLen
        self.n_sample = n_sample
        self.posRate = posRate
        self.reset()

    def reset(self):
        self.t = 1
        self._state = np.zeros([self.B,1],dtype=np.int32)
        self._state[:,0] = start_id
        self.c_input = np.random.binomial(1,self.posRate,(self.B,1))

    def step(self,action):
        """
        :param action: t's action
        :return:
        """
        self.t = self.t + 1
        reward = self.Q(action)
        is_episode_end = self.t > self.maxLen
        self._append_state(action)
        next_state = self.get_state()
        return [next_state,reward,is_episode_end]

    def Q(self,action):
        reward = np.zeros([self.B,1])
        if self.t==2:
            Y_base = self._state
        else:
            Y_base = self.get_state()
        if(self.t>self.maxLen+1):
            Y = self._append_state(action,state=Y_base)
            return self.discriminator.predict(Y)
        for _ in range(self.n_sample):
            Y = Y_base
            y_t = self.g_beta.act(Y,self.c_input)
            Y = self._append_state(y_t,state=Y)
            for _ in range(self.t+1,self.maxLen):
                y_tau = self.g_beta.act(Y,self.c_input)
                Y = self._append_state(y_tau,state=Y)
            reward += self.discriminator.predict(Y) / self.n_sample
        return reward

    def _append_state(self,word,state=None):
        word = word.reshape(-1,1)
        if state is None:
            self._state = np.concatenate([self._state,word],axis=-1)
        else:
            return np.concatenate([state,word],axis=-1)

    def get_state(self):
        if self.t==1:
            return self._state
        else:
            return self._state[:,1:]


    def get_inputs(self):
        return self.c_input











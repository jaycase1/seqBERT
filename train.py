from utils import load_data,read_gen_data,text_segmentate,GeneratorPretraingGenerator,DiscriminatorGenerator,tokenizer,start_id,pad_id,end_id,get_weights
from keras.optimizers import Adam
from seqGAN.rl import Agent,Environment
import os
import numpy as np
from seqGAN.models import Generator,Discriminator,GeneratePretrain

class Trainer:
    def __init__(self,batchSize,maxLen,d_E,d_H,c_E,d_dropout,path_pos,path_neg,g_lr=1e-3,d_lr=1e-3,n_sample=16,generate_sample=50):
        self.B = batchSize
        self.T = maxLen
        self.d_E = d_E
        self.d_H = d_H
        self.c_E = c_E
        self.d_dropout = d_dropout
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.top = os.getcwd()
        self.n_sample = n_sample
        self.generate_sample = generate_sample
        self.pos_data = load_data(self.path_pos)

        self.g_data = GeneratorPretraingGenerator(self.pos_data,self.B)
        self.agent = Agent(self.generate_sample,self.T,self.c_E,self.g_lr)
        self.g_beta = Agent(self.generate_sample,self.T,self.c_E,self.g_lr)

        self.discriminator = Discriminator(d_E,d_H)
        self.env = Environment(self.discriminator,self.g_data,self.g_beta,self.generate_sample,self.T,self.n_sample)

        self.generator_pre = GeneratePretrain(self.c_E,self.g_lr)

    def preTrain(self,g_epoch=20,d_epoch=2,g_pre_path=None,d_pre_path=None,g_lr=1e-3,d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epoch,g_pre_path=g_pre_path,g_lr=g_lr)
        self.reflect_pre_train()
        self.pre_train_discriminator(d_epochs=d_epoch,d_pre_path=d_pre_path,d_lr=d_lr)


    def pre_train_generator(self,g_epochs=1,g_pre_path=None,g_lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top,'data','save','generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path
        print("Generator pre-Training")
        self.generator_pre.summary()
        self.generator_pre.fit_generator(
            self.g_data.forfit(),
            steps_per_epoch=len(self.g_data),
            epochs=g_epochs
        )
        self.generator_pre.save(self.g_pre_path)

    def pre_train_discriminator(self,d_epochs=1,d_pre_path=None,d_lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.path_neg)
        G_data = read_gen_data(self.path_neg)
        self.d_data = DiscriminatorGenerator(self.pos_data,G_data,self.B,start_id=start_id,end_id=end_id)
        d_adam = Adam(d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print("Discriminator pre-training")
        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs
        )
        self.discriminator.save(self.d_pre_path)

    def reflect_pre_train(self):
        self.agent.generator.get_pretrain_model(self.generator_pre)
        self.g_beta.generator.get_pretrain_model(self.generator_pre)

    def train(self,steps=1,g_steps=2,d_steps=2,d_epochs=1,
              g_weights_path = "data/save/generator.pkl",
              d_weights_path = "data/save/discriminator.hdf5",
              verbose = 1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam,"binary_crossentropy")
        for step in range(steps):
            for _ in range(g_steps):
                rewards = np.zeros(shape=(self.generate_sample,self.T))
                for t in range(self.T):
                    state = self.env.get_state()
                    c_input = self.env.get_inputs()
                    action = self.agent.act(state,c_input)
                    _next_action,reward,is_episode_end = self.env.step(action=action)
                    self.agent.generator.update(state,action,reward,c_input)
                    reward[:,t] = np.reshape(reward,[self.generate_sample,])
                    if is_episode_end:
                        if verbose:
                            print("Reward: {:.3f},Episode end".format(np.average(rewards)))
                        break
            for _ in range(d_steps):
                self.agent.generator.generate_samples(output_file=self.path_neg)
                G_data = read_gen_data(self.path_neg)
                self.d_data = DiscriminatorGenerator(self.pos_data,G_data,self.B,start_id=start_id,end_id=end_id)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs
                )
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)

    def save(self,g_path,d_path):
        self.agent.save(g_path)
        self.discriminator(d_path)

    def load(self,g_path,d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(get_weights(d_path))

    def generate_txt(self):
        self.agent.generator.generate_samples(output_file=self.path_neg)
















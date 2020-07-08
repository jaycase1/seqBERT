from utils import *
from bert4keras.optimizers import Adam
from bert4keras.models import build_transformer_model
from bert4keras.layers import Loss
from bert4keras.backend import K
from keras.layers import *
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import pickle

config_path = 'G:/seqGAN_BERT/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'G:/seqGAN_BERT/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_model.ckpt'

class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true,y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1],K.floatx())[:,1:]
        y_true = y_true[:,1:]
        y_pred = y_pred[:,:-1]
        loss = K.sparse_categorical_crossentropy(y_true,y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def GeneratePretrain(c_e,g_pre_lr):
    c_in = Input(shape=(1,))
    c = Embedding(2,c_e)(c_in)
    c = Reshape((128,))(c)
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
        keep_tokens = keep_tokens,
        layer_norm_cond = c,
        additional_input_layers = c_in,
    )
    output = CrossEntropy(1)([model.inputs[0],model.outputs[0]])
    model = Model(model.inputs,output)
    model.compile(optimizer=Adam(g_pre_lr))
    return model


def HighWay(x,num_layers=1,activation='relu',name_prefix=''):
    '''
       Layer wrapper function for Highway network
       # Arguments:
           x: tensor, shape = (B, input_size)
       # Optional Arguments:
           num_layers: int, dafault is 1, the number of Highway network layers
           activation: keras activation, default is 'relu'
           name_prefix: str, default is '', layer name prefix
       # Returns:
           out: tensor, shape = (B, input_size)
       '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x


def Discriminator(E,H=64,V=dict_V,dropout=0.1):
    """
        :param V: int, Vocabrary Size
        :param E: Embedding Size
        :param H:hidden Size
        :param dropout: float
        :return:
            input: word ids, shape = (B, T)
                output: probability of true data or not, shape = (B, 1)
        """
    input = Input(shape=(None,),dtype='int32',name="Input")
    out = Embedding(V,E,mask_zero=True,name="Embedding")(input)
    out = LSTM(H)(out)
    out = HighWay(out,num_layers=1)
    out = Dropout(dropout,name="Dropout")(out)
    out = Dense(1,activation="sigmoid",name="FC")(out)

    discriminator = Model(input,out)
    return discriminator


class Generator:
    def __init__(self,generate_samples,maxLen,c_e,g_lr=1e-3,minLen=1,topK=5):
        self.B = generate_samples
        self.maxLen = maxLen
        self.c_e = c_e
        self.start_id = start_id
        self.end_id = end_id
        self.minLen = minLen
        self.topK = topK
        self.g_lr = g_lr
        self.model = self.build_model()

    def build_model(self):
        c_in = Input(shape=(1,))
        c = Embedding(2,self.c_e)(c_in)
        c = Reshape((self.c_e,))(c)
        model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            application='lm',
            keep_tokens=keep_tokens,
            layer_norm_cond = c,
            additional_input_layers = c_in,
        )
        output = model.outputs[0]
        model = Model(model.inputs,output)
        model.compile(optimizer=Adam(self.g_lr),loss=self.loss)
        return model

    def loss(self,y_true,y_pred):
        """
        :param y_pred:
        :param y_true: batchsize * t * (dictV + 1) : np.concatenate(action,reward)
        :return:
        """
        y_pred = y_pred[:,-1,:]
        y_pred = y_pred[:,np.newaxis]
        action_pred = y_pred
        action_true ,reward = y_true[:,:,1:],y_true[:,:,:1]
        reward = K.reshape(reward,(-1,1))
        log_prob = K.log(tf.reduce_mean(action_pred * action_true,axis=-1))
        loss = -log_prob * reward
        return loss

    def update(self,state,action,reward,c_inputs):
        state = state.reshape(self.B,-1)
        reward = reward.reshape(-1,1)
        label = np.concatenate([reward,to_categorical(action,dict_V)],axis=-1)
        segment_ids = np.zeros_like(state)
        label = label[:,np.newaxis]
        self.model.train_on_batch(x=[state,segment_ids,c_inputs],y=label)

    def get_pretrain_model(self,pretrain_model):
        weights = pretrain_model.get_weights()
        self.model.set_weights(weights)

    def save(self,path):
        weights = self.model.get_weights()
        with open(path,"wb") as f:
            pickle.dump(weights,f)

    def load(self,path):
        with open(path,'rb') as f:
            weigths = pickle.load(f)
        self.model.set_weights(weights=weigths)

    def predict(self,state,c_input):
        token_ids = state
        segment_ids = np.zeros_like(token_ids)
        return self.model.predict([token_ids,segment_ids,c_input])[:,-1]

    def sampling_word(self,prob):
        K_indices = prob.argpartition(-self.topK,axis=1)[:,-self.topK:]
        prob = np.take_along_axis(prob,K_indices,axis=1)
        prob /=  prob.sum(axis=1,keepdims=True)
        sample_func = lambda p : np.random.choice(len(p),p=p)
        sample_ids = np.apply_along_axis(sample_func,1,prob)
        sample_ids = sample_ids.reshape((-1,1))
        sample_ids = np.take_along_axis(
            K_indices,sample_ids,axis=1
        )
        return sample_ids

    def sampling_sentence(self,input,min_ends=1):
        output_ids = np.zeros_like(input)
        output_ids[:,:] = self.start_id
        result = []
        for step in range(self.maxLen):
            prob = self.predict(output_ids,input)
            prob /= prob.sum(axis=1,keepdims=True)
            sample_ids = self.sampling_word(prob)
            output_ids = np.concatenate([output_ids,sample_ids],1)
            end_counts = (output_ids == self.end_id).sum(1)
            if(output_ids.shape[1]>=self.minLen):
                flag =  (end_counts ==min_ends)
                if(flag.any()):
                    for ids in output_ids[flag]:
                        result.append(ids)
                    flag =(flag==False)
                    input = input[flag].reshape((-1,1))
                    output_ids = output_ids[flag]
                    end_counts = end_counts[flag]
                    if len(output_ids)==0:
                        break
        for ids in output_ids:
            result.append(ids)
        return result

    def generate_samples(self,output_file,posRate=0.5):
        """
        :param output_file: the path of saving generate text
        :param posRate: pos sentiment rate vs neg sentiment rate
        :return:
        """
        input = np.random.binomial(1,posRate,(self.B,1))
        result = self.sampling_sentence(input)
        results = [tokenizer.decode(ids) for ids in result]
        output_str = ""
        for sentence in results:
            sentence += "\n"
            output_str += sentence
        with open(output_file,'w',encoding='utf-8') as f:
            f.write(output_str)



















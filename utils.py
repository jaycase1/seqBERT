from bert4keras.tokenizers import Tokenizer,load_vocab
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
import numpy as np
import random
from keras.utils import Sequence
from seqGAN.config import getConfig
import pickle
config = getConfig("config.ini")
maxLen = config["max_length"]


dict_path = 'G:/seqGAN_BERT_2/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/vocab.txt'


def get_weights(path):
    with open(path,"rb") as f:
        weights = pickle.load(f)
    return weights

def load_data(filenames):
    """加载数据，并尽量划分为不超过maxlen的句子
    """
    D = []
    if(type(filenames)!=list):
        filenames = [filenames]
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    for filename in filenames:
        print(filename)
        with open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                for t in text_segmentate(text, maxLen - 2, seps, strips):
                    D.append((t, int(label)))
    return D

def read_gen_data(filename):
    G_data = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            G_data.append(line)
    return G_data


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]

class GeneratorPretraingGenerator(DataGenerator):
    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
        for is_end, (text,label) in self.sample(random):
            token_ids,segment_ids = tokenizer.encode(text,maxlen=maxLen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if(len(batch_token_ids)==self.batch_size or is_end):
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids,batch_segment_ids,batch_labels],None
                batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

class DiscriminatorGenerator(Sequence):
    def __init__(self,data_pos,data_neg,batchSize,start_id,end_id,min_count=1,shuffle=True):
        self.data_pos = data_pos
        self.data_neg = data_neg
        self.B = batchSize
        self.start_id = start_id
        self.end_id = end_id
        self.min_count = min_count
        self.shuffle = shuffle
        self.n_data_pos = sum(1 for line in self.data_pos)
        self.n_data_neg = sum(1 for line in self.data_neg)
        self.n_data = self.n_data_neg + self.n_data_pos
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        X,Y = [],[]
        start = idx * self.B + 1
        end = (idx + 1)* self.B + 1
        for i in range(start,end):
            idx = self.indicies[i]

            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = self.data_pos[idx]
            elif is_pos == 0:
                sentence = self.data_neg[idx]
            words,_ = tokenizer.encode(sentence,maxlen=maxLen)
            x = []
            x.extend(words)
            x.append(self.end_id)
            X.append(x)
            Y.append(is_pos)
            X = sequence_padding(X)
            X = np.array(X,dtype=np.int32)
            return (X,Y)


    def next(self):
        if self.idx >=self.len:
            self.reset()
            raise StopIteration
        X,Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X,Y)

    def reset(self):
        self.idx  = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos + 1)
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg + 1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.shuffle:
            random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()

    def __iter__(self):
        return self





token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]','[UNK]','[CLS]','[SEP]'],
)


tokenizer = Tokenizer(token_dict,do_lower_case=True)

# 2,3,0,13584
start_id = tokenizer._token_start_id
end_id = tokenizer._token_end_id
pad_id = tokenizer._token_pad_id
dict_V = tokenizer._vocab_size
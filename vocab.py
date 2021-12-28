# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Vocabulary wrapper"""

from collections import Counter
import argparse
import os
import json
from numpy.core.fromnumeric import shape
import gensim
import en_core_web_sm

annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            line.decode('utf-8')
            captions.append(line.strip())
    return captions

def build_vocab(data_path, data_name, caption_file, threshold):
    nlp = en_core_web_sm.load()
    doc = []
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        text = from_txt(full_path)
       
        for item in text:
            doc.append(nlp(item.decode('utf-8')))
           
        tokens = []        
        for i, item in enumerate(doc):
            for token in item:
                print(token)
                if token.lemma_ == '-PRON-':
                    token.lemma_ = token.orth_ 
                    token.lemma = token.orth 
                if token not in nlp.Defaults.stop_words:
                    tokens.append(token.lemma_ )
                if i % 1000 == 0:
                    print("[%d/%d] tokenized the captions." % (i, len(token)))
                
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(tokens):
        vocab.add_word(word)
    return vocab
    
   
def word2vec(data_path,data_name,caption_file):
    doc = []
    tokens = []
    nlp = en_core_web_sm.load()
    stop_words= nlp.Defaults.stop_words 
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)

        for item in captions:
            doc.append(gensim.utils.simple_preprocess(item))
           
        for i, token in enumerate(doc):
            stopped_tokens = [i for i in token if not i in stop_words]
            tokens.append(" ".join(stopped_tokens))
            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(doc)))
                
    documents = [_text.split() for _text in tokens] 
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=300, 
                                            window=7, 
                                            min_count=10, 
                                            workers=8,)    
    w2v_model.build_vocab(documents)
    words = w2v_model.wv.key_to_index.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)
    w2v_model.train(documents, total_examples=len(documents), epochs=32) 
    w2v_model.save("word2vec.model")
    money = w2v_model.wv.most_similar("money")
    print(money)



def main(data_path, data_name,w2vec):
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=4)
    if w2vec:
        word2vec(data_path, data_name, caption_file=annotations)
    serialize_vocab(vocab, './vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", './vocab/%s_vocab.json' % data_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='f30k_precomp')
    parser.add_argument('--w2vec', default=False,
                        help='train w2vec model')                    
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name,opt.w2vec)

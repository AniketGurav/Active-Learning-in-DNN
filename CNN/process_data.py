import numpy as np
import scipy.sparse as sp
import cPickle
from collections import defaultdict
import sys, re, os, logging, argparse
import pandas as pd
import random

logger = logging.getLogger("personalized.twsent.procdata")

def build_data(fname):
    """
    Loads and process data.
    """
    revs = []
    vocab = defaultdict(float)
    ins_idx = 0
    users = set()


    nonvector_count=0
    vector_count=0
    with open(fname, "rb") as f:
        for line in f:  
            parts = line.strip().split("\t")
            rev = parts[0]
            label = int(parts[1])

            split = int(parts[3])
            words = set(rev.split())
            userEmbedding = []

            # print len(userEmbedding)
            for word in words:
                vocab[word] += 1
            datum  = {"y":label,
                      "text": rev,
                      "num_words": len(rev.split()),
                      "split": split}
            revs.append(datum)
            ins_idx += 1
    max_l = np.max(pd.DataFrame(revs)["num_words"])

    logger.info("finish building data: %d tweets " %(ins_idx))
    logger.info("vocab size: %d, max tweet length: %d" %(len(vocab), max_l))
    return revs, vocab, max_l
    

class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, has_header=False):
        if binary == 1:
            word_vecs = self.load_bin_vec(fname, vocab)
        else:
            word_vecs = self.load_txt_vec(fname, vocab, has_header)
        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs
    
    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
    

if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_fname", help="path/name of pretrained word embeddings file")
    parser.add_argument("--cmty_fname", help="path/name of user community file")
    parser.add_argument("--cmty_col", type=int, default=1, help="column in cmty_fname for specific community detection algorithm")
    args = parser.parse_args()

    fname = "../data/semeval.txt"
    outfname = "../data/semeval.pkl"


    revs, vocab, max_l = build_data(fname)
    w2v_fname = "../data/semeval-all-rmrt-rm5tok-dim300.bin"
    wordvecs = None
    if w2v_fname is not None: # use word embeddings for CNN
        logger.info("loading and processing pretrained word vectors")
        wordvecs = WordVecs(w2v_fname, vocab, binary=1, has_header=False)

    cPickle.dump([revs, wordvecs, max_l], open(outfname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
    

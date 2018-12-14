"""
Docstring
"""
import multiprocessing
import gensim.models.word2vec as w2v
from gensim.models import keyedvectors
from parser_RL import *


# sents = [['first', 'sentence'], ['second', 'sentence']]
# moar_sentences = [['third', 'hurrpa'], ['fourth', 'durrpa']]

# sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]
# vocab = [s.encode('utf-8').split() for s in sentences]

# model = w2v.Word2Vec(vocab, min_count=1)
# model.build_vocab(sentences)
# model.train(moar_sentences, total_examples=model.corpus_count, epochs=model.epochs)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]


num_features = 300
min_word_count = 1
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1

model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

#model = w2v.Word2Vec(min_count=1)
model.build_vocab(sentences)

say_vector = model['say']  # get vector for word
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
a = model.wv['say']
b = model.wv.most_similar(positive=['cat', 'meow'], negative=['dog'])
print(a)
print(b)

#print(model)
#fname = "lol.wv"

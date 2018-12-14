"""
Most of this code was taken and modified from the following tutorials:
https://www.youtube.com/watch?v=pY9EwZ02sXU
https://datascienceplus.com/topic-modeling-in-python-with-nltk-and-gensim/
"""
import sys
import spacy
from sklearn import manifold
import os

spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 3]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


import random
import csv
text_data = []
csv.field_size_limit(sys.maxsize)
with open('dataset/mediumchunk.csv', "r") as f:
    next(f)
    reader = csv.reader(f)
    for line in reader:
        tokens = prepare_text_for_lda(line[9])
        if random.random() > .25:
            text_data.append(tokens)

print(text_data)

import multiprocessing
import gensim.models.word2vec as w2v
from gensim.models import keyedvectors

num_features = 300
min_word_count = 3
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

fname = "model.w2v"

model.build_vocab(text_data)
model.train(text_data, total_examples=model.corpus_count, epochs=model.iter)

if not os.path.exists("trained"):
    os.makedirs("trained")

model.save(os.path.join("trained", "model.w2v"))

#model.save(fname)

tsne = manifold.TSNE(n_components=2, random_state=0)
word_matrix = model.wv.syn0
print("Starting TSNE with multiprocessing.cpu_count() at: ", num_workers)
matrix_2d = tsne.fit_transform(word_matrix)
import matplotlib.pyplot as plt
import pandas as pd
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
                (word, matrix_2d[model.wv.vocab[word].index])
                for word in model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
    )

points.head(10)

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]
    print(points)
    print(slice)

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for _, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

import seaborn as sns

x_bds=[-10.0, 10.0]
y_bds=[-10.0, 10.0]

sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(40, 24))
plot_region(x_bds, y_bds)
plt.show()

    
try:
    print(model.most_similar("disney"))
except Exception:
    print("disney is not in the vocabulary")

try:
    print(model.most_similar("donald"))
except KeyError:
    print("Exception here")
else:
    print("heh do this instead")



    #b = model.wv.most_similar(positive=['economy', 'obama'], negative=['israel'])
#b = model.wv.most_similar(positive=['clinton', 'politics'], negative=['woman'])
#say_vector = model.wv['trump']  # get vector for word
#b = model.wv.similarity('trump', 'clinton')
#b = model.wv.doesnt_match("bitcoins stocks kings money cat cash".split())
#b = model.score(["The fox jumped over a lazy dog".split()])

#print(say_vector)
#print(b)

import sys
import spacy
import nltk
import csv
import random
spacy.load('en')
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
parser = English()
import pickle
import gensim
from gensim import corpora
from gensim.models import TfidfModel


def tokenize(text):
    lda_tokens = []
    #nlp = spacy.load('en_core_web_sm')
    #tokens = nlp(text)
    tokens = parser(text)
    for token in tokens:
        
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            #if token.tag == 'NN' or token.tag == 'NNP':
            lda_tokens.append(token.lower_)
                
        #if token.tag_ == 'NN' or token.tag_ == 'NNP':
        #    lda_tokens.append(token.text.lower())
    return lda_tokens



def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma



def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def csv_read(file_path,text_data):
    csv.field_size_limit(sys.maxsize)
    with open(file_path, "r") as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            #if random.random() > .8:
            tokens = prepare_text_for_lda(unicode(line[9], "utf-8"))
            text_data.append(tokens)


def csv_save(file_path,text_data):
    pickle.dump(text_data,open(file_path,'w'))
    



if __name__ == "__main__":
    #nltk.download('wordnet')
    #nltk.download('stopwords')

    text_data = []

    print("Preparing the text")

    ### Use the following two lines if you want to use new articles ###
    csv_read('dataset/test.csv', text_data)
    #pickle.dump(text_data,open('dataset/Modtest.csv','w'))

    ### Use the following line if you only want to train the network again ###
    ## text_data = pickle.load(file('dataset/Modtest.csv'))

    print("Making the dictionary")
    dictionary = corpora.Dictionary(text_data)

    print("Making the BOW list")
    corpus = [dictionary.doc2bow(text) for text in text_data]

    ### Save the dictionary and Corpus so they can be used later ###
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    print("TF-IDF")
    model = TfidfModel(corpus)
    tfidfCorpus = model[corpus]
    #print(vector)


    print("Training the network")
    NUM_TOPICS = 40
    #ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    #ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=100)
    ldamodel = gensim.models.ldamulticore.LdaMulticore(tfidfCorpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=50)
    ldamodel.save('ldamodel.gensim')

    lda = gensim.models.ldamodel.LdaModel.load('ldamodel.gensim')
    import pyLDAvis.gensim as gensimvis
    import pyLDAvis.gensim
    #lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    #gensimvis.display(lda_display)
    pyLDAvis.display(lda_display)
    pyLDAvis.show(lda_display)
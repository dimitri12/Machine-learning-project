
import sys
import spacy
import nltk
spacy.load('en')
from spacy.lang.en import English
parser = English()

def tokenize(text):
    lda_tokens = []
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(text)
    #tokens = parser(text)
    for token in tokens:
        
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            if token.tag == 'NN' or token.tag == 'NNP':
                lda_tokens.append(token.lower_)
                
        if token.tag_ == 'NN' or token.tag_ == 'NNP':
            lda_tokens.append(token.text.lower())
    return lda_tokens


#nltk.download('wordnet')

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


#nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

print("THINGS ARE HAPPENING")
import csv
import random
text_data = []
csv.field_size_limit(sys.maxsize)
with open('dataset/test2.csv', "r") as f:
    next(f)
    reader = csv.reader(f)
    for line in reader:
        tokens = prepare_text_for_lda((line[9]))
        #print(tokens)
        #if random.random() > .8:
        #print(tokens)
        text_data.append(tokens)




from gensim import corpora
print(text_data)

dictionary = corpora.Dictionary(text_data)



corpus = [dictionary.doc2bow(text) for text in text_data]


import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


import gensim
NUM_TOPICS = 40
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)
ldamodel.save('model5.gensim')


topics = ldamodel.print_topics(num_words=4)
print(topics[0].count)
for topic in topics:
    print(topic)



new_doc = "father father dady mother cake words sun spa But parents are making sacrifices to provide healthy food for their families because they know their kids want and need it. If my kids didn't go to Perea, they wouldn't want to eat vegetables"
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))


'''
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
ldamodel.save('model3.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')



import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)


lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')
lda_display3 = pyLDAvis.gensim.prepare(lda3, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display3)



lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display10)
'''


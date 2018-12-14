import sys
import spacy
import nltk
import gensim
import pickle
from gensim import models, corpora
from LDA_train import prepare_text_for_lda 

print("Let us test the LDA model!")

ldamodel = models.LdaModel.load("ldamodel.gensim")
dictionary = pickle.load(file("dictionary.gensim"))
corpus = pickle.load(file("corpus.pkl"))

topics = ldamodel.print_topics(num_words=6)

for topic in topics:
    print(topic)



#new_doc = "The diplomatically isolated North is almost entirely dependent on trade with China to prop up its impoverished economy, and about three quarters of that trade flows across the winding Yalu River to Dandong father father dady I am dreaming of a very special Christmas where Obama found a fish under a red sock. mother cake words sun spa But parents are making sacrifices to provide healthy food for their families because they know their kids want and need it. If my kids didn't go to Perea, they wouldn't want to eat vegetables"
new_doc = "The annual Star Wars Celebration is the only place Walt Disney Co licensees are selling a new Luke Skywalker action figure Stormtrooper helmets and other coveted merchandise. With an array of new products and exclusive items Disney is not simply rewarding passionate fans. It is also recognizing the role collectors play in stoking excitement around one of its most important franchises Star Wars items were the U.S. toy industry line for 2015 and 2016 with billion in sales over the two years, research firm NPD said. Disney, the world largest entertainment company bought Star Wars producer Lucasfilm in 2012 and began developing new movies in the celebrated science fiction franchise. The company then expanded the range of related products to attract both casual and serious collectors working with licensees on everything from bobbleheads to a Darth Vader figure. The volume of it has been unprecedented said Gus Lopez a Star Wars collector who has five books on the subject. It on a new level Collectors are a key part of Disney Star Wars business says Jim Silver of toy review site TTPM.  NPD reported that about 3 percent of Star Wars sales in 2016 came from collectibles defined as certain types of trading cards action figures and other products for collectors. Silver estimates that the share of sales to collectors is much higher as much as 33 percent to 45 percent, based on industry data about purchase habits, the age of buyers and the types of products bought, such as the   action figures. After acquiring Lucasfilm, Disney and its licensees created a range of new collectibles, including Elite Series action figures, which are die cast and heavier than plastic ones, Silver said. The figures flew off Disney Store shelves when they were released around the December 2015 debut of the movie The Force Awakens Silver said. They couldn keep them in stock he said.  Products come in a variety of price ranges. Bobbleheads from toymaker Funko, selling for have become a hit. Some fans buy every figure made and various poses of the same character said Paul Southern, senior vice president of Star Wars licensing at Disney consumer products and interactive division."
new_doc = prepare_text_for_lda(unicode(new_doc, "utf-8"))
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))



#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
##ldamodel.save('model3.gensim')
#topics = ldamodel.print_topics(num_words=4)
#for topic in topics:
#    print(topic)

#import pyLDAvis.gensim
#lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
#pyLDAvis.display(lda_display)
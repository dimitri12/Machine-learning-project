#Calculates the most important wordes in the input text to later be used to train the machine learning algorithm
import csv
import sys
import operator
import re
from textblob import *
import random

def open_csv():
    csv.field_size_limit(sys.maxsize)
    with open("./dataset/articles1.csv", "r") as f:
        next(f)
        reader = csv.reader(f)
        i = 0
        for row in reader:
            amount_words(row[1],row[9])
            i+=1
            if(i==1):
                return
        

def amount_words(id,article):
    words={}
    article.lower()
    for word in article.split():
        word = word.lower()
        word = re.sub('[^A-Za-z0-9]+', '', word)
        if not word in words:
            words[word]=1
        else:
            words[word]+=1
    sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    important_words(id,sorted_words)

def important_words(id,sorted_words):
    banned = []
    result = []
    result.append(id)
    with open('removed_words.txt', "r") as inputfile:
        for row in csv.reader(inputfile):
            banned.append(row)
    counter = 0
    for word in sorted_words:
        if not word[0] in banned[0]:
            counter += 1
            result.append(word[0])
            if(counter==10):
                break
    print(result)

def save_words(words):
    print("save the words with id in csv file")

##### Under here are implementations using TextBlob #####

def open_csv2():
    csv.field_size_limit(sys.maxsize)
    with open("./dataset/articles1.csv", "r") as f:
        next(f)
        reader = csv.reader(f)
        i = 0
        for row in reader:
            sentences(row[1],row[9])
            #select_tags2(row[1],row[9])
            i+=1
            if(i==1):
                return

def sentences(id, article):
    blob = TextBlob(article)
    for sentence in blob.sentences:
        print(sentence)
        

def select_tags(id,article):
    nouns = list()
    blob = TextBlob(article)
    blob = blob.lower()
    for word, tag in blob.tags:
        if tag == 'NN':
            nouns.append(word.lemmatize())
    
    print ("This text is about...")
    #for item in random.sample(nouns, 5):
    #    print(item)
    words={}
    for word in nouns:
        if not word in words:
            words[word]=1
        else:
            words[word]+=1
    sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    for j in range(0,10):
        print(sorted_words[j])

def select_tags2(id,article):
    nouns = list()
    blob = TextBlob(article)
    blob = blob.lower()
    print(blob.tags)
    print(blob.noun_phrases)

if __name__ == "__main__":
    open_csv2()
    #open_csv()
    #sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    #important_words(sorted_words)
    #i = 0
    #for word in sorted_words:
    #    i+=1
    #    print(word[0])
    #    if(i==200):
    #        break
    #print(sorted_words[:200])
    #print(words["Obama"])
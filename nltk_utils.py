import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for s in tokenized_sentence:
        for i, w in enumerate(all_words):
            if w == s:
                bag[i] = 1
    return bag

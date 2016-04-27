__author__ = 'Shubham'
import nltk

#nltk.download()
text=nltk.word_tokenize("Hasn't he lost it?")

x= nltk.pos_tag(text)
print x
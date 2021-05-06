
Introduction

Our group was working on various approaches to implement the personality trait for the user input. 
In the initial phase, the group focused on implementing javascript language using Node.js 
to implement the chat bot using  Microsoft Azure Bot Services as cloud chat bot hosting. 
With many challenges to implement the Natural Language Processing, 
Our group decided to switch the language and platform into Python and PyCharm. 
Python being a great language from machine learning, it was a lot easier to implement the NPL. 
The data input from the user is forwarded into google spread. The data from google spread was imported into pycharm. 
Import Spacy in python helped to parse all the user words into verbs, nouns, adjectives, and adverbs. 
Finally, Word2Vec algorithm for NLP is implemented to calculate the Neuroticism score of the user 
based on the word in the chat bot. The score was then forwarded into the chatbot user interface 
so that the user knows what is the score for neuroticism. 

For our implementation, we choose to focus on one personality trait because of time 
and resource restrictions due to recent circumstances. The Big Five personality trait we trained our NLP program on was Neuroticism. 
Yarkoni's Personality in 100,000 Words gave us an easily comparable database of base-case words 
to use with a Word2Vec model trained by the Wikipedia page on Neuroticism to capture words 
related to our chosen scorable database and provide an overall Neuroticism score. 
We also utilized Landbio.io, which is a simple non-program oriented chat-bot interface 
that allowed us to create a simple chat bot and focus more on it’s integration with NLP with Word2Vec 
and user text parsing with Spacy. Lastly, getting our user data from Landbot.io 
to our NLP program was accomplished through a Google Sheet update/retrieval integration.



pip install -U spacy
python -m spacy download en_core_web_sm
import spacy

# for stackhouse, word to  vec 
pip install beautifulsoup4
pip install lxml

import bs4 as bs
import urllib.request 
import re 
import nltk

pip install gensim

pip install python-Levenshtein


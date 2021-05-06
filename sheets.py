import time


#Gspread will import the google shee
import gspread
import spacy





#Authorizing google sheets and pulling the data from the Google sheets API into the IDE part
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/spreadsheets"
    ,"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

# setting up the credentials
creds = ServiceAccountCredentials.from_json_keyfile_name("Credential.json", scope)
#this will authorize the credential
client = gspread.authorize(creds)

sheet = client.open("CS4000 Chat Bot Data").sheet1

data = sheet.get_all_records()
row = sheet.row_values(3)
col = sheet.col_values(2)
cell = sheet.cell(1,2).value



# #This part of works on parsing the data that is imported from the spacy
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
#CompWords = row[10]

text = (row[7]+ " "+ row[8]  )

doc = nlp(text)


# Analyze syntax

noun_phrases = [chunk.text for chunk in doc.noun_chunks]
print("Noun Phrases: ", noun_phrases)

verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
print("Verbs: ", verbs)
#print("Comparing words: ", CompWords)
adj = [token.lemma_ for token in doc if token.pos_ == "ADJ"]
#float_adj = float(adj)
adj_str = " ".join(adj)
adv = [token.lemma_ for token in doc if token.pos_ == "ADV"]
adv_str = " ".join(adv)
print ("Adjective: ", adj)
print("Adverb: " , adv)
user_words = adj +adv






# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)




#Similar Words part

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from parse import *


#article_text = adj_str



#
# parsed_article = parse(text)
#
# paragraphs = parsed_article.find_all('p')
#
# article_text = ""
#
# for p in paragraphs:
#     article_text += p.text



#take the article for the URL
scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Neuroticism')
article = scrapped_data .read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text

processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

list_of_comparisons = ["awful", "sick", "terribly", "cranky", "stress", "feeling",
                       'stressful', 'myself', 'though', 'feel', 'sweater', 'scenario',
                       'ashamed', 'feels', 'spoiled', 'sick', 'yay', 'possibly',
                       'completely', 'though', 'desperately', 'pregnancy', "shouldn't",
                       'lazy', 'refuse', 'irony', 'pretend', 'horrible', 'harsh', 'stupid',
                       'uncomfortable', 'though', 'drugs', 'guardian', 'sizes', 'messy',
                       'silly', 'easier', 'opinions', 'lazy', 'shorter', 'expecting', 'fit',
                       'instead', 'realistic' , 'lazy', 'awful', 'uncomfortable', 'lately', 'myself', 'though']

all_words.append(adj)
all_words.append(adv)
all_words.append(list_of_comparisons)


# Removing Stop Words
from nltk.corpus import stopwords
# for i in range(len(all_words)):
#     all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]



from gensim.models import Word2Vec
word2vec = Word2Vec(all_words, min_count=1)

#vocabulary = word2vec.wv.vocab
#print(vocabulary)
all_vector = word2vec.wv[user_words]

neuro_score = 0


#list_of_comparisons += all_words

total_neuro_score = 0
count = 0
for w in list_of_comparisons:
    try:
        v1 = word2vec.wv[w]
        com_words = word2vec.wv.cosine_similarities(v1, all_vector)
        #rounded_com_words = round(com_words,2)
        #print(com_words)
        sum_comp_word = 0
        for i in com_words:
            if(i > 0):

                sum_comp_word += round(i,2)
            #print(sum_comp_word)
        if sum_comp_word > 0.4:
            #print(sum_comp_word)
            total_neuro_score += sum_comp_word
            count += 1

    except:
        pass

try:
    avg_neuro_score = total_neuro_score / count
    if(avg_neuro_score<0.6):
        print("The user is not neurotic.")
    elif (avg_neuro_score>0.6 and avg_neuro_score <= 0.7):
        print("The user is slightly neurotic.")
    print("Neuroticism score of the User: ", avg_neuro_score)
except:
    print("The user is not neurotic.")



#sim_words = word2vec.wv.most_similar("immature")


#print(all_words)
#print(sim_words)

# Writing results back to Google Sheets

#  while True:
#      i = 1;
#      val = sheet.cell(i, 9).value
#      if(val != ""):
#          i += 1
#      else:
#          time.sleep(60)
#          sheet.update_cell(i, 9, avg_neuro_score)
#          break;
sheet.update_cell(2, 10, avg_neuro_score)


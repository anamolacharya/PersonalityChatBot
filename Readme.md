
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

//{
//  "type": "service_account",
//  "project_id": "mimetic-codex-309523",
//  "private_key_id": "4b649ecf12a1e56a37e1a2005ea714e737f873ca",
//  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCNw9SW64IBq9Ek\n0Wn8JdhJ7eeFIowQSpfNQ0j0QjooZzB43gAEUbg5AaowZFFPGQTx+rzjyAAXskqy\nTDKlvt9qUuDPkPICqCvV0plUyZPJg6dbUJk3oHtIx2bHm7g7mim1TXG2iC1/1+tU\naY8uKaAIW2EXyRgDAHjVQiTumwLbZKuvoZyTL9mLPlBFTWVUupl3l/pvlN3HLLER\nX4vOT6UoapkPuh+WOfWSUGU4dtJCA63+RxtgH1+DfN3ApMni+XMxtEIInYO/FxTd\nkN2ISKjj5rf/if0laUxpcAvSAnqgjIClDSiogGc0XZKp1PyI+Ycer/f1o0kmTo/j\naR2L8lBfAgMBAAECggEAAVXyh1eBupVNW6zzdDJBFvU8ZLc+HZsuUftL0S1ckevi\n+4iP9Hp2o2beHSWc1IN2VfdYV0cXpSGDbamlIoWpYj2Uq7AAb8D45MNeuXtKvq/U\nYFpijzEeaEGkuMXBVyJtOS2ZIDCpSRaWO43hEPzIaru9rvUwmjqjaNL+ORdR0BJf\nqjwr+86L519jKqokcYF2St7e6hLsV9zC0skX+6Q9DhAq2YDK7fMEt8ZzjjwkmMM2\nOVcaZYawrCKLj7xNIQTExvAM1X3deloCk2u8jfb5QH0MjPXB3rIWO0xsspoc3Udd\nIM83+rBaH5OaUPtlTDj1PFV6osYdLQ39uP/yu9o20QKBgQDEyEFY2TaxIzN3326n\nhifza7ulsmJEVHOq4LAjppAbg93KPpOLc3DZabEC0sAdxaSkRCo1kXCmF1O51w4a\nuAK4sJc+9XYpnefi1HnrWcOEiLJblFn7WniNVII+D1DIKOWuzTW8lwgKe94sQqaa\nIfPdjwFtKfVMdph/1kq+0brY7wKBgQC4bSTjwt/IDSE3UGQwlTGDtLt2IqDpIqqk\n8/v97TOJhIkc5Mi0lGaUQ3/yYoWlzr5lU5PIUab47MvHjZwhQCZINBLzGKr+AQ9N\noiVCUYo8ALPTmvF6CflGfOwlo6362w4WYipsunq4XxAhjGev4QIn4rcNskl5Mjul\n7IDxJ6+fkQKBgQCBj0ufANyGgiOn3/7N44E4Po08ihcy39uL/QVbY5Xr18VWHB8u\nqGH7cx/tOO7uayt8T7jurgRaBm/EorgRlWeNTA84j4ot2l5LNRPUhbQ59Xpg22rn\nF+jZPHPIAnNwZaTbkxa3RUUxCd78iyF/x6z1CeupgP+VSVwchu2Ndy6rFwKBgE0G\n6N1nyudW9ISRwwa3iVKk7ZbNp783h2YVsS3BIEFTZaD3vQwO3zkVaB7oH0G9M7BG\nU/bag457+DCEaK1KibKmbTOzHdewwZ9/FWi5fa7J7FF46Vo7SC20hzzBPC0FyMB4\nh5eZ2x+eNLKOXdALfkcCXcoOqLlBzb/jI4eVN7jBAoGACc9upd4eNtn8C8W/Vi9L\nmO6bv7ZCiuUlw0jGbzzJfIcqKa1aUSsQrpa6aG6SSfov+60Je4TuXdjp+iYGyM0V\nxklLtrRY77AdMjyGRbHwxcx89tiP1NyKOYR9sMYt8TEMCofq1c0/2ymXOvdyY//O\n3R+Coy18tIBDOV/mYuVxxq4=\n-----END PRIVATE KEY-----\n",



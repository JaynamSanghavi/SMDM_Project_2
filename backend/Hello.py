import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import re
import nltk
from bs4 import BeautifulSoup
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from flask import Flask
app = Flask(__name__)

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def cleaning(data):
  preprocessed_reviews = []
  eng_stopwords = set(stopwords.words('english'))
  for sentance in tqdm(data['review'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in eng_stopwords)# removing stopwords and 
                                                                                           #converting into lower case
    preprocessed_reviews.append(sentance.strip())
  data['review']=preprocessed_reviews
  return data

@app.route('/')
def hello_world():
   return "Hello World"

@app.route('/dataPreprosessing')
def preProcess():
    data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
    #removing duplicate rows
    print('REMOVING DUPLICATE ROWS....')
    data=data.drop_duplicates(subset=['review'], keep='first', inplace=False)
    data.shape
    data_after_cleaning=cleaning(data) #dataframe after cleaning
    data_after_cleaning.to_csv('after_cleaning.csv')# saving csv file after cleaning
    return data_after_cleaning.to_string()
   

if __name__ == '__main__':
   app.run()
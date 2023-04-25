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

def stemmer(data):
  review_clean_ps = []
  ps = PorterStemmer()
  for sentance in tqdm(data['review'].values):
    ps_stems = []
    for w in sentance.split():
      if w == 'oed':
        continue
      ps_stems.append(ps.stem(w))  
     
    review_clean_ps.append(' '.join(ps_stems))   
  data['review']=review_clean_ps
  return data

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def lemmatization(data):
  review_clean_wnl = []
  wnl = WordNetLemmatizer()
  for sentance in tqdm(data['review'].values):
     wnl_stems = []
     token_tag = pos_tag(sentance.split())
     for pair in token_tag:
       res = wnl.lemmatize(pair[0],pos=get_wordnet_pos(pair[1]))
       wnl_stems.append(res)
     review_clean_wnl.append(' '.join(wnl_stems))
  data['review']=review_clean_wnl
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
   
@app.route('/predictUsingLogisticRegression')
def predictUsingLR():
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  data_stemmer=stemmer(dataClean)
  print("Doing lemma\n")
  data_lemma=lemmatization(data_stemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  # splitting into train and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  #tfidf
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  text_tfidf = tfidfvectorizer.fit(X_train.values) #fitting
  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  lr= LogisticRegression(C= 3.727593720314938)
  lr.fit(X_train_tfidf,  y_train)
  acc = (accuracy_score(y_test,lr.predict(X_test_tfidf)))
  print("Accuracy: ",acc)
  a = ["This is a great movie"]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = lr.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"

@app.route('/predictUsingNaiveBayesClassifier')
def predictUsingNBC():
  data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  data_stemmer=stemmer(dataClean)
  print("Doing lemma\n")
  data_lemma=lemmatization(data_stemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  # splitting into train and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  #tfidf
  print("Doing tfidf\n")
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  text_tfidf = tfidfvectorizer.fit(X_train.values) #fitting

  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  print("Doing fitting\n")
  navie_clf=MultinomialNB(alpha=1, class_prior=[0.5, 0.5], fit_prior=True)
  navie_clf.fit(X_train_tfidf, y_train)
  acc = accuracy_score(y_test,navie_clf.predict(X_test_tfidf))
  a = ["This is a great movie"]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = navie_clf.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"

@app.route('/predictUsingSVM')
def predictUsingSVM():
  data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  data_stemmer=stemmer(dataClean)
  print("Doing lemma\n")
  data_lemma=lemmatization(data_stemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  # splitting into train and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  #tfidf
  print("Doing tfidf\n")
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  text_tfidf = tfidfvectorizer.fit(X_train.values) #fitting

  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  print("Doing model\n")
  svm=SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  svm.fit(X_train_tfidf, y_train)
  acc = accuracy_score(y_test,svm.predict(X_test_tfidf))
  print("Accuracy: ",(accuracy_score(y_test,svm.predict(X_test_tfidf))))
  a = ["This is a great movie"]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = navie_clf.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"

if __name__ == '__main__':
   app.run()
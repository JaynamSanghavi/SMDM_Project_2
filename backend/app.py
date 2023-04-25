import numpy as np
import pandas as pd

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from flask import Flask, request
app = Flask(__name__)


#decosntructing the words like won't to will not

def decontracted(words):
    words = re.sub(r"won't", "will not", words)
    words = re.sub(r"can\'t", "can not", words)
    words = re.sub(r"n\'t", " not", words)
    words = re.sub(r"\'re", " are", words)
    words = re.sub(r"\'s", " is", words)
    words = re.sub(r"\'d", " would", words)
    words = re.sub(r"\'ll", " will", words)
    words = re.sub(r"\'t", " not", words)
    words = re.sub(r"\'ve", " have", words)
    words = re.sub(r"\'m", " am", words)
    return words

#removing the urls, tags, decontructing the word, number, special character and stop words from the data
def preProcessing(data):
  preProcessedReviews = []
  englishStopWords = set(stopwords.words('english'))
  for comments in tqdm(data['review'].values):
    comments = re.sub(r"http\S+", "", comments)
    comments = decontracted(comments)
    comments = BeautifulSoup(comments, 'lxml').get_text()
    comments = re.sub("\S*\d\S*", "", comments).strip()
    comments = re.sub('[^A-Za-z]+', ' ', comments)
    comments = ' '.join(e.lower() for e in comments.split() if e.lower() not in englishStopWords)
    preProcessedReviews.append(comments.strip())
  data['review']=preProcessedReviews
  return data

#stemming the data
def stemmer(data):
  cleanComment = []
  ps = PorterStemmer()
  for comments in tqdm(data['review'].values):
    psStems = []
    for w in comments.split():
      if w == 'oed':
        continue
      psStems.append(ps.stem(w))  
     
    cleanComment.append(' '.join(psStems))   
  data['review']=cleanComment
  return data

def getWordnetPos(treebankTag):

    if treebankTag.startswith('J'):
        return wordnet.ADJ
    elif treebankTag.startswith('V'):
        return wordnet.VERB
    elif treebankTag.startswith('N'):
        return wordnet.NOUN
    elif treebankTag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#lemmatizing the data
def lemmatization(data):
  cleanCommentWnl = []
  wnl = WordNetLemmatizer()
  for comments in tqdm(data['review'].values):
     wnlStems = []
     token_tag = pos_tag(comments.split())
     for pair in token_tag:
       res = wnl.lemmatize(pair[0],pos=getWordnetPos(pair[1]))
       wnlStems.append(res)
     cleanCommentWnl.append(' '.join(wnlStems))
  data['review']=cleanCommentWnl
  return data


@app.route('/')
def hello_world():
   return "Hello World"

#route for pre processing the data, it would be done first time only. Not need to run as clean csv is uploaded on github.
@app.route('/dataPreprosessing')
def preProcess():
    data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
    data=data.drop_duplicates(subset=['review'], keep='first', inplace=False)
    dataAfterCleaning=preProcessing(data)
    dataAfterCleaning.to_csv('after_cleaning.csv')
    return dataAfterCleaning.to_string()

#method predicting if the user comment is negative or positive using Linear Regression
def predictUsingLR(comment):
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  dataStemmer=stemmer(dataClean)
  print("Doing lemma\n")
  dataLemmatizing=lemmatization(dataStemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  textTfidf = tfidfvectorizer.fit(X_train.values)
  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  lr= LogisticRegression(C= 3.72)
  lr.fit(X_train_tfidf,  y_train)
  acc = (accuracy_score(y_test,lr.predict(X_test_tfidf)))
  print("Accuracy: ",acc)
  a = [comment]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = lr.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"


#method predicting if the user comment is negative or positive using Naive Bayes Classifier
def predictUsingNBC(comment):
  data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  dataStemmer=stemmer(dataClean)
  print("Doing lemma\n")
  dataLemmatizing=lemmatization(dataStemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  print("Doing tfidf\n")
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  textTfidf = tfidfvectorizer.fit(X_train.values) #fitting

  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  print("Doing fitting\n")
  navie_clf=MultinomialNB(alpha=1, class_prior=[0.5, 0.5], fit_prior=True)
  navie_clf.fit(X_train_tfidf, y_train)
  acc = accuracy_score(y_test,navie_clf.predict(X_test_tfidf))
  print("Accuracy: ",acc)
  a = [comment]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = navie_clf.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"

#method predicting if the user comment is negative or positive using Support Vector Machine
def predictUsingSVM(comment):
  data = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/dataset/IMDBDataset.tsv",header=0, delimiter="\t", quoting=3)
  dataClean = pd.read_csv("https://raw.githubusercontent.com/JaynamSanghavi/SMDM_Project_2/master/backend/after_cleaning.csv")
  print("Doing stemmer\n")
  dataStemmer=stemmer(dataClean)
  print("Doing lemma\n")
  dataLemmatizing=lemmatization(dataStemmer)
  X=dataClean['review']
  Y=dataClean['sentiment']
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)
  print("Doing tfidf\n")
  tfidfvectorizer = TfidfVectorizer(min_df=10,max_features=5000)
  textTfidf = tfidfvectorizer.fit(X_train.values)
  X_train_tfidf =tfidfvectorizer.transform(X_train.values) 
  X_test_tfidf =tfidfvectorizer.transform(X_test.values)
  print("Doing model\n")
  svm=SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  svm.fit(X_train_tfidf, y_train)
  acc = accuracy_score(y_test,svm.predict(X_test_tfidf))
  print("Accuracy: ",acc)
  a = [comment]
  a_tfidf =tfidfvectorizer.transform(a)
  p_answer = svm.predict(a_tfidf)
  if p_answer[0] == 0:
      return "Negative"
  else:
      return "Positive"

#this route will help to predict the if the user comment is positive or negative
@app.route('/predictComment', methods=['POST']) 
def predictComment():
    comments = request.form.get("comments")
    return predictUsingLR(comments)


if __name__ == '__main__':
   app.run()
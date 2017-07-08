"""
Title: Movie Review Sentiment Analysis
Author: Abhinav Thukral
Description: Implemented text analysis using machine learning models to classify movie review sentiments as positive or negative.
"""

#Importing Essentials
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

path = 'data/opinions.tsv'
data = pd.read_table(path,header=None,skiprows=1,names=['Sentiment','Review'])
X = data.Review
y = data.Sentiment
#Using CountVectorizer to convert text into tokens/features
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
#Using training data to transform text into counts of features for each message
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

#Accuracy using Naive Bayes Model
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using Logistic Regression Model
LR = LogisticRegression()
LR.fit(X_train_dtm, y_train)
y_pred = LR.predict(X_test_dtm)
print('\nLogistic Regression')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using SVM Model
SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Accuracy using KNN Model
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train_dtm, y_train)
y_pred = KNN.predict(X_test_dtm)
print('\nK Nearest Neighbors (NN = 3)')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#Naive Bayes Analysis
tokens_words = vect.get_feature_names()
print('\nAnalysis')
print('No. of tokens: ',len(tokens_words))
counts = NB.feature_count_
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(tokens_words)-positives)
#Check positivity/negativity of specific tokens
token_search = ['awesome']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
#Analyse False Negatives (Actual: 1; Predicted: 0)(Predicted negative review for a positive review) 
print(X_test[ y_pred < y_test ])
#Analyse False Positives (Actual: 0; Predicted: 1)(Predicted positive review for a negative review) 
print(X_test[ y_pred > y_test ])

#Custom Test: Test a review on the best performing model (Logistic Regression)
trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
LR_complete = LogisticRegression()
LR_complete.fit(X_dtm, y)
#Input Review
print('\nTest a custom review message')
print('Enter review to be analysed: ', end=" ")
test = []
test.append(input())
test_dtm = trainingVector.transform(test)
predLabel = LR_complete.predict(test_dtm)
tags = ['Negative','Positive']
#Display Output
print('The review is predicted',tags[predLabel[0]])

"""
Output:

Naive Bayes
Accuracy Score: 98.9161849711%
Confusion Matrix:
[[586  12]
 [  3 783]]

Logistic Regression
Accuracy Score: 99.3497109827%
Confusion Matrix:
[[593   5]
 [  4 782]]

Support Vector Machine
Accuracy Score: 99.0606936416%
Confusion Matrix:
[[592   6]
 [  7 779]]

K Nearest Neighbors (NN = 3)
Accuracy Score: 98.6994219653%
Confusion Matrix:
[[589   9]
 [  9 777]]

Analysis
No. of tokens:  294
No. of positive tokens:  143
No. of negative tokens:  151

Search Results for token/s: ['awesome']
      Token  Positive  Negative
11  awesome     896.0       1.0

Test a custom review message
Enter review to be analysed:  I really appreciate the details of the movie. It was an awesome experience.
The review is predicted Positive
"""

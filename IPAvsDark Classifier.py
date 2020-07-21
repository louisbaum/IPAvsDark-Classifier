# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:22:56 2020

@author: Louis
"""



#import general tools
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os

#import text handling
from wordcloud import STOPWORDS
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,roc_curve


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


import pydotplus
from IPython.display import Image


os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


stop = set(text.ENGLISH_STOP_WORDS).union(set(STOPWORDS))
stop = stop.union({'','beer','br','br br','of','hint','note','much','pours','nice','smell','glass','one','style','just','good','like','used','on tap','pours',
                   'taste','\d','thi','ha','bit','doe','oz','ne', 'ipa','porter','stout','ipas','href','cdn','cgi','l','email','protection','class','cf','data','cfemail'})

datapath = '\\beerdatacleaned.csv'


#perform some additional data cleaning - drop beers with fewer than 5 reviews, remove punctuation and stopwords
beerdata = pd.read_csv(datapath)
beerdata.drop('Unnamed: 0',axis=1,inplace = True)
beerdata.drop(index = beerdata.index[beerdata['num_reviews']<5],inplace=True)
beerdata.reset_index(inplace = True,drop = True)

#Crude removal of stopwords removes the "end" of contactions
beerdata['text'] = beerdata['text'].apply(lambda x: ' '.join([item for item in x.split(' ') if item not in stop and len(x)>2]))


#select specific styles for comparison
ipadata = beerdata.iloc[beerdata.index[beerdata['stylegroup']=='ipa']]
porterdata = beerdata.iloc[beerdata.index[beerdata['stylegroup']=='porter']]
stoutdata = beerdata.iloc[beerdata.index[beerdata['stylegroup']=='stout']]


#assemble test data and split in to training and testing subsections
testdata = pd.concat([ipadata,porterdata,stoutdata])
xtrain,xtest,ytrain,ytest = train_test_split(testdata,testdata['stylegroup']=='ipa',test_size=0.5,random_state=10)

#train model on numeric features
DTnum  = DecisionTreeClassifier(random_state=3,max_depth = 3)
DTnum = DTnum.fit(xtrain[['abv', 'avg_rating','ba_score', 'wants', 'num_reviews']], ytrain)
ypred = DTnum.predict(xtest[['abv', 'avg_rating','ba_score', 'wants', 'num_reviews']])
score=accuracy_score(ytest,ypred)
print(confusion_matrix(ytest,ypred))
print(roc_auc_score(ytest,ypred))

ypredprob2 = DTnum.predict_proba(xtest[['abv', 'avg_rating','ba_score', 'wants', 'num_reviews']])[:,1]
fpr, tpr, thresholds=roc_curve(ytest,ypredprob2)


# simple tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = None)
tfidf_train = tfidf_vectorizer.fit_transform(xtrain['text'])
tfidf_test = tfidf_vectorizer.transform(xtest['text'])

#train second model on text tokens
DTtext  = DecisionTreeClassifier(random_state=7,max_depth = 3)
DTtext = DTtext.fit(tfidf_train, ytrain)
ypred2 = DTtext.predict(tfidf_test)
score=accuracy_score(ytest,ypred2)
print(confusion_matrix(ytest,ypred2))
print(roc_auc_score(ytest,ypred2))

ypredprob2 = DTtext.predict_proba(tfidf_test)[:,1]
fpr2, tpr2, thresholds=roc_curve(ytest,ypredprob2)


#compare ROC curves for both models
plt.figure()
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label = 'DT only Numeric Features')
plt.plot(fpr2,tpr2,label = 'DT only Term Features')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

terms = tfidf_vectorizer.get_feature_names()
termsdf = pd.DataFrame(terms)

#ugly plot numeric model
plt.figure(figsize = [15,8])
tree.plot_tree(DTnum,feature_names = ['abv', 'avg_rating','ba_score', 'wants', 'num_reviews'],proportion=True,class_names = ['Porter or Stout', 'IPA']);

#ugly plot text model
plt.figure(figsize = [15,8])
tree.plot_tree(DTtext,feature_names = terms,filled = True,rounded = True,proportion=True,class_names = ['Porter or Stout', 'IPA']);

#nice plot text model
texttree = tree.export_graphviz(DTtext,
                               filled = True,
                               out_file=None,
                               feature_names = terms,
                               #proportion=True,
                               rounded=True,
                               class_names = ['Porter or Stout', 'IPA']
                               )
graph = pydotplus.graph_from_dot_data(texttree)
colors = ('saddlebrown','khaki')
nodes = graph.get_node_list()

for n in nodes:
    if n.get_label():
        values = [int(ii) for ii in n.get_label().split('value = [')[1].split(']')[0].split(',')]
        values = [int(255 * v / sum(values)) for v in values]
        n.set_fillcolor(colors[np.argmax(values)])
Image(graph.create_png())
graph.write_png('texttree.png')


#nice plot numeric model
Numtree = tree.export_graphviz(DTnum,
                               filled = True,
                               out_file=None,
                               feature_names = ['abv', 'avg_rating','ba_score', 'wants', 'num_reviews'],
                               #proportion=True,
                               rounded=True,
                               class_names = ['Porter or Stout', 'IPA']
                               )
graph = pydotplus.graph_from_dot_data(Numtree)

colors = ('saddlebrown','khaki')
nodes = graph.get_node_list()

for n in nodes:
    if n.get_label():
        values = [int(ii) for ii in n.get_label().split('value = [')[1].split(']')[0].split(',')]
        values = [int(255 * v / sum(values)) for v in values]
        n.set_fillcolor(colors[np.argmax(values)])
Image(graph.create_png())
graph.write_png('NumTree.png')




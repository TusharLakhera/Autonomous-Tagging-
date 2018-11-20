# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:57:53 2018

@author: Naman Kalra
"""

import pandas as pd
from collections import Counter
import string
import re
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import unicodedata
import inflect
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
Questions=pd.read_csv('Questions.csv',encoding='latin-1')
TagData=pd.read_csv('Tags.csv',encoding='latin-1')


def text_clean(text):
    global Blank
    Blank=''
    if not isinstance(text,str):
        return text
    text=re.sub('<pre><code>.*?</pre></code>',Blank,str(text))
    def clean_link(match):
        return Blank if re.match('[a-z]+://',match.group(1)) else match.group(1)
   
    text = re.sub('<a[^>]+>(.*)</a>',clean_link,str(text))
    return re.sub('<[^>]+>',Blank,str(text))
def HTML_ClEAN(text):
        
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text
        
   # for removing  unnecessary code snippets, ,links, URL...
def remove_CodeSnippet(text):
      
    return re.sub('<pre><code>.*?</code></pre>', '', str(text))
   
    #replacing paragraph and next line headers with a blank string
#implementing the De-noise Functions to clean the SampleData
def De_noise(text):
    text=  HTML_ClEAN(text)
    text= remove_CodeSnippet(text)
    return text

        #Non-Ascii Words are ignored for better accuracy purpose        
def is_Non_Ascii(ProcessedBodyData):
    NewProcessedBodyData = []
    for word in ProcessedBodyData:
       temp = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
       NewProcessedBodyData.append(temp)
    return NewProcessedBodyData

#converting everyword to lowercase to remove redundancy for ex-is & IS
def Case_lower(ProcessedBodyData):
    NewProcessedBodyData = []
    for word in ProcessedBodyData:
        Temp = word.lower()
        NewProcessedBodyData.append(Temp)
    return NewProcessedBodyData

#removing Punctuation like,0-;] for better data quality
def TextClean(ProcessedBodyData):
    NewProcessedBodyData = []
    for word in ProcessedBodyData:
        temp = re.sub(r'[^\w\s]', '', str(word))
        if  NewProcessedBodyData != '':
             NewProcessedBodyData .append(temp)
    return  NewProcessedBodyData 

#removing Numbers for better tag prediction
def Number_Removal(ProcessedBodyData):
    use = inflect.engine()
    NewProcessedBodyData = []
    for word in ProcessedBodyData:
        if word.isdigit():
          temp  = use.number_to_words(word)
          NewProcessedBodyData.append(temp)
        else:
            NewProcessedBodyData.append(word)
    return NewProcessedBodyData

#filtering out StopWords to before processing natural data
def StopWord_Removal(ProcessedBodyData):
    
     NewProcessedBodyData = []
     for word in ProcessedBodyData:
        if word not in stopwords.words('english'):
           NewProcessedBodyData.append(word)
     return  NewProcessedBodyData



def WordProcessing(Body_word):

    Body_word=is_Non_Ascii(Body_word)
    Body_word=Case_lower(Body_word)
    Body_word=TextClean(Body_word)
    Body_word=Number_Removal(Body_word)
    Body_word=StopWord_Removal(Body_word)

    return Body_word

Questions['Text']=Questions['Body'].apply(text_clean).str.lower()

TagData.Tag.nunique()
MostCommonTagCount=Counter(list(TagData.Tag)).most_common(40)
print(MostCommonTagCount)

TagData = TagData[(TagData.Tag == 'javascript') | (TagData.Tag == 'java') | (TagData.Tag == 'c#') | (TagData.Tag =='php') | (TagData.Tag =='android') | (TagData.Tag == 'jquery') | (TagData.Tag == 'python') | (TagData.Tag == 'html') | (TagData.Tag == 'c++') | (TagData.Tag == 'windows')|  (TagData.Tag == 'ios')]

TagData.head()
TextandTags=TagData.merge(Questions,on='Id')
TextandTags.Tag
TextandTags.Text

UnnecessaryColumns=['Id','OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title', 'Body']
TextandTags=TextandTags.drop( UnnecessaryColumns,axis=1,inplace=False)
Categories = TextandTags['Tag'].unique()
print(Categories)

graph = plt.figure(figsize=(11,6))
TextandTags.groupby('Tag').Text.count().plot.bar(ylim=0)
plt.show()

TextandTags = pd.DataFrame(TextandTags)
BalTextandTags = TextandTags.groupby('Tag')
BalTextandTags = pd.DataFrame(BalTextandTags.apply(lambda x: x.sample(BalTextandTags.size().min()).reset_index(drop=True)))
BalTextandTags.head()
BalTextandTags.Tag

BalTextandTags.to_csv("output.csv", index=False)

#BalTextandTags['Text']=BalTextandTags['Text'].apply(De_noise)
#BalTextandTags['Text']=BalTextandTags['Text'].apply(WordProcessing)
BalTextandTags.Text=BalTextandTags.Text.apply(lambda x:x.replace('"','').replace("\n","").replace("\t",""))
BalTextandTags.Text[0]

MostCommonTagCount=Counter(list(BalTextandTags.Tag)).most_common(11)
print(MostCommonTagCount)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(BalTextandTags['Text'],BalTextandTags['Tag'],random_state=42,
                                               test_size=0.2,shuffle=True)



def Convert_to_MB(Dataset):
    Result=sum(len(s.encode('utf-8'))for s in Dataset)/ 1e6
    return Result
Train_MB_size=Convert_to_MB(X_train)
Test_MB_size=Convert_to_MB(X_test)
print("%d documents - %0.3fMB (training set)" % (
    len(X_train), Train_MB_size))
print("%d documents - %0.3fMB (test set)" % (
    len(X_test),Test_MB_size))
print("%d Categories" % len(Categories))

from optparse import OptionParser
options = OptionParser()
options.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
options.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

import sys
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = options.parse_args(argv)
if len(args) > 0:
    options.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
options.print_help()
print()



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from time import time
#feature extraction using sparse vectorizer
tnought = time()
if opts.use_hashing:
    Vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train_new = Vectorizer.transform(X_train)
else:
    Vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train_new = Vectorizer.fit_transform(X_train)
TimeTakenTrain = time()-tnought
print("done in %fs at %0.3fMB/s" % (TimeTakenTrain, Train_MB_size / TimeTakenTrain))
print("n_samples: %d, n_features: %d" % X_train_new.shape)

#using the vectoriser for the test data now
tnought=time()
X_test_new=Vectorizer.transform(X_test)
TimeTakenTest= time()-tnought
print("done in %fs at %0.3fMB/s" % (TimeTakenTest, Test_MB_size / TimeTakenTest))
print("n_samples: %d, n_features: %d" % X_test_new.shape)


#SVM
from sklearn.svm import LinearSVC
classifiersvm=LinearSVC(multi_class='ovr',random_state=0)
classifiersvm.fit(X_train_new,Y_train)

Y_predsvm=classifiersvm.predict(X_test_new)

from sklearn.metrics import confusion_matrix
cmsvm=confusion_matrix(Y_test,Y_predsvm)

from sklearn import metrics
scoresvm=metrics.accuracy_score(Y_test, Y_predsvm)
print("accuracy:   %0.3f" % scoresvm)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_predsvm,target_names=Categories))

#RandomForest
from sklearn.ensemble import RandomForestClassifier
classifierrf=RandomForestClassifier(n_estimators=10 , criterion='entropy' ,random_state=0)
classifierrf.fit(X_train_new,Y_train)

Y_predrf=classifierrf.predict(X_test_new)

from sklearn.metrics import confusion_matrix
cmrf=confusion_matrix(Y_test,Y_predrf)

from sklearn import metrics
scorerf=metrics.accuracy_score(Y_test, Y_predrf)
print("accuracy:   %0.3f" % scorerf)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_predrf,target_names=Categories))

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

titles = ('SVC with linear kernel')

fig = plt.figure(figsize=(11,6))

X0, X1 = X_train_new[:, 0], X_train_new[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(classifier, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train_new, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Questions')
plt.ylabel('Tag')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_new, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Questions')
plt.ylabel('Tag')
plt.legend()
plt.show()

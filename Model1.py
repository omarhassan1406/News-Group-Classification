import os
import pickle
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix , ConfusionMatrixDisplay , confusion_matrix
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
stop_words = set(stopwords.words('english'))
stop_words.add(',')
stop_words.add('--')
stop_words.add(':')
stop_words.add('>')
stop_words.add('*')
stop_words.add('.')
stop_words.add(')')
stop_words.add('(')
stop_words.add('!')
stop_words.add('?')
stop_words.add('@')
stop_words.add("''")
stop_words.add('|')
stop_words.add('``')
stop_words.add('-')
stop_words.add('<')
stop_words.add(';')
stop_words.add("'s")
stop_words.add("n't")
stop_words.add('...')
stop_words.add('[')
stop_words.add(']')
stop_words.add('#')
stop_words.add('%')

Base_dir = 'E:/NLP_dataset/test'
lables = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
          'misc.forsale','rec.autos','rec.motorcycles',
          'rec.sport.baseball','rec.sport.hockey','sci.crypt',
          'sci.electronics','sci.med','sci.space',
          'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
          'talk.politics.misc','talk.religion.misc'
          ]

#Dataset Creation
def create_data_set():
    with open('E:/NLP_dataset/test_data.txt' , 'w' , encoding='utf8') as outfile:
        for lable in lables:
            dir = '%s/%s'%(Base_dir,lable)
            for filename in os.listdir(dir):
                fullfilename = '%s/%s'%(dir , filename)
                with open(fullfilename , 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n' , ' ')
                    outfile.write('%s\t\t\t%s\n'%(lable,text))

#create_data_set()

def setting_data():
    documents=[]
    with open('E:/NLP_dataset/data.txt' , 'r' , encoding='utf8') as data:
        for row in data:
            col = row.split('\t\t\t')
            doc =[col[0] , col[1]]
            documents.append(doc)
    return documents
data_doc = setting_data()

def test_setting_data():
    documents=[]
    p=0
    with open('E:/NLP_dataset/test_data.txt' , 'r' , encoding='utf8') as data:
        for row in data:
            col = row.split('\t\t\t')
            doc =[col[0] , col[1]]
            documents.append(doc)
    return documents
test_data_doc = test_setting_data()


def frecuency_dic():
    tokens = defaultdict(list)
    for doc in data_doc:
        lable = doc[0]
        text = doc[1]
        #convert to lowercase
        text = text.lower()

        token_words = word_tokenize(text)
        token_words = [t for t in token_words if not t in stop_words]
        tokens[lable].extend(token_words)


    for categ_lables , categ_text in tokens.items():
        print(categ_lables)
        fd = FreqDist(categ_text)
        print(fd.most_common(20))

#frecuency_dic()


def train_data_preparation(docs):
    random.shuffle(docs)
    X_train = []
    Y_train = []
    for i in range(len(docs)):
        X_train.append(docs[i][1])
        Y_train.append(docs[i][0])

    return X_train , Y_train

def test_data_preparation(docs):
    random.shuffle(docs)
    X_test = []
    Y_test = []
    for i in range(len(docs)):
        X_test.append(docs[i][1])
        Y_test.append(docs[i][0])

    return X_test, Y_test

def evaluation(vectorizer , classifier , X , Y):
    X_tf_idf = vectorizer.transform(X)
    Y_predict = classifier.predict(X_tf_idf)
    precision = metrics.precision_score(Y, Y_predict , pos_label='positive' , average='micro')
    recall = metrics.recall_score(Y , Y_predict, pos_label='positive' , average='micro')
    f1 = metrics.f1_score(Y , Y_predict, pos_label='positive' , average='micro')
    print("%f\t%f\t%f\t" % (precision , recall , f1))
    return Y_predict

X_train , Y_train = train_data_preparation(data_doc)
X_test , Y_test = test_data_preparation(test_data_doc)

# vectorize = TfidfVectorizer(stop_words=stop_words , ngram_range=(1,3) , min_df=3 )
# dtm = vectorize.fit_transform(X_train)
# naive_bayes_classifier = MultinomialNB().fit(dtm , Y_train)
vectorize = np.load('E:/NLP_dataset/vectorizer.pkl', allow_pickle=True)
naive_bayes_classifier = np.load('E:/NLP_dataset/naive_bayes_classifier.pkl', allow_pickle=True)
#classifier_name = 'naive_bayes_classifier.pkl'
#pickle.dump(naive_bayes_classifier , open(classifier_name , 'wb'))
#vect_name = 'vectorizer.pkl'
#pickle.dump(vectorize, open(vect_name , 'wb'))
print("Training Accuracy : ")
evaluation(vectorize , naive_bayes_classifier , X_train , Y_train)
print("Testing Accuracy : ")
Y_predict = evaluation(vectorize , naive_bayes_classifier , X_test , Y_test)
cm = confusion_matrix(Y_test , Y_predict , labels= naive_bayes_classifier.classes_)


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
          'misc.forsale','rec.autos','rec.motorcycles',
          'rec.sport.baseball','rec.sport.hockey','sci.crypt',
          'sci.electronics','sci.med','sci.space',
          'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
          'talk.politics.misc','talk.religion.misc'])

cm_display.plot()
plt.show()

import random
from nltk.corpus import stopwords
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from sklearn.metrics import plot_confusion_matrix , ConfusionMatrixDisplay , confusion_matrix
import matplotlib.pyplot  as plt

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

Base_dir = 'test'
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


if (os.path.exists('E:/NLP_dataset/Train_Data.npy')):
    #X_train = np.load('E:/NLP_dataset/Train_Data.npy', allow_pickle=True)
    X_test = np.load('E:/NLP_dataset/Test_Data.npy', allow_pickle=True)
    #Y_train = np.load('E:/NLP_dataset/Train_labels.npy', allow_pickle=True)
    Y_test = np.load('E:/NLP_dataset/Test_labels.npy', allow_pickle=True)
    vectorizer = np.load('E:/NLP_dataset/vectorizer.pkl', allow_pickle=True)

else:
    X_train, Y_train = train_data_preparation(data_doc)
    X_test, Y_test = test_data_preparation(test_data_doc)
    np.save('E:/NLP_dataset/Train_Data.npy', X_train)
    np.save('E:/NLP_dataset/Test_Data.npy', X_test)
    np.save('E:/NLP_dataset/Train_labels.npy', Y_train)
    np.save('E:/NLP_dataset/Test_labels', Y_test)

#vectorize = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,3), min_df=3 )
#dtm = vectorize.fit_transform(X_train)
#dtm = dtm.toarray()
#dtm = np.array(dtm)
#valid_dtm = dtm[10000:,:]
#valid_dtm = np.array(valid_dtm)
#dtm = dtm[:10000 , :]
Y_Train = []
test_dtm = vectorizer.transform(X_test)
test_dtm = test_dtm.toarray()
test_dtm = np.array(test_dtm)



Y_Test = []
for y in Y_test:
    if y == 'alt.atheism' :
        Y_Test.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif y == 'comp.graphics':
        Y_Test.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif y == 'comp.os.ms-windows.misc':
        Y_Test.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif y =='comp.sys.ibm.pc.hardware':
        Y_Test.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif y == 'comp.sys.mac.hardware':
        Y_Test.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif y == 'comp.windows.x':
        Y_Test.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'misc.forsale':
        Y_Test.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'rec.autos':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'rec.motorcycles':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'rec.sport.baseball':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'rec.sport.hockey':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'sci.crypt':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'sci.electronics':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif y == 'sci.med':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif y == 'sci.space':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif y == 'soc.religion.christian':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif y == 'talk.politics.guns':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif y == 'talk.politics.mideast':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif y == 'talk.politics.misc':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif y == 'talk.religion.misc':
        Y_Test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

LR = 0.0003
MODEL_NAME = 'Docs_CNN'

#CNN MODLE
Convolution_input = input_data(shape=[None,231462] ,  name='input')

Convolution1 = fully_connected(Convolution_input , 2048 ,  activation='relu')

Convolution2 = fully_connected(Convolution1, 1024 , activation='relu')

Convolution3 = fully_connected(Convolution2, 512 ,  activation='relu')

CNN_Output_Layer = fully_connected(Convolution3, 20 , activation='softmax')

CNN_Output_Layer = regression(CNN_Output_Layer, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

Model = tflearn.DNN(CNN_Output_Layer, tensorboard_dir='log', tensorboard_verbose=3)

if (os.path.exists('E:/NLP_dataset/Model.tfl.meta')):
    Model.load('E:/NLP_dataset/Model.tfl')

else:
    Model.fit({'input': dtm }, {'targets': Y_Train}, n_epoch=5,
           validation_set=({'input': valid_dtm }, {'targets': Y_valid}),
           snapshot_step= 500, show_metric=True, run_id= MODEL_NAME)
    Model.save('E:/NLP_dataset/Model.tfl')

predictions = Model.predict(test_dtm)

num_of_true = 0
Predictions_label = []
for y in range(len(Y_Test)):
    index = predictions[y].argmax()
    if index == 0:
        Predictions_label.append('alt.atheism')
    elif index == 1:
        Predictions_label.append('comp.graphics')
    elif index == 2:
        Predictions_label.append('comp.os.ms-windows.misc')
    elif index == 3:
        Predictions_label.append('comp.sys.ibm.pc.hardware')
    elif index == 4:
        Predictions_label.append('comp.sys.mac.hardware')
    elif index == 5:
        Predictions_label.append('comp.windows.x')
    elif index == 6:
        Predictions_label.append('misc.forsale')
    elif index == 7:
        Predictions_label.append('rec.autos')
    elif index == 8:
        Predictions_label.append('rec.motorcycles')
    elif index == 9:
        Predictions_label.append('rec.sport.baseball')
    elif index == 10:
        Predictions_label.append('rec.sport.hockey')
    elif index == 11:
        Predictions_label.append('sci.crypt')
    elif index == 12:
        Predictions_label.append('sci.electronics')
    elif index == 13:
        Predictions_label.append('sci.med')
    elif index == 14:
        Predictions_label.append('sci.space')
    elif index == 15:
        Predictions_label.append('soc.religion.christian')
    elif index == 16:
        Predictions_label.append('talk.politics.guns')
    elif index == 17:
        Predictions_label.append('talk.politics.mideast')
    elif index == 18:
        Predictions_label.append('talk.politics.misc')
    elif index == 19:
        Predictions_label.append('talk.religion.misc')


    if Y_Test[y][index] == 1:
        num_of_true += 1

Accuracy = num_of_true/len(Y_Test)

print(f"Accuracy = " + str(Accuracy))

#np.save('Predictions.npy', Predictions_label)


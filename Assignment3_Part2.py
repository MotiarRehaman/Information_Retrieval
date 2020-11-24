#################################################################### Importing necessary files ###################################################

import argparse
import pandas as pd
import numpy as np
import os
import re
from time import time 
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

######################################################################### For accepting the input ###################################################################################

parser = argparse.ArgumentParser(description = 'Input the text file')
parser.add_argument('file_path')
parser.add_argument('output_file')
args = parser.parse_args()

filepath = args.file_path
output   = args.output_file


	
######################################################################### Downloading necessary files ##########################################################

print('Downloading necessary libraries......')

nltk.download('punkt')     # For tokenizers
nltk.download('inaugural') # For dataset
nltk.download('wordnet')   # Since Lemmatization method is based on WorldNet's built-in morph function.
stopwords = nltk.corpus.stopwords.words('english')


########################################################################### Cleaning Function #####################################################################
wn = WordNetLemmatizer()

def clean_query(query):
    text   = "".join([c for c in query if c not in string.punctuation]) 
    remove = re.sub('[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(remove)
    lemmatized = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    processed_query = ' '.join([word for word in lemmatized])
    return processed_query



########################################################################## Getting the classes and data #############################


#for the class1 training and test 
class1_train_path  = os.getcwd() + '/dataset' + '/class1' + '/train'
training_files1    = os.listdir(class1_train_path)
class1_test_path   = os.getcwd() + '/dataset' + '/class1' + '/test'
testing_files1     = os.listdir(class1_test_path)

#for the class2 training and test 
class2_train_path  = os.getcwd() + '/dataset' + '/class2' + '/train'
training_files2    = os.listdir(class2_train_path)
class2_test_path   = os.getcwd() + '/dataset' + '/class2' + '/test'
testing_files2     = os.listdir(class2_test_path)




################################################################### Creating X_train, X_test, y_train, y_test ######################################

print('Text preprocessing .....')

X_train_class1 = []

for i in range(len(training_files1)):
    file         = open(class1_train_path + str('/') + training_files1[i] , 'rb').read().decode(errors='replace')
    cleaned_file = clean_query(file.lower())
    X_train_class1.append(cleaned_file)
    
    

X_train_class2 = []

for i in range(len(training_files2)):
    file         = open(class2_train_path + str('/') + training_files2[i] , 'rb').read().decode(errors='replace')
    cleaned_file = clean_query(file.lower())
    X_train_class2.append(cleaned_file)
    
    
y_train_class1 = ['class1' for i in range(len(X_train_class1))]
y_train_class2 = ['class2' for i in range(len(X_train_class2))]

X_train  = X_train_class1 + X_train_class2
y_train  = y_train_class1 + y_train_class2   

data = pd.DataFrame()
data['Mail']  = X_train
data['Class'] = y_train
#data = data.sample(frac = 1)
data.reset_index(inplace = True, drop = True)



X_test_class1 = []

for i in range(len(testing_files1)):
    file         = open(class1_test_path + str('/') + testing_files1[i] , 'rb').read().decode(errors='replace')
    cleaned_file = clean_query(file.lower())
    X_test_class1.append(cleaned_file)
    
    

X_test_class2 = []

for i in range(len(testing_files2)):
    file         = open(class2_test_path + str('/') + testing_files2[i] , 'rb').read().decode(errors='replace')
    cleaned_file = clean_query(file.lower())
    X_test_class2.append(cleaned_file)


y_test_class1 = ['class1' for i in range(len(X_test_class1))]
y_test_class2 = ['class2' for i in range(len(X_test_class2))]

X_test  = X_test_class1 + X_test_class2
y_test  = y_test_class1 + y_test_class2

test_data          = pd.DataFrame()
test_data['Mail']  = X_test
test_data['Class'] = y_test

#test_data          = test_data.sample(frac = 1)
test_data.reset_index(inplace = True, drop = True)

############################################################################## Encoding y_train and y_test ############################################################

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test  = encoder.fit_transform(y_test)

# creating the tf_idf vectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None)

tfidf_vect.fit(data['Mail'])
X_train_tfidf =  tfidf_vect.transform(X_train)
X_test_tfidf =  tfidf_vect.transform(X_test)


# creating a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(data['Mail'])


# transform the training and validation data using count vectorizer object
X_train_count =  count_vect.transform(X_train)
X_test_count =  count_vect.transform(X_test)





####################################################################### Applying the model for Rocchio Algorithm ##################################################

# creating cluster centers 
X_train_tfidf_list = X_train_tfidf.toarray()
X_test_tfidf_list  = X_test_tfidf.toarray()

center1 = np.zeros(X_train_tfidf.shape[1])
center2 = np.zeros(X_train_tfidf.shape[1])

for i in range(len(X_train_tfidf_list)):
    document_tfidf = np.array(X_train_tfidf_list[i])
    if y_train[i] == 0:
        center1 = center1 + document_tfidf
    if y_train[i] == 1:
        center2 = center2 + document_tfidf
        
center1 = center1 / list(y_train).count(0)
center2 = center2 / list(y_train).count(1)

b_ = [0]

prediction = [[] for i in range(4)]

    
for j in range(len(b_)):
    
    b = b_[j]
    #print(b)
    
    for i in range(len(X_test_tfidf_list)):

        document_tfidf = np.array(X_test_tfidf_list[i])
    
        dist_from_center1 = np.linalg.norm(center1 - document_tfidf)
        dist_from_center2 = np.linalg.norm(center2 - document_tfidf)
        
        
        flag = 0
        
        if dist_from_center1 < dist_from_center2 - b :
            prediction[j].append(0)
            flag = 1
        elif dist_from_center2 < dist_from_center1 - b :
            prediction[j].append(1)
            flag = 2
        else:
            prediction[j].append(2)
        
        #print(dist_from_center1, dist_from_center2, flag, b)

        
f1_score_rocchio = []

for i in range(len(b_)):
    f1_score_rocchio.append(f1_score(y_test,prediction[i], average ='macro'))


    
   

############################################################################## Creating the submittable format ###################################################################


file1 = open(os.getcwd() + '/' + str(output) + '.txt', 'w')


# Rocchio code starts here 
file1.write('b value          0                ')
file1.write('\n')
file1.write('F1 score         '+str('%.5f'%f1_score_rocchio[0]))

file1.close()











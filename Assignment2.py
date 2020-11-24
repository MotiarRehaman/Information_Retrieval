######################################################### Importing the libraries ########################################

import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html
from urllib.request import urlopen
import os
import re
import json
from collections import Counter
import argparse

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from gensim import utils
import gensim.parsing.preprocessing as gsp

######################################################### For accepting the input #############################################

parser = argparse.ArgumentParser(description = 'Input the text file')
parser.add_argument('query_file')
args = parser.parse_args()

queries = open(args.query_file, 'r').readlines()

for i in range(len(queries)):
    queries[i] = queries[i].rstrip('\n')
   


########################################################## Downloading required files ##########################################

print('Downloading relevant libraries for text pre-processing')

nltk.download('punkt') # For tokenizers
nltk.download('inaugural') # For dataset
nltk.download('wordnet') # Since Lemmatization method is based on WorldNet's built-in morph function.
stopwords = nltk.corpus.stopwords.words('english')





############################################################Importing the static scores and the leaders file ########################



############################# Getting the path ###################################

path1         = os.getcwd()
parent_folder = os.path.dirname(os.getcwd())
path  	      = parent_folder +'/Dataset'
files 	      = list(os.listdir(path))

total_files = len(files)


print('Importing the static scores and leaders list')

filename1 = parent_folder + '/StaticQualityScore.pkl'

input1 = open(filename1,'rb')
static_scores = pickle.load(input1)

filename2 = parent_folder +'/Leaders.pkl'

input2 = open(filename2,'rb')
leaders = pickle.load(input2)


############################################################## Cleaning the text ##################################################



# For Lemmatization

wn = WordNetLemmatizer()

########################################################## Function for text preprocessing ##################################################

def clean_query(query):
    text   = "".join([c for c in query if c not in string.punctuation]) 
    remove = re.sub('[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(remove)
    lemmatized = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    processed_query = ' '.join([word for word in lemmatized])
    return processed_query


filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
          ]

def cleantext(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)  
    return s


def strip_html(s):
    gsp.strip_tags(s)
    return s






########################################################## Creating the Inverted Positional Index ################################




print('Creating the inverted positional index')
print('Might take a few minutes')

Inverted_Positional_Index = {}

for i in range(len(files)):
    
    doc_name    = files[i]
    file = open(path + str('/') + files[i], encoding='utf8')
    soup = BeautifulSoup(file, 'html5lib')
    text = soup.get_text()
    
    cleaned_text   = clean_query(text.lower())
    term_freq_dict = Counter(word_tokenize(cleaned_text))
    
    # creating the term_frequency_list
    #term_and_freq = []
    keys = list(term_freq_dict.keys())
    
    
    for j in range(len(keys)):
        word  = keys[j]
        freq  = term_freq_dict[word]     
        
        tf    = np.log(1 + freq)/2.303
        
        
        if word in Inverted_Positional_Index:
            Inverted_Positional_Index[word].append((doc_name, tf))
        
        else:
            Inverted_Positional_Index[word] = []
            Inverted_Positional_Index[word].append(0)
            Inverted_Positional_Index[word].append((doc_name, tf))
        
    
        

# Adding the idf for the terms 

words = list(Inverted_Positional_Index.keys())

for i in range(len(words)):
    word = words[i]
    
    dft = len(Inverted_Positional_Index[word]) - 1      
    idf = np.log(1000/dft)/2.303
    
    Inverted_Positional_Index[word][0] = idf
    
    
# Creating the format given in the question

inverted_positional_index = {}

words = list(Inverted_Positional_Index.keys())

for i in range(len(words)):
    word = words[i]
    idf  = Inverted_Positional_Index[word][0]
    inverted_positional_index[(word,idf)] = Inverted_Positional_Index[word][1:]
    

print('Inverted Positional Index created ')
print('Creating the ChampionListLocal ')

################################## Building the champion list local ####################################


ChampionListLocal = {}

keys = list(inverted_positional_index.keys())
    

    
for i in range(len(keys)):
    term = keys[i][0]
    idf  = keys[i][1]
    posting = inverted_positional_index.get(keys[i])
        
    # Create a dictionary of postings and use counter for top 50
    temp_dict = {}
        
    for j in range(len(posting)):
        doc_name = posting[j][0]
        doc_tf   = posting[j][1]
        temp_dict[doc_name] = doc_tf
            
    posting1 = Counter(temp_dict)
        
    value =  []
    
    for m, n in posting1.most_common(50):
        value.append(m)
            
    ChampionListLocal[term] = value
            
    
print('Creating the ChampionListGlobal')      
            
       


################################# Creating the Champion List Global #######################



temp = {}
global_values = {}
temp_top_50   = {}
ChampionListGlobal = {}

keys = list(inverted_positional_index.keys())

for i in range(len(keys)):
    term = keys[i][0]
    idf  = keys[i][1]
    posting = inverted_positional_index.get(keys[i])
    
    for j in range(len(posting)):
        doc_name = posting[j][0]
        doc_tf   = posting[j][1]
        doc_tf_idf = doc_tf * idf
        
        index    = int(doc_name.split('.html')[0])
        gd       = static_scores[index]
        value    = gd + doc_tf_idf
        
        
        global_values[term] = (doc_name, value)
        temp[term] = (doc_name, doc_tf_idf)
        temp_top_50[doc_name] = value
    
    # Getting top 50 from this global_values dictionary
    top_50_list = []
    
    top_50 = Counter(temp_top_50)
    
    for m,n in top_50.most_common(50):
        top_50_list.append(m)
        
    ChampionListGlobal[term] = top_50_list
    
    



########################## creating the master document dictionary for ease of computation ################################

document = {}

keys = list(inverted_positional_index.keys())

for i in range(len(keys)):
    
    term = keys[i][0]
    idf  = keys[i][1]
    
    postings = inverted_positional_index[keys[i]]
    
    for j in range(len(postings)):
        
        doc_name = int(re.split('.html',postings[j][0])[0])
        tf       = postings[j][1]
        
        if doc_name in document:
            
            document[doc_name][term] = (tf,idf,tf * idf)
        
        else:
            
            document[doc_name]       = {}
            document[doc_name][term] = (tf,idf,tf * idf)
    


###################################################### creating mod_vd for all the documents ##########################

mod_vd = [0 for i in range(len(files))]

keys = list(inverted_positional_index.keys())

for i in range(len(keys)):
    key = keys[i][0]
    idf = keys[i][1]
    
    postings = inverted_positional_index[keys[i]]
    
    for j in range(len(postings)):
        doc_name         = int(re.split('.html',postings[j][0])[0])
        tf               = postings[j][1]
        tf_idf           = tf * idf
        tf_idf_sq        = tf_idf**2
        mod_vd[doc_name] += tf_idf_sq
    
 
    
for i in range(len(mod_vd)):
    mod_vd[i] = mod_vd[i]**0.5
    
    
print('Getting the list of followers for cluster pruning')  
    
########################### Preprocessing for Cluster Pruning: Getting the followers of the leaders #########################




followers = {}


for i in range(1000):
    
    doc_name  = i
    doc_terms = document[i]
    terms     = list(doc_terms.keys())
    doc_mod_vq = mod_vd[doc_name]
    max_score     = 0
    

    
    for j in range(len(leaders)):
        leader           = leaders[j]
        leader_doc_terms = document[leader]
        leader_terms     = list(leader_doc_terms.keys())
        leader_mod_vq     = mod_vd[leader]
        
        intersection     = list(set(terms) & set(leader_terms))

        doc_vd    = []
        leader_vd = []
        num       = 0
        
        for k in range(len(intersection)):
            
            word = intersection[k]
            
            doc_tf_idf = doc_terms[word][2]
            lea_tf_idf = leader_doc_terms[word][2]
            
            #doc_vd.append(doc_tf_idf)
            #leader_vd.append(lea_tf_idf)
            
            num += doc_tf_idf * lea_tf_idf
        
        score = num /(leader_mod_vq * doc_mod_vq)
        
        if score >= max_score:
            max_score  = score
            doc_leader = leader
            
    
    if doc_leader in followers:
        
        followers[doc_leader].append(doc_name)
        
    else:
        
        followers[doc_leader] = []
        followers[doc_leader].append(doc_name)
            
            
    
    
print('Query PROCESSING!!!')
   
   
####################################################### query processing ##############################################




keys = list(inverted_positional_index.keys())
doc_keys = list(document.keys())


############################################# Final loop for generating the results ###############################

for i in range(len(queries)):
    query = queries[i]
    
    print('Getting results for query' + str(i + 1))
    
    cleaned_query = clean_query(query.lower())
    query_len  = len(word_tokenize(cleaned_query))
    query_list = word_tokenize(cleaned_query)

    doc_vd     = [0 for i in range(len(files))]
    
    
    for m in range(len(doc_keys)):
        
        doc_name = m
        vd  = []
        
        for j in range(len(query_list)):
        
            term = query_list[j]
        
            if term in document[doc_name]:
            
                tf_idf = document[doc_name][term][2]
            
            else:
            
                tf_idf = 0
            
            vd.append(tf_idf)
        
        doc_vd[doc_name] =  vd


    doc_scores = []


    ######################## scoring the tf_idf scores ######################

    

    vq   = []

    for m in range(query_len):
        term = query_list[m]
        
        flag = 0
        
        for j in range(len(keys)):
            key = keys[j][0]
            if term == key:
                flag = 1
                vq.append(keys[j][1])
                
        if flag == 0:
            vq.append(0)
            

    mod_vq = 0            
    for m in range(len(vq)):
        mod_vq += vq[m]**2

    mod_vq = mod_vq**0.5



    for m in range(len(doc_keys)):
        doc_name     = m
        vd           = doc_vd[m]
        mod_vd_doc   = mod_vd[m]
    
        num = 0
        
        
        for j in range(query_len):
                        
            num += vd[j]*vq[j]
    
        
        tf_idf_score = num/(mod_vd_doc * mod_vq)
    
        doc_scores.append((doc_name, tf_idf_score))  
    
    
    
    result1 = sorted(doc_scores, key = lambda x:x[1], reverse = True)[:10]
    
    
    ############################### ChampionListocal #############################################
  

    ############ Getting champion list doc_champion_list for the query terms ###########

    doc_champion_list = []
    cham_loc_keys     = list(ChampionListLocal.keys())

    for m in range(query_len):
        term = query_list[m]
    
    
        for j in range(len(cham_loc_keys)):
        
            word     = cham_loc_keys[j]
       
            postings = []
            if term == word:
                postings.append(ChampionListLocal[word])
                      
                for k in range(len(postings[0])):
                    doc_name = int(re.split('.html',postings[0][k])[0])
                    doc_champion_list.append(doc_name)
        
        
                doc_champion_list = list(set(doc_champion_list))
        
        
    result2 = []

    for m in range(len(doc_champion_list)):
        doc_name_local = doc_champion_list[m]
        score          = doc_scores[doc_name_local][1]
        result2.append((doc_name_local, score))

    if len(result2) <= 10:
        pass
    else:
        result2 = sorted(result2, key = lambda x:x[1], reverse = True)[:10]


        
        
    #################### Global Champion List #############################
    

    doc_champion_list_global = []
    cham_glo_keys     = list(ChampionListGlobal.keys())

    for m in range(query_len):
        term = query_list[m]
    
    
        for j in range(len(cham_glo_keys)):
        
            word     = cham_glo_keys[j]
       
            postings = []
            if term == word:
                postings.append(ChampionListGlobal[word])
                      
                for k in range(len(postings[0])):
                    doc_name = int(re.split('.html',postings[0][k])[0])
                    doc_champion_list_global.append(doc_name)
        
        
                doc_champion_list_global = list(set(doc_champion_list_global))

    result3 = []

    for m in range(len(doc_champion_list_global)):
        doc_name_local = doc_champion_list_global[m]
        score          = doc_scores[doc_name_local][1]
        result3.append((doc_name_local, score))

    if len(result3) <= 10:
        pass
    else:
        result3 = sorted(result3, key = lambda x:x[1], reverse = True)[:10]


    
        
        
    #################### Cluster Pruning #################################

    l_scores = []
    max_score = 0

    for m in range(len(leaders)):
        leader = leaders[m]
    
        for j in range(len(doc_scores)):
            doc_score = doc_scores[j]
            doc       = doc_score[0]
            score     = doc_score[1]
            if doc == leader:
                if score >= max_score:
                    max_score = score
                    q_leader  = leader
    



    follower_keys = list(followers.keys())    
    l_followers = followers[q_leader]

   
    
    
    
    result4 = []
    
    
    for m in l_followers:
        doc_score = doc_scores[m]
        result4.append((m, doc_score[1]))
    
    length_result4 = len(result4)
    
    if length_result4 <= 10:
        pass
    else:
        result4 = sorted(result4, key = lambda x:x[1], reverse = True)[:10]
    
    
    
    
    results = open(os.getcwd() + str('/RESULTS2_18CH3FP26.txt'), 'a')

    output = ''
    output = output + query
    output = output +str('\n')

    for i in range(len(result1)):
        output = output + str('<') + str(result1[i][0]) + str(',') + str(result1[i][1]) + str('>,')
    
    output = output +str('\n') 
    for i in range(len(result2)):
        output = output + str('<') + str(result2[i][0]) + str(',') + str(result2[i][1]) + str('>,')

    output = output +str('\n') 
    for i in range(len(result3)):
        output = output + str('<') + str(result3[i][0]) + str(',') + str(result3[i][1]) + str('>,')
    
    
    output = output +str('\n') 
    for i in range(len(result4)):
        output = output + str('<') + str(result4[i][0]) + str(',') + str(result4[i][1]) + str('>,')

    output = output +str('\n\n')    

    results.write(output)
    results.close()



######################################################################### The End ##################################################################



















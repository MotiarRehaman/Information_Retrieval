# For task 3

# Importing libraries
import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from gensim import utils
import gensim.parsing.preprocessing as gsp

#downloading the required files

nltk.download('punkt') # For tokenizers
nltk.download('inaugural') # For dataset
nltk.download('wordnet') # Since Lemmatization method is based on WorldNet's built-in morph function.
stopwords = nltk.corpus.stopwords.words('english')

# For Lemmatization

wn = WordNetLemmatizer()

################################################################################## Function for text preprocessing #############################################################################

def clean(txt):
    text = "".join([c for c in txt if c not in string.punctuation])                 #Removes punctuations
    tokens = re.split('\W+',text)                                                   #Tokenizes the text
    lemmatized = [wn.lemmatize(word) for word in tokens if word not in stopwords]   #Removes stopwords
    sentences = ' '.join([word for word in lemmatized])
    return text


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



################################################################################## Text Preprocessing #############################################################################

path1 = os.getcwd()
path2 = path1 + str('/ECTText')
files = list(os.listdir(path2))

print("Text Preprocessing....")

for i in range(len(files)):
    
    file = files[i]
    text = open(path2 + str('/') + file, 'r').read()
    
    cleaned_text = cleantext(text.lower())
    
    tokens       = re.split('\W+',cleaned_text)
    lemmatized   = [wn.lemmatize(word) for word in tokens]   #Removes stopwords
    sentences    = ' '.join([word for word in lemmatized])
    
    document = open(path2 + str('/') + file, 'w')
    document.write(sentences)
    
    document.close()
    
    #print(i)
    
    

############################################################################### Creating the Positional Index ########################################################################

positional_index = {} 


print("Creating the Positional Index....")

for i in range(len(files)):
    
    document_name = files[i]
    
    text           = open(path1 + '/ECTText/' + files[i], 'r').read()
    tokenized_text = word_tokenize(text)
    
    
    for position, word in enumerate(tokenized_text):
        
        if word in positional_index:
            
            positional_index[word][0] += 1
            
            if document_name in positional_index[word][1]:
                
                positional_index[word][1][document_name].append(position)
            
            else:
                
                positional_index[word][1][document_name] = [position]
        
        else:
            
            positional_index[word] = []
            positional_index[word].append(1)
            positional_index[word].append({})
            positional_index[word][1][document_name] = [position]
            
    
    
    
############################################################################## Saving positional_index as a json file ###########################################################################

print("Saving the json file ....")

with open('Positional_Index.json', 'w') as fp:
    json.dump(positional_index, fp)
    
    
    
    
    
    
    
   

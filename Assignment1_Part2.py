# For task 2

# Importing the libraries 

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html
from urllib.request import urlopen
import os
import re
import json


# Getting the path

path1 = os.getcwd()
path  = path1 +'/ECT'
files = list(os.listdir(path))


################################################################ Creating the final ECTNestedDict ###########################################

ECTNestedDict = {}


print("Creating the ECTNestedDict......")

for i in range(len(files)):
    
    

    #################################################################### Name of the articles ##################################
    

    file = files[i]
    
    #name = re.findall(r'article\-\d+',str(file))
    #name = name[0]
    name  = i
    #print(name) 
    
    
    
    file = open(path + str('/') + file, encoding='utf8')
    soup = BeautifulSoup(file, 'html5lib')
    table = soup.find_all('p')


    
    ########################################################################## Date Part #############################################

    
    # date part
    string = str(table[0])
    date = re.findall(r'\w+\s+\d+\,\s\d+', string)
    if len(date) == 0:
        #print(i)
        string = str(table[2])
        date = re.findall(r'\w+\s+\d+\,\s\d+', string)
        #print(date)
        
        if len(date) == 0:
            #print(i)
            string = str(table[3])
            date = re.findall(r'\w+\s+\d+\,\s\d+', string)
            #print(date)
            
        if len(date) == 0:
            #print(i)
            string = str(table[2])
            date = re.findall(r'\w+\s+\d+\s\d+', string)
            #print(date)        

    date = date[0]
    #print(date)
    
    ############################################################################ Participants Part ############################
    
    
    participants_t = []
    participants_e = []

    flag_t = 0
    flag_e = 0

    j = 2

    # 'Conference Call Participants '

    

    try:
        j = 2
    
        while(soup.find_all('p')[j].text != 'Conference Call Participants'):
            temp = soup.find_all('p')[j].text
            participants_t.append(temp)
            j +=1
        
            flag_t = 1

    except IndexError as e1:
    
        absent = 0

        for row in table:
            text = row.text
    
            if text == 'Conference Call Participants ':
                absent = 1
    
        j = 2
    
        if absent == 1:     
            while(soup.find_all('p')[j].text != 'Conference Call Participants '):
                temp = soup.find_all('p')[j].text
                participants_e.append(temp)
                j +=1
        
                #print('Error took the except block')
        
            flag_e = flag_t + 1
            
        else:
            j = 2
    
            while(len(str(soup.find_all('p')[j]).split('<strong>')) != 2):
                temp = soup.find_all('p')[j].text
                participants_e.append(temp)
    
                j +=1
            flag_e = flag_t + 1
    

# if only try works out flag_t = 1 and flag_e = 0
# if except block works matlab flag_e = 2, flag_t = 1

    if flag_t > flag_e:
        participants = participants_t
    
    else:
        participants = participants_e
    
        
    j += 1

    j1 = j



    participants_1 = []
    participants_2 = []

    flag_1 = 0
    flag_2 = 0

    

    try:
        while(len(str(soup.find_all('p')[j1]).split('<strong>')) != 2):
            temp = soup.find_all('p')[j1].text
            participants_1.append(temp)
    
            j1 +=1
        
            flag_1 = 1
    


    except IndexError as e:
    
            while(soup.find_all('p')[j].text != 'Operator'):
                temp = soup.find_all('p')[j].text
                participants_2.append(temp)
    
                j +=1
                flag_2 = flag_1 + 1
            
            
        
    if flag_1 > flag_2:
        participants = participants + participants_1
    
    else:
        participants = participants + participants_2
    

    #print('Length of part ', len(participants))
   
    
    
    
    ######################################################################################## Presentation Part ##################################################3#################
    presentation = {}

    #Question-And-Answer Session 
    #Question-and-Answer Session

    index = 0

    for k in range(len(table)):
        if table[k].text.lower() == 'question-and-answer session':
            index = k
        elif table[k].text.lower() == 'question-and-answer session ':
            index = k

    
    #print('index value is ',index)

    for j in range(index):
        row = table[j]
        if len(str(row).split('<strong>')) == 2 and row.text != 'Company Participants' and row.text != 'Conference Call Participants':
            
            k = j + 1
            #print(k)
            para = table[k]
            #print(para)
            text1 = ' '
            while(len(str(para).split('<strong>')) != 2 and k < index):
            
                text1 = text1 + table[k].text
                #print(text1)
                k +=1
                para = table[k]
            
            
            presentation[row.text] = text1
            
    #print('presentation ', len(presentation))
    
    
    ############################################################################################ Questionaire Part #########################################################################
    
    questionaire = {}
    count = 0

    #print(i)
    

    for j in range(index + 1,len(table),1):
        row = table[j]
    
        if len(str(row).split('<strong>')) == 2:
        
            count += 1
            k = j + 1
            #print(k)
        
            try:
                para = table[k]
                #print(para)
                text1 = ' '
                while(len(str(para).split('<strong>')) != 2 and k < len(table)):
            
                    text1 = text1 + table[k].text
                    #print(text1)
            
                    if k == len(table) -1:
                        break
                    else:
                        k +=1
                        para = table[k]
            
                questionaire[count] = {row.text : text1}
                
            except IndexError as error:
            
                questionaire[1] = 'No questions were asked'

                
    #print('questionaire ', len(questionaire))
    
    
    TransDict = {}
    TransDict['Date'] = date
    TransDict['Participants'] = participants
    TransDict['Presentation'] = presentation
    TransDict['Questionaire'] = questionaire
    
            
    ECTNestedDict[name] = TransDict
    
    
    
    
# Making a json file of the ECTNestedDict

with open('ECTNestedDict.json', 'w') as fp:
    json.dump(ECTNestedDict, fp)
    

    
# loading the json file

with open('ECTNestedDict.json', 'r') as transcript_file:
    transcript = json.load(transcript_file)

    
################################################################## Creating the ECT text files of the documents ##########################################################################


article_names    = list(transcript.keys())


os.mkdir(os.path.join(os.getcwd(), 'ECTText'))
path2 = path1 + str('/ECTText')



print('Creating ECT text files of the documents... ')

for i in range(len(article_names)):
    
    document = open(path2 + str('/') + article_names[i] + str('.text'), 'w')
    
    individual_trans = list(transcript[article_names[i]].keys())
    questionaire     = transcript[article_names[i]][individual_trans[3]]
    ques_keys        = list(questionaire.keys())   
    
    presentation     = transcript[article_names[i]][individual_trans[2]]
    present_keys     = list(presentation.keys()) 
    
    
    for j in range(len(present_keys)):
    
        text    = transcript[article_names[i]][individual_trans[2]][present_keys[j]]
        document.write(text + str(' '))
        
    document.write(str('\n\n'))
    
    
    for j in range(len(ques_keys)):
        
        try :
            key              = list(transcript[article_names[i]][individual_trans[3]][ques_keys[j]].keys())[0]
            text             = transcript[article_names[i]][individual_trans[3]][ques_keys[j]][key]
            document.write(text +str(' '))
            
        except AttributeError as error:
            
            pass
        
    document.close()
    
    
    

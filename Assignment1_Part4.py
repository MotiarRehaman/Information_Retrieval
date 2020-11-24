# For task 4


# Importing the libraries'
import os
import json
import re
import argparse


######################################################################### For accepting the input ###################################################################################

parser = argparse.ArgumentParser(description = 'Input the text file')
parser.add_argument('query_file')
args = parser.parse_args()

queries = open(args.query_file, 'r').readlines()

for i in range(len(queries)):
	queries[i] = queries[i].rstrip('\n')
	


######################################################################### Creating Permuterm and Inverted Dictionary for query processing ##################################################



# loading the json file

with open('Positional_Index.json', 'r') as positional_ind:
    pos_ind = json.load(positional_ind)
    
keys = list(pos_ind.keys())

def rotated(str, n):
    return str[n:] + str[:n]



file1 = open("PermutermIndex.txt","w")



for key in sorted(keys):
    dkey = key + "$"
    for i in range(len(dkey),0,-1):
        out = rotated(dkey,i)
        file1.write(out)
        file1.write(" ")
        file1.write(key)
        file1.write("\n")
file1.close()


######################################################### Creating Sample Inverted Index and permuterm dictionary for query processing ###############################################################

Inverted = {}

for key, value in pos_ind.items():
    term    =  key
    nested_dict = pos_ind[key][1]
    articles     = list(nested_dict.keys())
    Inverted[term] = articles
    
print(len(Inverted))   
    
permuterm = {}
with open('PermutermIndex.txt') as f:
    
    for line in f:
        temp = line.split( )
        permuterm[temp[0]] = temp[1]
        
print(len(permuterm))


def prefix_match(term, prefix,case):
    term_list = []
    
    final_list = []
    for tk in term.keys():
        if tk.startswith(prefix):
            term_list.append(term[tk])
            
    if case == 3:
        
        first_letter = prefix[0]
        if len(prefix) > 1:
            second_letter = prefix[1]
        if len(prefix) > 2:
            third_letter = prefix[2]
        
        for terms in term_list:
 
            if len(prefix) > 2:   
                if terms[0] == first_letter and terms[1] == second_letter and terms[2] == third_letter:
                    final_list.append(terms)

            if len(prefix) > 1:   
                if terms[0] == first_letter and terms[1] == second_letter:
                    final_list.append(terms)
            else:
                if terms[0] == first_letter:
                    final_list.append(terms)               
                
        return list(set(final_list))
        
    return list(set(term_list))
       




result_file = open("RESULTS1_18CH3FP26.txt","w")

for query in queries:
    
    print(query)
    
    parts = query.split('*')
    if len(parts[0]) != 0 and len(parts[1]) !=0:
        case = 1                                             # in the middle
        prefix = parts[1] +str('$') + parts[0]
        #print(prefix)
    
    elif len(parts[0]) == 0:
        case = 2                                             # in the start
        prefix = parts[1] + str('$')
        #print(prefix)
    
    else:
        case = 3                                             # in the end
        prefix = parts[0]
        #print(prefix)
    

    term_list = prefix_match(permuterm, prefix, case)
    #print(term_list)
    #print(len(term_list))

   
    

    final_text = ''
    for term in term_list:
        
        pos_ind_dict = pos_ind[term][1]
        pos_ind_keys = list(pos_ind[term][1].keys())
        
        text = ''
        
        for j in range(len(pos_ind_keys)):
            article_name = pos_ind_keys[j]
        
            list_of_positions =  list(set(pos_ind_dict[article_name]))
            
        
            
            for i in range(len(list_of_positions)):
                pos = list_of_positions[i]
            
                #print(len(list_of_positions))
                if i == 0 and j == 0:
                
                    if j == len(pos_ind_keys) -1:
                        text = text + term + str(':<') + str(article_name) + str(',') +str(pos) + str('>;')
                    else:
                        text = text + term + str(':<') + str(article_name) + str(',') +str(pos) + str('>,')
                else:
                
                    if j == len(pos_ind_keys) -1:
                        text = text + str('<') + str(article_name) + str(',') +str(pos) + str('>;')
                    else:
                        text = text + str('<') + str(article_name) + str(',') +str(pos) + str('>,')
               
        final_text += text            
    
    final_text += str('\n')
    #print(final_text)
    result_file.write(final_text)
    
    
    

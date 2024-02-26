import re
from smart_open import open  # for transparently opening remote files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import itertools as it
from time import time
#jugement
g_DictSymbols = { '/*': '\n', '*': '\n', '*/': '\n', '//': '\n'}


def get_code_from_file(dic, code):
    '''
    According map and code, return original code 

    arg:
        dic: dictionary which requires code as key, filename,function name and target as value.
    code；
        string: number code string
    '''
    #print("The code in {} is obtain...".format(code))
    stoplist = set(';'.split())
    filename = dic[code][0]
    functionName = dic[code][1]
    folder_path = './patch/' + filename + '.txt'
    try:
        content = open(folder_path, "r",encoding='windows-1252')
    except: 
        print("No such file")
        return ""
      
    #Catch code and save in txt variable.
    target1 = False
    target2 = False
    txt = """"""
    
    for row in content:
        #print("-------{}-------".format(row))
        if target1 and target2:
            
            #if arrived at @@ or next diff commend,It means we dont need copy the code anymore
            if re.search('^@+', row) or re.search('^diff', row): 
                break
            else:
                if not re.search('^\+', row) and row != "":
                    # if exist "-" delete "-" then append to string
                    
                    txt += row[1:] if re.search('^\-', row) else row
        else :
            #if arrived at function name, then ++
            if functionName in row:
                target1 = True
            #if arrived at @@, means next line is command that we need
            if re.search('^@+', row) and target1:                
                target2 = True

    
    #Remove the comment
    txt = rm_comments(txt,code)
    
    #TODO: remove header
    #txt = rm_includeline(txt)

    #Remove empty line
    txt = rm_emptyline(txt)
    
    return txt
    



def rm_comments(s,code):
    '''
        This function will remove all comment from code
    '''
    fromPos = 0
    while (fromPos < len(s)):
        
        result = get1stSymPos(s, fromPos)
        # if code == "264402":
        #     print("---len fromPos {}, len s {}".format(fromPos,result))
        if result[0] == -1:  #Finished checking
            return s
        else:
            #Ending of line
            endPos = s.find(g_DictSymbols[result[1]], result[0] + len(result[1]))
            if result[1] == '//':  # Single comment
                if endPos == -1:
                    endPos = len(s)
                s = s.replace(s[result[0]:endPos], '', 1)
                fromPos = result[0]
            elif result[1] == '/*':  #block comment
                s = s.replace(s[result[0]:endPos + 2], '', 1)
                fromPos = result[0]
            elif result[1] == '*': #lack comment
                s = s.replace(s[result[0]:endPos + 2], '', 1)
                fromPos = result[0]    
            else:  #Normal string
                fromPos = endPos + len(g_DictSymbols[result[1]])
    
    return s

def rm_emptyline(ms):
    '''
        Remove empty line from code
    '''
    ms = "".join([s for s in ms.splitlines(True) if s.strip()])
    return ms


def rm_includeline(ms):
    '''
    Remove the header
    '''
    ms = "".join([s for s in ms.splitlines(True) if '#include' not in s])
    return ms

# 判断dictSymbols的key中，最先出现的符号是哪个，并返回其所在位置以及该符号
def get1stSymPos(s, fromPos=0):
    listPos = []  # 位置,符号
    for b in g_DictSymbols:
        pos = s.find(b, fromPos)
        listPos.append((pos, b))  # 插入位置以及结束符号 ex: [[5,"//"]]
    minIndex = -1  # 最小位置在listPos中的索引
    index = 0  # 索引
    while index < len(listPos):
        pos = listPos[index][0]  # 位置
        if minIndex < 0 and pos >= 0:  # 第一个非负位置
            minIndex = index
        if 0 <= pos < listPos[minIndex][0]:  # 后面出现的更靠前的位置
            minIndex = index
        index = index + 1
    if minIndex == -1:  # 没找到
        return (-1, None)
    else:
        return (listPos[minIndex])

def split_hold_txt(txt,split_type = 0):
    '''
        Input code to decompose all strings into string arrays for output

        args:
            txt: mutilple code
            split_type : int    0 means [['xx','xx'],['xx','xx']] 
                                1 means ['xx','xx','xx','xx'] 
                                2 means ['xx xx xx xx']
                                3 means ['xx xx','xx xx']
        output:
            array for string
    '''
    spc_symbol = ['(', ')', '[', ']', '{','}','.',';','#','"']
    stop_word = [""]
    output = []
    tempStr = ""
    #get each line of code
    
    for line in txt.splitlines():
        
        fixIndex = 0
        #get each char in line
        for index,char in enumerate(line):
            if char in spc_symbol:
                #insert space to special symbol
                line = insertChr_Before_After(line,index + fixIndex)
                fixIndex+=2
        #default split space
        if split_type == 0:
            output += [line.split()]
        elif split_type == 1:
            output += line.split()
        elif split_type == 2:
            line = line.replace('\t',"")
            tempStr += line
        elif split_type == 3:
            line = line.replace('\t',"")
            output += line
    if split_type == 2:
        return [tempStr]
    return output

def insertChr_Before_After(line, index):
    '''
        According index we have, insert space " " to code
        ex: (abc,2) = "ab c "
    '''
    tempList = list(line)
    tempList.insert(index," ")
    tempList.insert(index+2," ")
    return ''.join(tempList)





#------------------------- Printing part result --------------------
def dic_iterate_print(dic):
    '''
    Obtain map, iterate print the detail
    '''
    for i in dic.keys():
        dic_single_print(dic,i)


def dic_single_print(dic,code):
    '''
    Obtain map and code number, print the detail
    '''
    print('The code {} is pair with {}, {} and {}\n'
        .format(code, dic[code][0], dic[code][1], dic[code][2]))


def print_results(results):
    '''
        This function is use to print the result.

        Args:
            results cv
        output:
            results toString
    '''
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        
    print("-------------------------")

from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score

def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred, average='micro'), 3)
    recall = round(recall_score(labels, pred, average='micro'), 3)
    f1 = f1_score(labels, pred, average='macro')
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / F1: {}  / Latency: {}ms'.format(
        name, accuracy, precision, recall,f1, round((end - start)*1000, 1)))






















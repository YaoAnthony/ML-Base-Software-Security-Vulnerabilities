#TODO: Explore the datase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import re
import random
import string
import operator


from util import get_code_from_file,split_hold_txt

 


class ScoreInfo:
    score = 0
    content = ''
    target = 0

class NGram:

    def __init__(self,fileDic,model="bigram"):

        self._dicWordFrequency = dict() #Word frequency P(W1)

        self._dicPhraseFrequency = dict() #词段频 P(W1W2)

        self._dicPhraseProbability = dict() #词段概率 P(W1|W2)

        self._dicCoding = fileDic

        self.model = model

  


    def print_result_NGram(self):
        print("frequency of word")
        for key in self._dicWordFrequency.keys():
            print('%s\t%s'%(key,self._dicWordFrequency[key]))
        print("***********word***********")
        for key in self._dicPhraseFrequency.keys():
            print('%s\t%s'%(key,self._dicPhraseFrequency[key]))
        print('***********the probability of code***********')
        for key in self._dicPhraseProbability.keys():
            print('%s\t%s'%(key,self._dicPhraseProbability[key]))
    
    
    def append(self):
        '''
        training the N-gram model

        :param content:  code

        '''
        #clear sapce
        a = []
        keys = []

        #collect all code
        for key in self._dicCoding.keys():
            a.extend(split_hold_txt(get_code_from_file(self._dicCoding,key)))
        
        if self.model == "bigram":
        #bigram
            ie = self.getIterator_bigram(a) #2-Gram model
            for w in ie:
                #word frequency
                k1 = w[0] # i
                k2 = w[1] # love

                #If k1 doesn't appear, then setting k1
                if k1 not in self._dicWordFrequency.keys(): 
                    self._dicWordFrequency[k1] = 0 # key[i] = 0

                if k2 not in self._dicWordFrequency.keys():
                    self._dicWordFrequency[k2] = 0 # key[love] = 0
                self._dicWordFrequency[k1] += 1 # key[i] = 1
                self._dicWordFrequency[k2] += 1 # key[love] = 1

                #词段频
                key = "%s %s"%(w[0],w[1]) # i love
                keys.append(key) # i love

                if key not in self._dicPhraseFrequency.keys():
                    self._dicPhraseFrequency[key] = 0
                    
                self._dicPhraseFrequency[key] += 1
                #[i love] = 1

            #词段概率
            for word in keys:

                w1w2 = word.split()
                w1 = w1w2[0]
                w1Freq = self._dicWordFrequency[w1]
                w1w2Freq = self._dicPhraseFrequency[word]
                # P(w1w2|w1) = w1w2出现的总次数/w1出现的总次数 = 827/2533 ≈0.33 , 即 w2 在 w1 后面的概率

                self._dicPhraseProbability[word] = round(w1w2Freq/w1Freq,2)
            pass

        else:
        #trigram 
            ie = self.getIterator_trigram(a) #3-Gram model
            for w in ie:
                #word frequency
                k1 = w[0]
                k2 = w[1]
                k3 = w[2]
                #If k1 doesn't appear, then setting k1
                if k1 not in self._dicWordFrequency.keys():
                    self._dicWordFrequency[k1] = 0

                if k2 not in self._dicWordFrequency.keys():
                    self._dicWordFrequency[k2] = 0

                if k3 not in self._dicWordFrequency.keys():
                    self._dicWordFrequency[k3] = 0  

                self._dicWordFrequency[k1] += 1
                self._dicWordFrequency[k2] += 1
                self._dicWordFrequency[k3] += 1

                #词段频
                key = "%s %s %s"%(w[0],w[1],w[2])
                keys.append(key)

                if key not in self._dicPhraseFrequency.keys():
                    self._dicPhraseFrequency[key] = 0
                    
                self._dicPhraseFrequency[key] += 1
    

            #词段概率
            for word in keys:

                w1w2 = word.split()
                w1 = w1w2[0]
                w1Freq = self._dicWordFrequency[w1]
                w1w2Freq = self._dicPhraseFrequency[word]
                # P(w1w2|w1) = w1w2出现的总次数/w1出现的总次数 = 827/2533 ≈0.33 , 即 w2 在 w1 后面的概率

                self._dicPhraseProbability[word] = round(w1w2Freq/w1Freq,2)
            pass

    def getIterator_bigram(self,txt):
        '''
            bigram 模型迭代器

            :param txt: 一段话或一个句子

            :return: 返回迭代器,item 为 tuple,每项 2 个值

        '''
        #only one word not good for n-gram
        if len(txt)<2:
            return txt
            # 0 ~ ct
        for i in range(len(txt)-1):
            w1 = txt[i]
            w2 = txt[i+1]
            yield (w1,w2)


    def getIterator_trigram(self,txt):
        if len(txt)<3:
            return txt
            # 0 ~ ct
        for i in range(len(txt)-2):
            w1 = txt[i-1]
            w2 = txt[i]
            w3 = txt[i+1]
            yield (w1,w2,w3)


    def getScore(self,code):
        '''
        使用 ugram 模型计算 str 得分
        :param txt: 
        :return: score object
        '''
        print(self._dicCoding[code][2])
        txt = split_hold_txt(get_code_from_file(self._dicCoding, code))

        ie = self.getIterator_bigram(txt) if self.model == "bigram" else self.getIterator_trigram(txt)
        
        score = 1
        fs = []
        for w in ie:
            if self.model == "bigram":
                key = '%s %s'%(w[0],w[1])
                freq = self._dicPhraseProbability[key]
                fs.append(freq)
                score = freq * score
            else:
                #word frequency
                key = '%s %s %s'%(w[0],w[1],w[3])
                freq = self._dicPhraseProbability[key]
                fs.append(freq)
                score = freq * score
        print(fs)
        #return str(round(score,2))
        info = ScoreInfo()
        info.score = score
        info.content = txt
        info.target = target
        return info
import numpy as np
from gensim import similarities,corpora,models #tf-idf model
from gensim.models import Word2Vec,KeyedVectors
from collections import defaultdict
from itertools import islice
from util import split_hold_txt
import pprint
import pickle

from nltk.tokenize import WordPunctTokenizer


def training_embedding(fileDic, df, save_name, size = 100, window = 5, 
    sg=1, hs=0,negative=5,min_count = 5, iter = 5):
    '''
        Training the corpus
        sort the weigh
        remove words that appear in every category
        Comine them, getting feature collection
        Normalization and save as model
        CBOW模型在小的数据集上面表现不错，在大的数据集里，skip-gram表现更好
        parm:
            fileDic : dictionary
            df : Dataframe
            size : dimension of word, default 100
            window : Word vector context maximum distance
            sg : 0 - CBOW model, 1 - skip model
            hs : 0 - negative sampling, 1 - Hierarchical Softmax
            negative: If we choice nagative sampling, then require the number of nagative sampling
            min-count : the minimum word frequency of the word vector
            iter : Maximum number of iterations in stochastic gradient descent method
        
        output:
            Model : embedding model
    '''
    
    #Getting the code as array
    tokenizer = WordPunctTokenizer()
    code = []
    for row in df.itertuples():
        code += split_hold_txt(getattr(row, 'code'))
        #split_hold_txt(getattr(row, 'code'))
    model = Word2Vec(code, vector_size=size, window=window,sg = sg, hs =hs, min_count=min_count, workers=4)
    try:
        model.wv.save(save_name)
        print("model is saved as \"{}\" ".format(save_name))
        del model
    except:
        print("save the model fail")
    

def tfidf(code,slope=0.25, normalize = True,threshold=0.25, topk = 3):
    '''
        
    '''
    code = split_hold_txt(code,split_type = 0)
    if len(code) == 0 or len(code) == 1:
        return []
    else:
        # Count word frequencies
        frequency = defaultdict(int)
        
        for text in code: #every sentence
                for token in text: #every token
                    frequency[token] += 1

        # Only keep words that appear more than once 
        processed_corpus = [[token for token in text if frequency[token] > 0] for text in code]
        #pprint.pprint(processed_corpus)
        dictionary = corpora.Dictionary(processed_corpus)  
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        
        # train the model
        tfidf = models.TfidfModel(bow_corpus,slope=slope, normalize=normalize)
        #print(tfidf[bow_corpus[1]][5][0])
        tfidf = sorted(tfidf[bow_corpus[1]],key=lambda x:x[1],reverse=True)
        return list(islice([dictionary[w] for w,score in tfidf if score > threshold],topk))

def normalization(data):
    # Min-Max normalization
    return (data - np.min(data))/ (np.max(data) - np.min(data))



def p():
    model = Word2Vec.load('./w2v.wordVector')  #WORD_MODEL是我已经生成的模型
    
    print(wv.index2word())    #获得所有的词汇
    for word in wv.index2word():
        print(word,model[word])








    


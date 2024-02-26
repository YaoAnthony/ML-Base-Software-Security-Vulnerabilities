 #---------------------------------embedding.py




    # index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
    
    #计算词的重要性
    #tf_idf = tfidf[dictionary.doc2bow(code)]
#     tfidf_vec = []
#     for i in range(len(bow_corpus)):
#         string_tfidf = tfidf[bow_corpus[i]]
#         tfidf_vec.append(string_tfidf)
#     maxValue = -1
#     #selected_W
#     for sentence in tfidf_vec:
#         a = sorted(sentence,key=(lambda x: x[1]))
#         print(a)

#     #print(tfidf[bow_corpus[1]][5][0])

# from sklearn.feature_extraction.text import TfidfTransformer 
# from sklearn.feature_extraction.text import CountVectorizer

# def tfidf2(corpus, size =3):
#     vectorizer = TfidfTransformer()
#     tdm = vectorizer.fit_transform(corpus)
#     space = vectorizer.vocabulary_
#     print(space)
    










 #计算哪个句子的重要
    # maxSentenceCount = 0
    # w_sentence = ""
    # for sentence in code:
    #     #获得一个词的权重真的对获得一个代码的表征有帮助嘛
    #     tf_idf = tfidf[dictionary.doc2bow(sentence)]
    #     temp = 0
    #     for tup in tf_idf:
    #         temp += tup[1]
    #     temp /= (len(tf_idf) +0.01)
    #     if maxSentenceCount < temp :
    #         w_sentence = sentence
    # return w_sentence


    # def training_embedding2(fileDic,df):
#     code = []
#     for row in df.itertuples():
#         code += split_hold_txt(getattr(row, 'code')) #hold code in every file
#     model = Word2Vec(code, vector_size=100, window=5, min_count=1, workers=4)
#     #vector size: dimension, 
#     model.save('word2vec.model')

# def model_s():
#     loaded_model = KeyedVectors.load('word2vec.model', mmap='r')
#     print(len(loaded_model.wv.keys()))



    #pprint.pprint(processed_corpus)
    #Training model
    # sentences = [['first', 'sentence'], ['second', 'sentence']]
    # # sentences: data 
    # # size: dimension 
    # # window: 词向量上下文最大距离
    # # min_count：需要计算词向量的最小词频
    # # workers：完成训练过程的线程数，默认为1即不使用多线程
    # # https://blog.csdn.net/HappyCtest/article/details/85091686
    # model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)


    # model.save('word2vec.model')  # 保存模型
    # loaded_model = Word2Vec.load('word2vec.model')  # 加载模型
    # model.train([["hello", "world"]], total_examples=1, epochs=1)

    # wv = model.wv # 得到训练之后的词向量
    # wv.save('word_vector') # 保存word vectors
    # loaded_wv = KeyedVectors.load('word_vector', mmap='r') # 加载保存的word vectors 









#--------------------------------------main.py


#Split the training set
    # vun_fileDic = defaultdict(list)
    # non_fileDic = defaultdict(list)
    
    # for key in fileDic.keys():
    #     if int(fileDic[key][2]) == 1:
    #         vun_fileDic[key].append(fileDic[key][0])
    #         vun_fileDic[key].append(fileDic[key][1])
    #         vun_fileDic[key].append(fileDic[key][2])
    #     else:
    #         non_fileDic[key].append(fileDic[key][0])
    #         non_fileDic[key].append(fileDic[key][1])
    #         non_fileDic[key].append(fileDic[key][2])

    #N-gram training
    # vun_NGram = NGram(vun_fileDic)
    # vun_NGram.append()
    #vun_NGram.print_result_NGram()
    # non_NGram = NGram(non_fileDic)
    #non_NGram.append()


    # txt = "static int __init sb1250_pcibios_init(void)"
    # c = vun_NGram.getScore(split_hold_txt(txt))
    # print(c.score)
    
    #Scoring testing code, and collect them to dictionary
    # vun_arr = []
    # non_arr = []

    #Data Frame [score1, score2, vun_target]
    #df = create_DF(vun_arr,non_arr)
    
    #SVM
    #svm(df)


    #---Test---
    # #test for iteration
    # a = []
    # for key in fileDic.keys():
    #     if int(fileDic[key][2]) == 1:
    #         a.extend(split_hold_txt(get_code_from_file(fileDic,key)))

    #test for embedding package
    
    #Test for split code
    # print (split_hold_txt(get_code_from_file(fileDic, \
    #         selected_test(fileDic, "262705",print_code=True))))

    #selected_test(fileDic, "262705", print_code = True)

    #Test for embedding
    #training_embedding(fileDic)


    #Test for N-gram
    #nGram.append()
    #nGram.print_result_NGram()
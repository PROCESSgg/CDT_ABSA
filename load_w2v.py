import os
import copy
import torch
import random
import gensim
import linecache
import numpy as np

# load pretrained word emb
#按照word_list中的单词，从glove中读取词向量，得到word_list顺序的所有词向量，glove中没有的就用0向量初始化。
def load_pretrained_embedding(glove_dir, word_list, dimension_size=300, encoding='utf-8'):
    pre_words = []
    count = 0
    
    with open(glove_dir+"/glove_words.txt", 'r',encoding=encoding) as fopen:
        for line in fopen:
            pre_words.append(line.strip())
    word2offset = {w: i for i, w in enumerate(pre_words)}

    word_vectors = []
    for word in word_list:
        if word in word2offset:
            #从名为第一个参数的文件中得到第第二个参数行。这个函数从不会抛出一个异常–产生错误时它将返回”（换行符将包含在找到的行里）。如果文件没有找到，这个函数将会在sys.path搜索。
            line = linecache.getline(glove_dir+"/glove.840B.300d.txt", word2offset[word]+1)
            #strip函数，默认参数是去掉字符串头尾的换行和空白格，判断要找的单词和词向量首部单词是否对号
            assert(word == line[:line.find(' ')].strip())
            #np.fromstring函数是从字符串中按照sep=指定的分隔符转化到narray数组中。
            word_vectors.append(np.fromstring(line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
            count += 1
        else:
            # init zero
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
    print("Loading {}/{} words from vocab...".format(count, len(word_list)))

    return word_vectors


#文本相似度计算
#链接https://blog.csdn.net/laojie4124/article/details/93756562

import re
import jieba
import pickle
import numpy as np
import pandas as pd
from gensim import corpora
from gensim import models

jieba.load_userdict('userdict,txt')
filepath = r'stopwords.txt'
stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()] # 提取停用词

with open('data.pk','rb') as f:
    all_dick,idf_dict = pickle.load(f)

def read_file(file_path):
    with open(file_path,'r',encoding='utf-8-sig') as f:
        fina_outlist = [line.strip() for line in f.readlines()]
    return fina_outlist

def read_file2matrix(file_path):
    fina_outlist = []
    with open(file_path,'r',encoding='utf8-sig') as f:
        for line in f.readlines():
            outlist = [float(i) for i in line.strip().split('') if i != '']
            fina_outlist.append(outlist)
    return fina_outlist

def split_word(words):
    word_list = jieba.cut_for_search(words.lower().strip(),HMM=True)
    word_list = [i for i in word_list if i not in stopwords and i!= ' ']
    return word_list

def make_word_freq(word_list):
    freword = { }

    if i in word_list:
        if str(i) in freword:
            freword[str(i)] += 1
        else:
            freword[str(i)] = 1
    return freword

def make_ifidf(word_list,all_dick,idf_dict):
    length = len(word_list)
    word_list = [word for word in word_list if word in all_dick]
    word_freq = make_word_freq(word_list)
    w_dic = np.zeros(len(all_dick))

    for word in word_list:
        ind = all_dick[word]
        idf = idf_dict[word]
        w_dic[ind] = float(word_freq[word]/length)*float(idf)
    return w_dic

def Cos_Distance(vector1,vector2):
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)
    return float(np.sum(vec1 * vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def similarity_words(vec,vecs_list):
    Similarity_list = []
    for vec_i in vecs_list:
        Similarity = Cos_Distance(vec,vec_i)
        Similarity_list.append(Similarity)
    return  Similarity_list

def main(words, file_path, readed_path):
    words_list = read_file(file_path)
    vecs_list = read_file2matrix(readed_path)
    word_list = split_words(words)
    vec = make_tfidf(word_list,all_dick,idf_dict)
    similarity_lists = similarity_words(vec, vecs_list)
    sorted_res = sorted(enumerate(similarity_lists), key=lambda x: x[1]) # 按相似度排序
    outputs = [[words_list[i[0]],i[1]] for i in sorted_res[-10:]] # 取前十个相似的
    return outputs

#测试
# words = '小米8 全面屏游戏智能手机 6GB+128GB 黑色 全网通4G 双卡双待  拍照手机'
# words = '荣耀 畅玩7X 4GB+32GB 全网通4G全面屏手机 标配版 铂光金'
words = 'Apple iPhone 8 Plus (A1864) 64GB 深空灰色 移动联通电信4G手机'
# words = '小米8'
# words = "黑色手机"
# words = 'Apple iPhone 8'
# words = '索尼 sony'
file_path = r'MobilePhoneTitle.txt'
readed_path = r"MobilePhoneTitle_tfidf.txt"
outputs = main(words, file_path, readed_path)
# print(outputs)
for i in outputs[::-1]:
    print(i[0] + '     ' + str(i[1]))

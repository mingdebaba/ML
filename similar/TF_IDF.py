from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
 
def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
 
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
 
s1 = '你在干嘛呢'
s2 = '你在干什么呢'
print(tfidf_similarity(s1, s2))
#https://blog.csdn.net/m0_37870649/article/details/103090937?spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-12.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-12.pc_relevant_antiscanv2&utm_relevant_index=14
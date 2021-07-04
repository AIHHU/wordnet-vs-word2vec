# wordnet-vs-word2vec
实验代码，比较了在常规词下，wordnet与word2vec相似度计算的结果

该项目的完整文件我将后续上传至我的百度AI studio
# 文件解释
embedding.npy 训练得到的词向量。

text8.txt word2vec的训练集，这里用于用频度为每次词赋唯一的ID，作为缩影，在embedding中调用该词的词向量，用于计算余弦相似度。

以上两个文件太大了，传不了，大家可以自己找个demo训练词向量，text8那种demo一般都会有。

wordnetVSword2vec.py 实验代码
# 实验设计
本实验目的为常规词语下wordnet的相似性与word2vec相似性的比较，本实验所用的常规词有:puppy,dog,cat,human为一类，都属于animal kingdom,在理想情况下，本实验预期看到puppy和dog相似性高,（puppy,cat）和（dog,cat）的相似度近似，human和其它相似度大相径庭（但不排除狗是人类好朋友这种情况），最后一组实验主要测试动词间的相似性。
# 实验工具
word2vec训练到的词向量 

environment: numpy nltk

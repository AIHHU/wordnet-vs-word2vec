from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import numpy as np
brown_ic=wordnet_ic.ic('ic-brown.dat')
puppy=wn.synset('puppy.n.01')
dog=wn.synset('dog.n.01')
human=wn.synset('human.n.01')
cat=wn.synset('cat.n.01')
love=wn.synset('love.v.01')
like=wn.synset('like.v.02')
embedding_matrix = np.load('embedding.npy')
def load_text8():
    with open("text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()
    return corpus
corpus = load_text8()
def data_preprocess(corpus):
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus
corpus = data_preprocess(corpus)
def build_dict(corpus):
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    word2id_dict = dict()

    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
    return word2id_dict
word2id_dict=build_dict(corpus)

def get_cos(query1_token, query2_token, embed):
    W = embed
    x = W[word2id_dict[query1_token]]
    y = W[word2id_dict[query2_token]]
    cos = np.dot(x, y) / np.sqrt(np.sum(y * y) * np.sum(x * x) + 1e-9)
    return cos
def Similarity(word1,word2):
    result=[]
    result.append(word1.path_similarity(word2))
    result.append(word1.lch_similarity(word2))
    result.append(word1.wup_similarity(word2))
    result.append(word1.res_similarity(word2,brown_ic))
    result.append(word1.jcn_similarity(word2,brown_ic))
    result.append(word1.lin_similarity(word2,brown_ic))
    result.append(get_cos(word1.lemmas()[0].name(),word2.lemmas()[0].name(),embedding_matrix))
    return result

print(Similarity(puppy,dog))
print(Similarity(puppy,human))
print(Similarity(puppy,cat))
print(Similarity(dog,human))
print(Similarity(dog,cat))
print(Similarity(human,cat))
print(Similarity(love,like))




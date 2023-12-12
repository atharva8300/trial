import string
import nltk
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy.linalg import norm

f = open("input.txt",'r')
f.seek(0)
my_list = f.read().split("\n")
def remove_punctuation(test_list):
    new_list =[]
    for x in test_list:
        new_sentence = x.translate(str.maketrans('', '', string.punctuation))
        new_list.append(new_sentence)
    return new_list

def lower_case(test_list):
    new_list = []
    for x in test_list:
        new_list.append(x.lower())
    return new_list

def to_tokens(test_list):
    new_list = []
    for x in test_list:
        new_list.append(word_tokenize(x))
    # print("after tokenization")
    # print(new_list)
    return new_list

def remove_stopwords(test_list):
    english_stopwords = set(stopwords.words('english'))
    new_list =[]
    for x in test_list:
        filtered_words = [i for i in x if i not in english_stopwords]
        new_list.append(filtered_words)
    # print("\nafter removing stopwords")
    # print(new_list)
    return new_list

def lemontize(test_list):
    lemmatizer = WordNetLemmatizer()
    final_list = []
    for x in test_list:
        new_sentence = ""
        for y in x:
            new_sentence = new_sentence+lemmatizer.lemmatize(y)+" " 
        final_list.append(new_sentence)
    # print("\nafter lemmatization")
    # print(final_list)
    return final_list

def remove_last_space(test_list):
    new_list = []
    for x in test_list:
        y = x[:-1]
        new_list.append(y)
    return new_list

def preprocess(test_list):
    test_list = remove_punctuation(test_list)
    test_list = lower_case(test_list)
    test_list = to_tokens(test_list)
    test_list = remove_stopwords(test_list)
    test_list = lemontize(test_list)
    test_list = remove_last_space(test_list)
    return test_list

my_list = preprocess(my_list)
print("final preprocessed list of sentences")
print(my_list)

##################### task 2 #########################
word_list = []
for x in my_list:
    for y in x.split():
        word_list.append(y)

word_list = list(set(word_list))
rows = len(word_list)
columns = len(my_list)
tf_idf_vector = np.zeros((rows,columns))

def sen_with_word(word):
    count = 0
    for x in my_list:
        if word in x:
            count+=1
#     print(f"{word} appears in {count} sentences")
    return count

def calculate_tf_idf(x,y):
    word = word_list[x]
    sentence = my_list[y]
    tf = sentence.count(word)
#     print(f"tf of {word} = {tf}")
    idf = math.log2(columns/sen_with_word(word))
#     print(f"idf of {word} = {idf}")
#     if tf*idf > 1:
#         print(x,y)
    return tf*idf

for x in range(rows):
    for y in range(columns):
        tf_idf_vector[x,y] = calculate_tf_idf(x,y)

# print(tf_idf_vector)

######################TASK 3##############################

G = nx.complete_graph(columns)
def cos_sim(u,v):
    a = tf_idf_vector[:,u]
    b = tf_idf_vector[:,v]
    cos = np.dot(a,b)/(norm(a)*norm(b))
    return cos

for u,v in G.edges:
    weight = cos_sim(u,v)
    G[u][v]['weight'] = weight

for u,v,data in G.edges(data=True):
    print(f"Edge ({u}, {v}) has weight: {data['weight']}")

pagerank_scores = nx.pagerank(G)

for node, score in pagerank_scores.items():
    print(f"Node {node}: PageRank Score {score}")

sorted_pageranks = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
def final_summary(n,sorted_pageranks):
    count = 0
    isSelected = [0]*columns
    for x,y in sorted_pageranks:
        if(count<n):
            isSelected[x] = 1
#             print(my_list[x])
            count+=1
    return isSelected

# checking for n=10
isSelected = final_summary(10,sorted_pageranks)
summary = []
for x in range(columns):
    if isSelected[x] == 1:
        print(my_list[x])
        summary.append(my_list[x])
with open('Summary_PR.txt', 'w') as file:
    for i in summary:
        file.write(i)
        file.write("\n")


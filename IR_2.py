import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import math
import timeit
import os
from bs4 import BeautifulSoup

path = "C:\\Users\\Raaed\\Documents\\SEM 5\\Information Retrieval Assignment\\Assignment 2\\corpus\\corpus\\fulltext"
start = timeit.default_timer()

total_doc = []
i = 0
for file in os.listdir(path):
    i = i+1
    if(i>400):
        break
    with open(os.path.join(path,file),'r') as f:
        data = f.read()
        bs_data = BeautifulSoup(data,"xml")
        b_catch = bs_data.find_all('catchphrase')
        doc = ""
        for c in b_catch:
            phrase = c.text
            k=phrase.split(">")
            doc= doc + k[1] + " "
        total_doc.append(doc)
    
stop = timeit.default_timer()
print('Import: ', stop - start)

start = timeit.default_timer()
tokenizer = RegexpTokenizer("[\w']+")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

for i in range(len(total_doc)):
    tokens = tokenizer.tokenize(total_doc[i])
    stopset = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stopset]
    stemmed_sentence = [stemmer.stem(w) for w in filtered_sentence]
    lemmatized_sentence = [lemmatizer.lemmatize(w) for w in stemmed_sentence]
    total_doc[i] = lemmatized_sentence

stop = timeit.default_timer()
print('Preprocess: ', stop - start)

#Initializing the parameters
start = timeit.default_timer()

k_shingles = 2
shingle_set = {}

# Constructing the K shingle_set
##count=0
for i in range(len(total_doc)):
    for j in range(len(total_doc[i])- k_shingles):
        shingle = total_doc[i][j:j+k_shingles]
        shingle = ' '.join(shingle)
        if shingle not in shingle_set.keys():
            a = np.zeros(len(total_doc))
            a[i] = 1
            shingle_set[shingle] = a
        else:
            shingle_set[shingle][i] = 1

i=0 ##shingle_id mapped to shingles
shingleID = [i for i in range(len(shingle_set))]
shingleMap = []
for key in shingle_set.keys():
    shingleMap.append([i,key])
    i+=1

stop = timeit.default_timer()
print('Shingling: ', stop - start)

start = timeit.default_timer()
#Generating N hash functions 
N_HASHES = 10
hash_funcs = []
for i in range(N_HASHES):
    perm = random.sample(shingleID, len(shingleID))
    while perm in hash_funcs:
        perm = random.sample(shingleID, len(shingleID))
    hash_funcs.append(perm)

stop = timeit.default_timer()
print('Hashing: ', stop - start)
     
start = timeit.default_timer()
#Calculating the signature matrix 
signature = np.full(shape=(N_HASHES, len(total_doc)), fill_value=len(shingle_set)+5)
for i in range(N_HASHES):
    for id in shingleID:
        hash_id = hash_funcs[i][id]
        for p in range(len(total_doc)):
            if shingle_set[shingleMap[id][1]][p]==1:
                if hash_id<signature[i][p]:
                    signature[i][p] = hash_id

stop = timeit.default_timer()
print('Signature Matrix: ', stop - start)

def jaccard(c1,c2):
    """
    Calculates the Jaccard similarity of Columns C1 and C2
    """
    a=0
    b=0
    c=0
    for key in shingle_set.keys():
        if shingle_set[key][c1] == shingle_set[key][c2] and shingle_set[key][c1] == 1:
            a+=1
        elif shingle_set[key][c1] == 1:
            b+=1
        elif shingle_set[key][c2] == 1:
            c+=1
    return a/(a+b+c)

def hamming(c1, c2):
    """
    Calculates the Hamming distance between Columns C1 and C2
    """
    a=0
    for key in shingle_set.keys():
        if shingle_set[key][c1] != shingle_set[key][c2]:
            a+=1
    return a

def cosine(c1, c2):
    """
    Calculates the Cosine similarity of Columns C1 and C2
    """
    a=0
    b=0
    c=0
    for key in shingle_set.keys():
        if shingle_set[key][c1] == shingle_set[key][c2] and shingle_set[key][c1] == 1:
            a+=1
        if shingle_set[key][c1] == 1:
            b+=1
        if shingle_set[key][c2] == 1:
            c+=1
    return a/math.sqrt(b*c)

def euclidean(c1, c2):
    """
    Calculates the Euclidean Distance between Columns C1 and C2
    """
    a=0
    for key in shingle_set.keys():
        a+=(shingle_set[key][c1]-shingle_set[key][c2])**2    
    return math.sqrt(a)

start = timeit.default_timer()

rows = 2
bands = math.ceil(len(signature)/rows)
bucket_size = 3
buckets = np.zeros(shape=(bands, len(total_doc)))

for band in range(bands):
    for j in range(signature.shape[1]):
        b = 0
        for i in range(rows):
            b+=signature[band*rows + i][j]%bucket_size
        buckets[band][j]=b

stop = timeit.default_timer()
print('Buckets and Bands: ', stop - start)

##Checking similarity LSH for Doc1
## Outputting candidate pairs
doc1 = int(input("Enter a document ID "))
start = timeit.default_timer()
threshold = 2
candidates = []
for c in range(signature.shape[1]):
    count = 0
    for band in range(bands):
        if c !=doc1:
            if buckets[band][c] == buckets[band][doc1]:
                count+=1
    if count>=threshold:
        candidates.append(c)

print("The candidate pairs are-")
for c in candidates:
    print(str(doc1)+","+str(c))

jacc = [(jaccard(doc1, c), c) for c in candidates]
jacc.sort(reverse=True)
cos = [(cosine(doc1, c),c) for c in candidates]
cos.sort(reverse=True)
euc = [(euclidean(doc1, c),c) for c in candidates]
euc.sort()
ham = [(hamming(doc1, c),c) for c in candidates]
ham.sort()

print("\nJaccard Similarity")
for c in range(len(candidates)):
    print("Doc" + str(jacc[c][1]) +" "+ str(jacc[c][0]))

print("\nCosine Similarity")
for c in range(len(candidates)):
    print("Doc" + str(cos[c][1]) +" "+ str(cos[c][0]))

print("\nEuclidean Distance")
for c in range(len(candidates)):
    print("Doc" + str(euc[c][1]) +" "+ str(euc[c][0]))

print("\nHamming Distance")
for c in range(len(candidates)):
    print("Doc" + str(ham[c][1]) +" "+ str(ham[c][0]))

stop = timeit.default_timer()
print('\nRetrieval: ', stop - start)

import pandas as pd
import numpy as np
import csv
import re
import string
import matplotlib.pyplot as plt

import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from Project.project_functions import getallrows, getclosed


#apply function to get data of ps2 games on sale
ps2data = getallrows(459)
#apply function to get data of ps2 games sold
ps2closed = getclosed(459)

#combine dataframes of on sale and closed ps2 games
pieces = [ps2data, ps2closed]
ps2dataframe = pd.concat(pieces, ignore_index=True)
#save to csv file
ps2dataframe.to_csv(r'project/data/ps2.csv')

#PREPROCESSING

ps2datacopy = ps2dataframe.copy()

titles = ps2datacopy['title']
titles = titles.str.lower()

#remove punctuation
titles = titles.str.replace('[{}]'.format(string.punctuation),' ')
#remove stopwords and extra whitespace

with open('stop-word-list_huuto.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        stopwords = row
stopwords = list(map(str.strip, stopwords))

def RemoveSpace(word):
    regex = re.compile(r'[\n\r\t\xa0]')
    word = regex.sub(" ", word)
    _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
    _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")

    word = _RE_COMBINE_WHITESPACE.sub(" ", word)
    word = _RE_STRIP_WHITESPACE.sub("", word)
    return word

#titles = titles.str.replace(f'\\b({"|".join(stopwords)})\\b','')

#titles2 = titles.apply(lambda x: (RemoveSpace(x)))

#titles2 = titles.
titlesa=[]
for word in titles:
    titlesa.append(" ".join(word.split()))

titles2 = pd.Series(titlesa,name="title")

#remove empty titles
titles2 = titles2[titles2 != '']
titles2[1440:1450]

dataset = titles2.str.split()
dataset[:10]

data = [d for d in dataset]

#create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

train_data = list(create_tagged_document(data))

print(train_data[:1])

#init the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(epochs = 40)

#build the Volabulary
model.build_vocab(train_data)

#train the Doc2Vec model
model.train(train_data, total_examples = model.corpus_count, epochs = model.epochs)

#test
#doesnt work wel
#model.most_similar(['honor'])

#########TEST WORD2VEC#########

sentences = dataset.copy()

m = Word2Vec(sentences, size =50, min_count=1, sg=1)
def vectorizer(sent,m):
    vec =[]
    numw =0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else: vec = np.add(vec, m[w])
            numw +=1
        except:
            pass

    return np.asarray(vec) / numw

l=[]
for i in sentences:
    l.append(vectorizer(i,m))
X = np.array(l)

#works better
#m.wv.most_similar(['honor'])
#('rising', 0.9943820238113403),
#('medal', 0.9924663305282593),
#('sun', 0.9921729564666748),
# ('frontline', 0.9775959253311157),
#('european', 0.9692680239677429),
#('assault', 0.9476239681243896),
#('of', 0.9251711368560791),
#('vanguard', 0.9188636541366577),
#('ale', 0.8655203580856323),
#('40', 0.8401911854743958)


#########K MEANS TEST#########

#apply kmeans to X
wcss = []
for i in range(1,50):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plot
plt.plot(range(1,50),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#test k means clustering efficiency
n_clusters = 650
clf = KMeans(n_clusters = n_clusters, max_iter =200, init = 'k-means++', n_init=1)
labels = clf.fit_predict(X)
print(labels)
for index, sentence in enumerate(sentences):
    print (str(labels[index])+ ":" + str(sentence))

#make a dataframe from kmeans clusters
kmean_label = []
kmean_title = []

for index, sentence in enumerate(sentences):
    kmean_label.append(labels[index])
    kmean_title.append(sentence)

kmean_df = pd.DataFrame(list(zip(kmean_label, kmean_title)), columns =['label', 'title'])
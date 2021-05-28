import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import json
import string
from nltk.stem.snowball import SnowballStemmer

######BASICS######

#create normal lists with numbers 0 - (1e7-1)
forA = list(range(0,10000000))
forB = list(range(0,10000000))



#########ASSERTION#########
assert len(forA) == 10000000, 'The length of list forA does not match'
assert len(forB) == 10000000, 'The length of list forB does not match'
###########################



#define function to sum element-wise two arrays together
def add_with_for(a, b):
    c = []
    for i in range(len(a)):
        c.append((a[i] + b[i]))
    return c

#create a new array with the function created above
forC = add_with_for(forA,forB)



#########ASSERTION#########
assert len(forC) == 10000000, 'The length of list forC does not match'
assert forC[0] == 0, 'index 0 in forC does not match'
assert forC[1] == 2, 'index 1 in forC does not match'
assert forC[2] == 4, 'index 2 in forC does not match'
###########################



#DO THE ABOVE WITH NUMPY

#create numpy arrays with numbers 0 - (1e7-1)
numA = np.array(range(0,10000000))
numB = np.array(range(0,10000000))



#########ASSERTION#########
assert len(numA) == 10000000, 'The length of list numA does not match'
assert len(numB) == 10000000, 'The length of list numB does not match'
###########################



#define function to sum element-wise two arrays together with numpy
def add_with_numpy(a, b):
    c = np.add(a,b)
    return c

#create a new numpy array with the function created above
numC = add_with_numpy(forA,forB)



#########ASSERTION#########
assert len(numC) == 10000000, 'The length of list numC does not match'
assert numC[0] == 0, 'index 0 in numC does not match'
assert numC[1] == 2, 'index 1 in numC does not match'
assert numC[2] == 4, 'index 2 in numC does not match'
###########################



##########ARRAY MANIPULATION##########

#STEP a: array
arrA1 = np.array(range(0,100))
arrA1.reshape(10,10)

#STEP b: array
arrB1 = np.zeros(50)
arrB2 = np.ones(50)
arrB3 = np.dstack((arrB1,arrB2))
arrB3 = arrB3.reshape(10,10)

arrB1 = np.array([0.,1.])
arrB2 = np.tile(arrB1, (10,5))

#STEP c: array
arrC1 = np.zeros(10) - 1
arrC1 = np.diag(arrC1) + 1

#STEP d: array
arrD1 = np.zeros(10) - 1
arrD1 = np.diag(arrD1)
arrD1 = np.fliplr(arrD1) + 1

#STEP e: det(CD) and det(C) * det(D)
detProd = round(np.linalg.det(arrC1.dot(arrD1)), 2)
prodDet = round((np.linalg.det(arrC1) * np.linalg.det(arrD1)), 2)



#########ASSERTION#########
assert detProd == prodDet, ('The determinant of C1 and D1 should be the ' +
                            'same as the product of det(C1) and det(D1)')
###########################



#######SLICING#######

########SLICING EXERCISE 1########

#A) load boston housing dataset
#shape should be (506, 13)
dataset = load_boston()
X = dataset['data']
y = dataset['feature_names']



#########ASSERTION#########
assert X.shape == (506, 13), 'Boston dataset shape does not match'
###########################



#B) rows where crime > 1
#len should be 174
rows = np.where(X[:,0] > 1)
result = X[rows]



#########ASSERTION#########
assert len(result) == 174, 'result length of crime > 1 does not match'
###########################



#C) rows where pupil-to-teach > 16 and < 18 (exclusive)
#len should be 100
rows = np.where((X[:,10] > 16) & (X[:,10] < 18))
result = X[rows]



#########ASSERTION#########
assert len(result) == 100, ('result length of pupil-to-teach ' +
                            '> 16 and < 18 does not match')
###########################



#D) mean NO concentration for homes of median price > $25000
#should be around 0.492.
rows = np.where((dataset['target'] > 25))
result = X[rows]
nox = result[:,4]
noxmean = np.mean(nox)



#########ASSERTION#########
assert round(noxmean, 3) == 0.492, ('result of mean NO concentration ' +
                                    'does not match')
###########################



########SLICING EXERCISE 2########

#read data and define list of stop words
data = pd.read_json('data/Automotive_5.json', lines=True)
stopwords = ["a", "about", "above", "above", "across", "after",
             "afterwards", "again", "against", "all", "almost",
             "alone", "along", "already", "also","although","always",
             "am","among", "amongst", "amoungst", "amount",  "an",
             "and", "another", "any","anyhow","anyone","anything",
             "anyway", "anywhere", "are", "around", "as",  "at",
             "back","be","became", "because","become","becomes",
             "becoming", "been", "before", "beforehand", "behind",
             "being", "below", "beside", "besides", "between", "beyond",
             "bill", "both", "bottom","but", "by", "call", "can",
             "cannot", "cant", "co", "con", "could", "couldnt", "cry",
             "de", "describe", "detail", "do", "done", "down", "due",
             "during", "each", "eg", "eight", "either", "eleven","else",
             "elsewhere", "empty", "enough", "etc", "even", "ever",
             "every", "everyone", "everything", "everywhere", "except",
             "few", "fifteen", "fify", "fill", "find", "fire", "first",
             "five", "for", "former", "formerly", "forty", "found",
             "four", "from", "front", "full", "further", "get", "give",
             "go", "had", "has", "hasnt", "have", "he", "hence", "her",
             "here", "hereafter", "hereby", "herein", "hereupon", "hers",
             "herself", "him", "himself", "his", "how", "however",
             "hundred", "i", "ive, " "ie", "if", "in", "inc", "indeed",
             "interest", "into", "is", "it", "its", "itself", "keep",
             "last", "latter", "latterly", "least", "less", "ltd", "made",
             "many", "may", "me", "meanwhile", "might", "mill", "mine",
             "more", "moreover", "most", "mostly", "move", "much", "must",
             "my", "myself", "name", "namely", "neither", "never",
             "nevertheless", "next", "nine", "no", "nobody", "none",
             "noone", "nor", "not", "nothing", "now", "nowhere", "of",
             "off", "often", "on", "once", "one", "only", "onto", "or",
             "other", "others", "otherwise", "our", "ours", "ourselves",
             "out", "over", "own","part", "per", "perhaps", "please", "put",
             "rather", "re", "same", "see", "seem", "seemed", "seeming",
             "seems", "serious", "several", "she", "should", "show", "side",
             "since", "sincere", "six", "sixty", "so", "some", "somehow",
             "someone", "something", "sometime", "sometimes", "somewhere",
             "still", "such", "system", "take", "ten", "than", "that", "the",
             "their", "them", "themselves", "then", "thence", "there",
             "thereafter", "thereby", "therefore", "therein", "thereupon",
             "these", "they", "thickv", "thin", "third", "this", "those",
             "though", "three", "through", "throughout", "thru", "thus",
             "to", "together", "too", "top", "toward", "towards", "twelve",
             "twenty", "two", "un", "under", "until", "up", "upon", "us",
             "very", "via", "was", "we", "well", "were", "what", "whatever",
             "when", "whence", "whenever", "where", "whereafter", "whereas",
             "whereby", "wherein", "whereupon", "wherever", "whether",
             "which", "while", "whither", "who", "whoever", "whole", "whom",
             "whose", "why", "will", "with", "within", "without", "would",
             "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

#lowercase the text and remove unctuation
data['reviewText'] = data['reviewText'].str.lower()
data['reviewText'] = data['reviewText'].str.replace('[{}]'.format(string.punctuation), '')

#remove words included in the 'stopwords' list
data['reviewText'] = data['reviewText'].str.replace(f'\\b({"|".join(stopwords)})\\b','')

#apply nltk.snowballstemmer
stemmer = SnowballStemmer("english")
data['reviewText'] = data['reviewText'].apply(lambda x : filter(None,x.split(" ")))     # Split sentence into words
data['reviewText'] = data["reviewText"].apply(lambda x: [stemmer.stem(y) for y in x])   # Then stemm it
data['reviewText'] = data['reviewText'].apply(lambda x : " ".join(x))

#filter the good and bad reviews into own dataframes
pos = data[(data.overall == 4) | (data.overall == 5)]
neg = data[(data.overall == 1) | (data.overall == 2)]

reviewText = ['reviewText']

#save good and bad reviews to own txt-files
pos[reviewText].to_csv(r'data/positive_reviews.txt', index=False)
neg[reviewText].to_csv(r'data/negative_reviews.txt', index=False)

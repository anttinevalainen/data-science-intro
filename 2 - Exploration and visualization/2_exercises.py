import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import scipy

########EXERCISE 1 - TITANIC########

#read in data
data = pd.read_csv(r'data\titanic\train.csv', sep=',', header=0)
data = data.drop(['Name'], axis=1)

#define function for getting the deck number from cabin numbers
def deckNumber(i):
    if pd.isna(i):
        return i
    else:
        return ord(i[0])

#apply function to data
data['Deck'] = data['Cabin'].apply(lambda x : deckNumber(x))

#fill na-values
data['Deck'] = data['Deck'].fillna(data['Deck'].mode()[0])
data['Deck'] = data['Deck'] - data['Deck'].min()
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Age'] = data['Age'].round(1)

#change categorical values to numeric values
dic = {'female': 0, 'male': 1}
data['Sex'] = data['Sex'].map(dic)

dic2 = {'S': 0, 'C': 1, 'Q': 2}
data['Embarked'] = data['Embarked'].map(dic2)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

#drop useless rows
data = data.drop(['Ticket', 'Cabin'], axis=1)

#save data as csv/json
data.to_csv(r'data\titanic_imputated.csv')
data.to_json(r'data\titanic_imputated.json', orient='records')


########EXERCISE 2 - TITANIC v2.0########

#read data
data2 = pd.read_csv(r'data\titanic_imputated.csv', sep=',', header=0, index_col=0)

#for categorical variables, find out the mode. For numerical variables, calculate the median value
#constructing 'average passenger' function

def average_passenger(dataframe):

    if dataframe['Pclass'].mode()[0] > 2:
        print('Class: Third class')
    elif dataframe['Pclass'].mode()[0] < 2:
        print('Class: First class')
    else:
        print('Class: Second class')

    if dataframe['Sex'].mode()[0] > 0:
        print('Sex: Male')
    else:
        print('Sex: Female')

    print('Age: ' + str(round(dataframe['Age'].mean(), 2)))
    print('Siblings/spouse on board: ' + str(dataframe['SibSp'].mode()[0]))
    print('Parents/children on board: ' + str(dataframe['Parch'].mode()[0]))

    if dataframe['Embarked'].mode()[0] > 1:
        print('Port of Embarkation: Queenstown')
    elif dataframe['Embarked'].mode()[0] < 1:
        print('Port of Embarkation: Southampton')
    else:
        print('Port of Embarkation: Cherbourg')

    if dataframe['Survived'].mode()[0] > 0:
        print('Survived the disaster')
    else:
        print('Did not survive the disaster')

    print('***')

#for quriosity, construct a function to see statistical
#key values for a group within the dataframe
def passenger_statistics(dataframe):
    print('Age:')
    print('Min: ' + str(dataframe['Age'].min()))
    print('Max: ' + str(dataframe['Age'].max()))
    print('Mean: ' + str(dataframe['Age'].mean()))
    print('Mode: ' + str(dataframe['Age'].mode()[0]))
    print('Standard deviation: ' + str(dataframe['Age'].std()))

    print('Sex:')
    print('% of women: ' + str(len(dataframe.loc[(data2['Sex'] == 0)])
                               / len(dataframe) * 100))
    print('% of men: ' + str(len(dataframe.loc[(data2['Sex'] == 1)])
                             / len(dataframe) * 100))

    print('Class:')
    print('% of 1st class passengers: ' + str(len(dataframe.loc[(data2['Pclass'] == 1)])
                                              / len(dataframe) * 100))
    print('% of 2nd class passengers: ' + str(len(dataframe.loc[(data2['Pclass'] == 2)])
                                              / len(dataframe) * 100))
    print('% of 3rd class passengers: ' + str(len(dataframe.loc[(data2['Pclass'] == 3)])
                                              / len(dataframe) * 100))

    print('Fare:')
    print('Average passenger fare: ' + str(dataframe['Fare'].mean()))

#define separate dataframes for titanic victims and survivors
victimfilter = (data2['Survived'] == 0)
survivefilter = (data2['Survived'] == 1)

vic = data2.where(victimfilter)
vic = vic.dropna()
sur = data2.where(survivefilter)
sur = sur.dropna()

#define separate dataframes for men and women passengers
womenfilter = (data2['Sex'] == 0)
menfilter = (data2['Sex'] == 1)

women = data2.where(womenfilter)
women = women.dropna()
men = data2.where(menfilter)
men = men.dropna()

#define average passengers for each group
avg_passenger = average_passenger(data2)
avg_victim = average_passenger(vic)
avg_survivor = average_passenger(sur)

#check the nearest matches to the average passenger based on the specifications above:
filter_passenger = ((data2['Survived'] == 0) & (data['Sex'] == 1) &
                    (data['Pclass'] == 3) & (data['Age'] < 30) & (data['Age'] > 29) &
                    (data['Embarked'] == 0) & (data['Parch'] == 0) & (data['SibSp'] == 0))

#check the nearest matches to the average victim based on the specifications above:
filter_victim = ((data2['Survived'] == 0) & (data['Sex'] == 1) &
                 (data['Pclass'] == 3) & (data['Age'] < 31) & (data['Age'] > 30) &
                 (data['Embarked'] == 0) & (data['Parch'] == 0) & (data['SibSp'] == 0))

#check the nearest matches to the average survivor based on the specifications above:
filter_survivor = ((data2['Survived'] == 1) & (data['Sex'] == 0) &
                   (data['Pclass'] == 1) & (data['Age'] <= 29) & (data['Age'] >= 28) &
                   (data['Embarked'] == 0) & (data['Parch'] == 0) & (data['SibSp'] == 0))

average_passenger = data2.where(filter_passenger)
average_passenger = average_passenger.dropna()
#multiple values because of the average age assigned to those, who have NaN value as age!!

average_victim = data2.where(filter_victim)
average_victim = average_victim.dropna()
#one match!

average_survivor = data2.where(filter_survivor)
average_survivor = average_survivor.dropna()
#one match!

print('Whole dataset:')
passenger_statistics(data2)
print('****************')
print('Survivors:')
passenger_statistics(sur)
print('****************')
print('Victims:')
passenger_statistics(vic)


#plot the Titanic passenger statistics

#create 3x1 plot
fig, (ax1,ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))


#survival by CLASS
ax = data2.groupby('Pclass')['Survived'].mean().plot.bar(
    ax = ax1,
    title = 'Survival rate by passenger Class'
    )
ax.set_xticklabels(['1st','2nd', '3rd'])
ax.set_yticklabels(['0', '20%','40%', '60%', '80%', '100%'])
ax.set_xlabel('Passenger Class')
ax.set_ylim(0.0,1.0)


#survival by SEX
ax = data2.groupby('Sex')['Survived'].mean().plot.bar(
    ax = ax2,
    title = 'Survival rate by Sex',
    sharey = True
    )
ax.set_xticklabels(['women','men'])
ax.set_xlabel('Sex')
ax.set_ylim(0.0,1.0)

#survival by EMBARKATION PORT
ax = data2.groupby('Embarked')['Survived'].mean().plot.bar(
    ax = ax3,
    title = 'Survival by port of embarkation',
    sharey = True
    )
ax.set_xticklabels(['Southampton','Cherbourg', 'Queenstown'])
ax.set_xlabel('Port of Embarkation')
ax.set_ylim(0.0,1.0)


#create 3x1 plot
fig, (ax1,ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))

#ticket fare paid by passengers by passenger numbers
ax1 = data2['Fare'].plot.hist(
    bins = 50,
    color = 'blue',
    ax = ax1,
    title = 'Ticket fare paid by passengers of Titanic'
    )
ax1.set_ylim(0, 350)
ax1.set_xlim(0, 500)
ax1.set_ylabel('Number of passengers')
ax1.set_xlabel('Ticket fare - Average: ' +
               str(round(data2['Fare'].mean(), 2)))
ax1.grid()

#ticket fare paid by survivors by survivor numbers
ax2 = sur['Fare'].plot.hist(
    bins = 50,
    color = 'lightblue',
    ax = ax2,
    title = 'Ticket fare paid by survivors of Titanic',
    sharey = True
    )
ax2.set_ylim(0, 350)
ax2.set_xlim(0, 500)
ax2.set_xlabel('Ticket fare - Average: ' +
               str(round(sur['Fare'].mean(), 2)))
ax2.grid()

#ticket fare paid by victims by victim numbers
ax3 = vic['Fare'].plot.hist(
    bins = 50,
    color = 'salmon',
    ax = ax3,
    title = 'Ticket fare paid by victims of Titanic',
    sharey = True
    )
ax3.set_ylim(0, 350)
ax3.set_xlim(0, 500)
ax3.set_xlabel('Ticket fare - Average: ' +
               str(round(vic['Fare'].mean(), 2)))
ax3.grid()


#create a single plot
fig, (ax1) = plt.subplots(1, 1, figsize=(10,5))

#bars for number of survivors in each class
sur['Pclass'].value_counts().sort_index().plot.bar(
    color = 'Blue',
    label = 'Survivors',
    ax = ax1,
    align = 'center',
    alpha = 0.5
    )
#bars for number of victims in each class
vic['Pclass'].value_counts().sort_index().plot.bar(
    color = 'Red',
    label = 'Victims',
    ax = ax1,
    align = 'edge',
    alpha = 0.5
    )

ax1.set_xlabel('Passenger class')
ax1.set_ylabel('Number of passengers')
ax1.set_ylim(0, 400)
ax1.set_xticklabels(['1st','2nd', '3rd'])
ax1.set_title('Survivors and victims of Titanic by travel class')
plt.legend()

#barplot : fares of titanic passengers by sex
ax1 = women.plot.hist(y = 'Fare', color='Blue', label='Women', alpha=0.5)
men.plot.hist(y = 'Fare', color='Red', label='Men', alpha=0.5, ax=ax1)
ax1.set_xlabel('Number of passengers')
ax1.set_ylabel('Ticket Fare')
ax1.set_title('Fare of titanic passengers by sex')
ax1.set_ylim(0, 600)

#scatterplot : fares of each titanic passenger
ax2 = sur.plot.scatter(x = 'Fare', y = 'Age', label='Survivors', color = 'blue', alpha = 0.1, s = 50)
vic.plot.scatter(x = 'Fare', y = 'Age', label='Victims', ax = ax2, color = 'red', alpha = 0.1, s = 50)
ax2.set_title('Fare and age of each titanic passenger')
plt.legend()



########EXERCISE 3 - TEXT DATA v2.0########

#read in data from week 1 exercises
pos_data = pd.read_csv(r'data\positive_reviews.txt', sep=",", skiprows = 0)
neg_data = pd.read_csv(r'data\negative_reviews.txt', sep=",", skiprows = 0)

#drop na-values
pos_data = pos_data.dropna()
neg_data = neg_data.dropna()

#get the most used words in each df
top_pos_words = pd.Series(' '.join(pos_data.reviewText).split()).value_counts()
top_neg_words = pd.Series(' '.join(neg_data.reviewText).split()).value_counts()

#get the most used words as a list
top_pos_words2 = (' '.join(pos_data['reviewText'].tolist())).split()
posCounts = Counter(top_pos_words2)
top_neg_words2 = (' '.join(neg_data['reviewText'].tolist())).split()
negCounts = Counter(top_neg_words2)

#get the number of unique negative words and positive words
posUnique = len(Counter(top_pos_words2))
negUnique = len(Counter(top_neg_words2))

#create BAGS OF WORDS from neg and pos words
bagOfWordsA = top_pos_words2
bagOfWordsB = top_neg_words2

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1


#create function to calculate term frequency
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

#create function to calculate inverse document frequency
def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

#create function to calculate TF/IDF
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


#apply above functions to bags of words to get neg and pos TFIDF values
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
idfs = computeIDF([numOfWordsA, numOfWordsB])
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

#create dataframe from above values and change column names
df = pd.DataFrame([tfidfA, tfidfB])
transA = df.transpose()
transA.columns = ['pos', 'neg']


#PLOTS FOR NEG AND POS WORDS

#create a 2x1 plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))

##top20 mentioned words in positive reviews
top_pos_words[:20].plot.bar(ax = ax1,
                            color = 'lightblue',
                            title = ('Most mentioned words in the Amazon ' +
                                     'review dataset (positive reviews)')
)
ax1.set_ylim(0,15000)
ax1.set_ylabel('times mentioned')
ax1.set_xlabel('size of the data: 33921 reviews')

#top20 mentioned words in negative reviews
top_neg_words[:20].plot.bar(ax = ax2,
                            color = 'salmon',
                            title = ('Most mentioned words in the Amazon ' +
                            'review dataset (negative reviews)')
)
ax2.set_ylim(0,1000)
ax2.set_ylabel('times mentioned')
ax2.set_xlabel('size of the data: 7649 reviews')

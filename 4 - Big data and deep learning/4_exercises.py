import numpy as np
import pandas as pd
import ast
from collections import Counter
from pandas.core.algorithms import isin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from PIL import Image
from tpot import TPOTClassifier

########EXERCISE 1########

#read in the data and make sure it's correct size
data = pd.read_csv(
    r'data\ted_main.csv',
    converters = {
        'ratings': eval,
        'tags': eval
        }
    )
assert len(data) == 2550, 'data length should be 2550 rows'

#get rows with funny, confusing inspiring tags from the data
#make respectible columns for the tags in the data
funny = []
confusing = []
inspiring = []

for index, row in data.iterrows():
    for i in row['ratings']:
        if i['id'] == 7:
            funny.append(int(i['count']))
        elif i['id'] == 2:
            confusing.append(int(i['count']))
        elif i['id'] == 10:
            inspiring.append(int(i['count']))

data['funny'] = funny
data['confusing'] = confusing
data['inspiring'] = inspiring

#get the top100 most common tags from the dataset
tags = []
for i in data['tags']:
    tags += i
most_common = Counter(tags).most_common(100)
tagDataframe = pd.DataFrame(most_common, columns=['tag', 'count'])
itags = tagDataframe.set_index(tagDataframe['tag'])

#set binary value for each row whether top100 tags are used
for i in range(len(itags)):
    x = itags.index[i]
    data[x] = 0

for i, row in data.iterrows():
    for value in row['tags']:
        if itags.index.isin([value]).any():
            data.at[i, value] = 1


#construct a function for the linear regression to fit the data in
def linRegression(df, target_column):
    '''Returns predicted outcome from a 20% test batch of a given dataframe.
    The function must be used with the TED talk dataset handled with the same
    procedures as done above in this .py-file

    Args:
        - df : A dataframe to fit the linear regression to
        - target_column: String column name the linear regression is used with

    Returns:
        - predictions: A numpy array of the predicted ratings and number of views
        for the 20% test set'''

    assert isinstance(df, pd.DataFrame), 'df must be a dataframe'
    assert type(target_column) == str, 'target_column must be a string'

    result = pd.DataFrame(
        columns=['prediction', 'truth', 'subtract']
    )

    target = df[target_column]

    drop = df.drop(
        [target_column, 'ratings', 'related_talks',
        'tags','description', 'event',
        'film_date', 'name', 'num_speaker',
        'published_date', 'url', 'main_speaker',
        'published_date', 'speaker_occupation', 'title'],
        axis = 1
    )

    (X_train, X_test,
     y_train, y_test) = train_test_split(
         drop,
         target,
         test_size=0.2,
         random_state=1
    )
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    predictions = linreg.predict(X_test)
    return predictions

#fit the linear regression function to video views
#and funny, confusing, inspiring tags
viewPredict = linRegression(data, 'views')

funnyPredict = linRegression(data, 'funny')
confusingPredict = linRegression(data, 'confusing')
inspiringPredict = linRegression(data, 'inspiring')

#test the above without function
target = data['views']
drop = data.drop(['views', 'ratings', 'related_talks',
                  'tags', 'description', 'event',
                  'film_date', 'name', 'num_speaker',
                  'published_date', 'url', 'main_speaker',
                  'published_date','speaker_occupation','title'],
                 axis = 1)

(X_train, X_test,
 y_train, y_test) = train_test_split(
     drop,
     target,
     test_size=0.2,
     random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
predictions = linreg.predict(X_test)

y_test_n = y_test.to_numpy()
subtract = y_test_n - predictions

#PLOT THE RESULTS

#create a 3x1 subplot
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(15,8))
#plot the linear regression by views
ax1.plot(y_test_n,color='k')
ax2.plot(predictions, color='gray')
ax3.plot(subtract, color='blue')
ax1.set_title('Views by index')
ax2.set_title('Views by prediction')
ax3.set_title('difference in total views and predicted views')


#construct a function to define good and bad tags to get views
#and funny, inspiring, confusing tags
def goodAndBadTags(variables, target, minmax):
    '''Get good and bad tags for videos when aiming to get most views,
    most funny-tags, most-confusing-tags and most inspiring-tags

    Args:
        - variables: A dataframe to fit the linear regression to
        - target: target column (pd.Series)
        - minmax: the spectrum for the plot (list).
            A good spectrum for each target:
                - Views: [-100000, 100000]
                - Funny = [0, 10]
                - Inspiring = [-150, 150]
                - Confusing = [-2, 5]

    Returns:
        - None
            - Prints out the good and the bad tags for given target value'''

    assert isinstance(variables, pd.DataFrame), 'variables must be a dataframe'
    assert isinstance(target, pd.Series), 'target must be a pandas series'
    assert type(minmax) == list, 'minmax must be a list'

    # load data
    X = variables
    y = target

    # Use L1 penalty
    estim = LassoCV(cv=5, normalize = True)
    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(
        estim,
        threshold = 0.25,
        prefit = False,
        norm_order = 1,
        max_features = None
        )
    sfm.fit(X, y)

    feature_id = sfm.get_support()
    feature_name = X.columns[feature_id]

    colname = y.name
    coeffs = sfm.estimator_.coef_

    badTags = []
    goodtags = []

    for index in range(X.shape[1]):
        if (coeffs[index] <minmax[0]):
            badTags.append(X.iloc[: , index].name)
        if(coeffs[index]>minmax[1]):
            goodtags.append(X.iloc[: , index].name)


    print("Bad tags for " + colname)
    print(badTags)
    print("Good tags for " + colname)
    print(goodtags)


#apply the function of good and bad tags to the views
#funny, inspiring and confusing columns

#VIEWS
target = data['views']
drop = data.drop(['views','ratings','related_talks','tags','description', 'event', 'film_date','name','num_speaker','published_date','url','main_speaker','published_date','speaker_occupation','title'], axis = 1)
viewsmm = [-100000, 100000]
goodAndBadTags(drop, target, viewsmm)


print('***')

#FUNNY
target = data['funny']
drop = data.drop(['funny','ratings','related_talks','tags','description', 'event', 'film_date','name','num_speaker','published_date','url','main_speaker','published_date','speaker_occupation','title'], axis = 1)
funnymm = [0,10]
goodAndBadTags(drop, target, funnymm)


print('***')

#INSPIRING
target = data['inspiring']
drop = data.drop(['inspiring','ratings','related_talks','tags','description', 'event', 'film_date','name','num_speaker','published_date','url','main_speaker','published_date','speaker_occupation','title'], axis = 1)
inspiringmm = [-150,150]
goodAndBadTags(drop, target, inspiringmm)


print('***')

#CONFUSING
target = data['confusing']
drop = data.drop(['confusing','ratings','related_talks','tags','description', 'event', 'film_date','name','num_speaker','published_date','url','main_speaker','published_date','speaker_occupation','title'], axis = 1)
confusingmm = [-2,5]
goodAndBadTags(drop, target, confusingmm)



########EXERCISE 2#########
#######HASYv2 PART 2#######

#read in and filter the data
data = pd.read_csv(r'data/hasy-data-labels.csv')

#filter the data further to two lists:
#one of the image data and one of the label data
imagedata = []
labeldata = []

for i, row in data.iterrows():

    fp = data['path'][i]

    #when opening, onvert the images to B&W and resize them to 32x32
    img = Image.open(fp).convert('L').resize(size=(32,32))

    r = np.array(img)
    r = r.flatten()
    r = r.tolist()

    labeldata.append([row['path'], row['symbol_id']])
    imagedata.append(r)

labeldf = pd.DataFrame(data=labeldata, columns = {'path', 'symbol_id'})

X_train, X_test, y_train, y_test = train_test_split(
    imagedata,
    labeldf,
    test_size = 0.2,
    random_state = 0
    )

#apply random forext classifier to the HASY data
forest = RandomForestClassifier()
forest.fit(X_train, y_train['symbol_id'])
predictions = forest.predict(X_test)

#print predictions
print('Predictions: ' + str(predictions))
print('True values: ' + str(y_test['symbol_id'].values))
print('the precision of random forest classification ' +
      'without tuned parameters: ' + str(
          forest.score(X_test, y_test['symbol_id'])))

#construct a function to raise the treebranch amount by 10
#and that loops through from 10 to 200 branches
treedf = pd.DataFrame()
def treeMaker10to200(dataframe, variabledf, test_size, random_state):
    '''Applies random forest classifier to a given dataframe and raises the amount of trees
    by 10 each run through between 10-200

    Args:
        dataframe: dataframe the classifier is applied to
        variabledf: dataframe of the variables in the dataframe
        test_size: size of train/test split
        random_state: state of the randomiser

    Returns:
        treedf: A dataframe with tree number and precision value from each loop'''

    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe'
    assert isinstance(variabledf, pd.DataFrame), 'variabledf must be a pandas dataframe'
    assert type(test_size) == float, 'test_size must be a floating point number'
    assert type(random_state) == int, 'random_state must be an integer number'

    (X_train, X_test,
    y_train, y_test) = train_test_split(
        dataframe,
        variabledf,
        test_size = test_size,
        random_state = random_state
    )

    treedf['trees'] = None
    treedf['precision'] = None
    for i in range(1, 21):
        forest = RandomForestClassifier(n_estimators=i*10)
        forest.fit(X_train, y_train['symbol_id'])
        predictions = forest.predict(X_test)
        precision = forest.score(X_test, y_test['symbol_id'])
        treedf.at[i, 'trees'] = i*10
        treedf.at[i, 'precision'] = precision
    return treedf

#apply the random forest function to the hasy data
treedf = treeMaker10to200(imagedata, labeldf, 0.2, 0)

#plot the treedf precision column
treedf.plot(x='trees', y='precision')

#ANSWER TO WHY THIS ISNT THE WAY TO GO ADDED TO INSTRUCTIONS.MD


#construct a function to print out validation results
#of each tree branch number between 10-200
#(raised by 10 each loop)

def validationTest(dataframe, variabledf, random_state):
    '''Applies random forest classifier to a given dataframe and raises
    the amount of trees by 10 each run through between 10-200.
    Prints out the validation results of each loop

    Args:
        dataframe: dataframe the classifier is applied to
        variabledf: dataframe of the variables in the dataframe
        random_state: state of the randomiser

    Returns:
        None: Prints out the validation score of each loop'''

    (X_train, X_test1,
    y_train, y_test1) = train_test_split(
        dataframe,
        variabledf,
        test_size = 0.2,
        random_state = 0)
    (X_test, X_valid,
    y_test, y_valid) = train_test_split(
        X_test1,
        y_test1,
        test_size = 0.5,
        random_state = 0)

    for i in range(1, 21):
        forest = RandomForestClassifier(n_estimators=i*10)
        forest.fit(X_train, y_train['symbol_id'])
        predictions = forest.predict(X_valid)
        precision = forest.score(X_valid, y_valid['symbol_id'])

        print('the precision of random forest classification validation ' +
              'data with ' + str(i*10) + ' branches: ' + str(precision))

#test the function to the hasy data
validationTest(imagedata, labeldf, 0)

#seems that 110 and 200 branches could be the way to go
#but the values differ, again, by the shuffle


#construct a function to test the random forest classificator to the dataset
#and create a plot #for the symbols the classificator guesses wrongly
def testDataTest(dataframe, variabledf, random_state, no_branches):
    '''Applies random forest classifier to a dataframe of HASY data and
    prints out the wrongly guessed symbols

    Args:
        dataframe: dataframe the classifier is applied to
        variabledf: dataframe of the variables in the dataframe
        random_state: state of the randomiser
        no_branches: number of branches/trees in the classifier

    Returns:
        None: Prints out the wrongly guessed symbols'''

    (X_train, X_test1,
    y_train, y_test1) = train_test_split(
        dataframe,
        variabledf,
        test_size=0.2,
        random_state=0
        )

   (X_test, X_valid,
    y_test, y_valid) = train_test_split(
        X_test1,
        y_test1,
        test_size = 0.5,
        random_state = 0
        )

    forest = RandomForestClassifier(n_estimators=no_branches)

    forest.fit(X_train, y_train['symbol_id'])
    predictions = forest.predict(X_test)
    precision = forest.score(X_test, y_test['symbol_id'])

    wrong = y_test.loc[((y_test['symbol_id']-predictions) != 0)]

    #plot the wrong symbols
    i = 0
    for idx, row in wrong.iterrows():
        img = Image.open(row['path'])
        plt.subplot(5,5,i+1)
        plt.imshow(img)
        i+=1

#Number of wrong symbols has shrunk but the numbers are still quite easily distinguishable!
#Using the validation data before test data did help with altering the parameters (branch numbers)

#TESTING TPOT

#make np array from imagedata dataframe
imagedata_n = np.array(imagedata)

#train test split (90/10)
(X_train, X_test1,
y_train, y_test1) = train_test_split(imagedata_n, labeldf, train_size=0.90)

#apply TPOT classifier to the data
tpot = TPOTClassifier(generations=3, population_size=10, verbosity=2, random_state=42)
tpot.fit(X_train, y_train['symbol_id'])

#get digits from external file
tpot.export('tpot_digits_pipeline.py')
# Average CV score on the training set was: 0.6247619047619047
exported_pipeline = BernoulliNB(alpha=0.1, fit_prior=True)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(X_train, y_train['symbol_id'])
results = exported_pipeline.predict(X_test1)

results = exported_pipeline.predict(X_test1)
precision = exported_pipeline.score(X_test1, y_test1['symbol_id'])

wrong_tpot = y_test1.loc[((y_test1['symbol_id']-results) != 0)]

#plot the wrong symbols
i = 0
for idx, row in wrong_tpot.iterrows():
    img = Image.open(row['path'])
    plt.subplot(5,5,i+1)
    plt.imshow(img)
    i+=1

#The model managed to half the wrong values compared to the random forest classifier!
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


########EXERCISE 1 - GIS########

#get data from files and check CRS
map_data = gpd.read_file(r'data\world_m\world_m.shp')
city_data = gpd.read_file(r'data\cities\cities.shp')
if map_data.crs != city_data.crs:
    print('CITY_DATA CRS: 'city_data.crs)
    print('MAP_DATA CRS: 'map_data.crs)

#reproject the city data to match with map_data (world mercator)
city_data_proj = city_data.to_crs(epsg=3395)
map_data_proj = map_data.to_crs(epsg=3395)

#reproject both datas to robinson projection
city_data_proj2 = city_data.to_crs('+proj=robin')
map_data_proj2 = map_data.to_crs('+proj=robin')



#########ASSERTION#########
assert city_data_proj == map_data_proj, 'bot projections should be world mercator'
assert city_data_proj2 == map_data_proj2, 'both projections should be robinson'
###########################


#PLOT THE GIS DATA

#create 2x1 plot
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,10))

#world mercator projection
map_data_proj.plot(ax=ax1)
city_data_proj.plot(ax=ax1, color='orange', markersize=6)
ax1.set_title('WGS 84 | World Mercator | EPSG:3395')

#robinson projection
map_data_proj2.plot(ax=ax2)
city_data_proj2.plot(ax=ax2, color='orange', markersize=6)
ax2.set_title('World Robinson projection | ESRI:54030')



########EXERCISE 2 - SYMBOL CLASSIFICATION########

#read in and filter the data
data = pd.read_csv(r'data/hasy-data-labels.csv')
data = data.loc[(data['symbol_id'] >= 70) & (data['symbol_id'] <= 80)]

#filter the data further to two lists: One of the image data and one of the label data
imagedata = []
labeldata = []

for i, row in data.iterrows():

    #when opening, onvert the images to B&W and resize them to 32x32
    img = Image.open(data['path'][i]).convert('L').resize(size = (32,32))

    r = np.array(img)
    r = r.flatten()
    r = r.tolist()

    labeldata.append([row['symbol_id'], row['path']])
    imagedata.append(r)

labeldf = pd.DataFrame(
    data = labeldata,
    columns = {'symbol_id',
               'path'
            }
)

(image_train, image_test,
 label_train, label_test) = train_test_split(
     imagedata,
     labeldf,
     test_size = 0.2,
     random_state = 0)



#########ASSERTION#########
assert np.shape(imagedata) == (1020, 1024)
assert np.shape(labeldata) == (1020, 2)
assert len(image_train) + len(image_test) == 1020
assert len(label_train) + len(label_test) == 1020
###########################



#apply logistic regression to train data
logreg = LogisticRegression()
logreg.fit(image_train, label_train['symbol_id'])

#get prediction score
predictions = logreg.predict(image_test)
logregscoreTrain = logreg.score(image_test, label_test['symbol_id'])

#apply the logistic regression to test data and get score
logreg.predict(image_test[:20])
logregscoreTest = logreg.score(image_test,label_test['symbol_id']))

#define a very stupid 'dummyclassifier' to classify images
#by the most frequent values in set
dummy_clf = DummyClassifier(strategy = "most_frequent")
dummy_clf.fit(image_train, label_train['symbol_id'])
dummy_clf.predict(image_test)
dummyscore = dummy_clf.score(image_test, label_test['symbol_id'])

#check scores
print('Prediction score with dummy classifier (most frequent value): ' +
      str(dummyscore))
print('Prediction score with logistic regression classifier: ' +
      str(logregscoreTest))

#create a list with all the wrong guessed symbols from the test dataset
wrong = label_test.loc[((label_test['symbol_id'] - predictions) != 0)]

#plot the wrong symbols
i = 0
for idx, row in wrong.iterrows():
    img = Image.open(row['path'])
    plt.subplot(5,5,i+1)
    plt.imshow(img)
    i+=1
# Week 3 - Statistical methods & Machine learning

## Exercise 1 - GIS

To get at least a bit familiar with GIS data and the concept of map projections, we’ll do a task of plotting two data sets that are given in different coordinate systems.

Download the `world_m` and `cities` files that each include a set of GIS files. Most notably, the `shp` -files are files with coordinates. The `prj` -files contain information about the coordinate systems. Open the files using your choice of programming environment and packages.

1. Plot the `world_m` -file contains borders of almost all countries in the world.

2. Plot another layer of information on top of the world_m layer, namely the capital cities of each country from the `cities` -dataset. However, the cities will probably all appear to be in the Gulf of Guinea, near the coordinates `(0°, 0°)`.

3. Perform a map projection to bring the two datasets into the same coordinate system. Now plot the two layers together to make sure the capital cities are where they are supposed to be.


## Exercise 2 - Symbol classification

We’ll be looking into machine learning by checking out the `HASYv2` -dataset that contains hand written mathematical symbols as images. The whole dataset is quite big, so we’ll restrict ourselves to doing 10-class classification on some of the symbols.

1. Extract the data and find inside a file called `hasy-data-labels.csv`. This file contains the labels for each of the images in the hasy_data folder. Read the labels in and only keep the rows where the symbol_id is within the inclusive range [70, 80]. Read the corresponding images as black-and-white images and flatten them so that each image is a single vector of shape `32x32 == 1024`. Your dataset should now consist of your input data of shape `(1020, 1024)` and your labels of shape `(1020, )`. That is, a matrix of shape `1020 x 1024` and a vector of size `1020`.

2. Shuffle the data, and then split it into training and test sets, using the first 80% of the data for training and the rest for evaluation.

3. Fit a logistic regression classifier on your data. Note that since logistic regression is a binary classifier, you will have to, for example, use a so-called “one-vs-all” strategy where the prediction task is formulated as “is the input class X or one of the others?” and the classifier selects the class with the highest probability.

4. Plot some of the images that were classified wrongly. Can you think of why this happens? Would you have gotten it right?

```bash
Above I have plotted all the symbols the regression model has guessed wrong. On most parts the figures are most likely too irregular, misshaped or wrongly positioned in the bounding box compared to the 'norm' the regression creates. Most of the figures reach also out of the bounding box, which may cut the figure to that extent that it's non-recognizable.

I'd say all of the figures above are easily distinguished by any human viewing them.
```
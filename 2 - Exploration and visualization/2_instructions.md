# Week 2 - Exploration & visualization

## Exercise 1 - Titanic

Download the Titanic dataset from Kaggle, and complete the following exercises. The dataset consists of personal information of all the people on board the Titanic, along with the information whether they survived the disaster or not.

1. Read the data in your favorite language.

Have a look at the data. We will build new representations of the dataset that are better suited for a particular purpose. Some of the columns simply identify a person and cannot be useful for prediction tasks - remove them.

The column `Cabin` contains a letter and a number. We can conclude that the letter stands for a deck on the ship and having just the deck information might improve the results of a classifier predicting an output. Add a new column to the dataset, which is simply the deck letter.

You’ll notice that some of the columns, such as the previously added deck number are categorical. Their representation as a string is not efficient for further computation. Transform them into numeric values so that a unique integer id corresponds to each distinct category.

Some of the rows in the data have missing values, e.g when the cabin number of a person is not known. Most machine learning algorithms have trouble with missing values, and they need to be handled in preprocessing:

- For continous values, replace the missing values with the `mean` of the non-missing values of that column.

- For discrete and categorical values, replace the missing values with the `mode` of the column.


At this point, all data are numeric. Write the data, with the modifications, to a .csv file. Then, write another file, this time in the json format, with the following structure:

```bash
[
    {
        "Deck": 0,
        "Age": 20,
        "Survived", 0
        ...
    },
    {
        ...
    }
]
```

You can study the records to see if there is an evident pattern in the chances of survival. Later, we will be doing more systematic analysis, so your work here will be rewarded.


## Exercise 2 - Titanic v2.0

In this exercise, we’ll continue to study the Titanic dataset from the last exercise. Now that we have preprocessed it a bit, it’s time to do some exploratory data analysis.

First consider each feature variable in turn. For categorical variables, find out the most frequent value, i.e., the mode. For numerical variables, calculate the median value.

Combining the modes of the categorical variables, and the medians of the numerical variables, construct an imaginary “average survivor” on board of the ship. Also following the same procedure for using subsets of the passengers, construct the the “average non-survivor”.

Now study the distributions of the variables in the two groups. How well do the average cases represent the respective groups? Can you find actual passengers that are very similar to the representative of their own group (survivor/non- survivor)? Can you find passengers that are very similar to the representative of the other group?

To give a more complete picture of the two groups, provide graphical displays of the distribution of the variables in each group whenever appropriate.

One step further is the analysis of pairwise and multivariate relationships between the variables in the two groups. Try to visualize two variables at a time using, e.g., scatter plots and use a different color to display the survival status.

Finally, recall the preprocessing that was carried out in last exercises. Can you say something about the effect of the choices that were made, in particular, to use the `mode` or the `mean` to impute missing values, instead of, for example, ignoring passengers with missing data?

### ANSWER:

```bash
Replacing some of the missing values with a single value profed to be a bad choice. For example, doing graphics with passenger age continues to remind that many of the ages, before stated NaN, have been replaced with the same value, average of the age of the passengers, whose age is known. In the case of age, there could have been made a list with the length of Na-values, filled with normal distribution of the data. There is also wider datasets available on the subjects, where at least some of the age data could have been checked.

In many cases, where even slightly larger datasets are being processed, replacing Na-values with a single value is never a good idea. Depending on the matter and the quantity of the dataset, sometimes it's wiser to drop rows with not enough data (Na values)
```


## Exercise 3 - Text data v2.0

From week 1 exercise, find `pos.txt` and `neg.txt`.

Find the most common words in each file. What are they? Are some of them clearly general terms relating to the nature of the data, and not just the emotion?

Compute a `TF/IDF` vector for each of the text files, and make them into a `2 x m` matrix, where `m` is the number of unique words in the data. The problem with using the most common words in a review to analyze its contents is that words that are common overall will be common in all reviews. This means that they probably don’t tell anything about a specific review. `TF/IDF` stands for Term Frequency / Inverse Document Frequency, and is designed to help by taking into consideration not just the term frequency, but also inverse document frequency.

List the words with the highest `TF/IDF` score in each class, and compare them to the most common words. What do you notice? Did `TF/IDF` work as expected?

Plot the words in each class and their corresponding `TF/IDF` scores. If you can’t plot them all, plot a subset.

### Answers/ Notes:
```bash
- Most of the words used in both review types are of normal nature of the data (car part/supply products).

- Most of the words in both lists are the same.
```


## Exercise 4 - Junk Charts

There’s a thriving community of chart enthusiasts who keep looking for statistical graphics that they find inappropriate, and which they call “junk charts”, and who often also propose fixes to improve them.

Find at least three statistical visualizations that you think aren’t very good, and identify their problems. Copying examples from various junk chart websites isn’t accepted – you should find your own junk charts. You should be able to find good (or rather, bad) examples quite easily since a large fraction of charts have at least some issues. The examples you choose should also have different problems, so don’t look for three column or bar charts whose axes don’t begin at zero. Try to find as interesting and diverse examples as you can.

Try to produce improved versions of the charts you selected. The data is of course often not available, but perhaps you can try to extract it, at least approximately, from the chart. Or perhaps you can simulate data that looks similar enough to make the point.

Submit a `PDF` with the charts you found and the ones you produced

### Answer:

```bash
I stumbled across a graph of rocket launches over the years by different space companies/agencies. The graph shows the data pretty efficiently, by the lines of each agency seem to mix up with themselves a little bit. I made the graph larger and added colors (color blind proof!) to make the graph easier to read.

All in all, line graph is a good way to introduce year-by-year data. Underneath is the altered version, to which I added larger titles elsewhere. Both pictures can be found in the pdf file I'm submitting.

(I could not find the data myself, so I hand picked the values from the old graph as well as I could)
(Sidenote: It's painstakingly difficult to make rounded lines in matplotlib line chart!)
```
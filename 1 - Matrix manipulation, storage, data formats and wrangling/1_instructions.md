# Week 1 - Matrix manipulation, storage, data formats and wrangling

## Basics
Create two arrays, `a` and `b` which each have the integers `0, 1, 2 ..., 1e7 - 1`. Use the normal arrays or lists of your programming language, e.g `list` or `[]` in Python.
Create a function that uses a for-loop or equivalent to return a new array, which contains the element-wise sum of `a` and `b`. Something like

Now create another function that uses `Numpy` (or equivalent) to do the same. To try it out, allocate two arrays and add the arrays together using your function. Don’t use loops, instead, find out how to add the two arrays together directly. What do you notice?

## Array manipulation
Note: for these exercises, only use `Numpy` or equivalent functions.

(a) Create the following array

```bash
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
```

(b) Create the following array

```bash
array([[0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]])
```

(c) Create the following array

```bash
array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]])
```

(d) Create the following array

```bash
array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
       [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
       [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
       [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
       [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
       [1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
       [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
```

(e) Call the last two matrices `C` and `D`, respectively. Show that the determinant of their product (matrix multiplication) is the same as the product of their determinants. That is, calculate both `det(CD)` and `det(C) * det(D)`, and show that they are the same.
Why does this hold in this instance? Does it hold in general?

### Answer:

```bash
In matrix calculations, determinant of two matrices product and product of their determinant is the same. This holds in all cases, when defining the determinant of two matrices. Therefore, when A and B are two matrix, det(AB) equals det(A)*det(B).
```

## Slicing

Practice array slicing with the following exercises.

### Exercice 1 - Boston housing dataset

(a) Load the Boston housing dataset. The data should be a matrix of shape `(506, 13)`. Use the shape attribute of numpy arrays to verify this.

(b) Select rows where the Crime feature `CRIM` is higher than 1. There should be 174 rows

(c) Select the rows where the pupil-to-teach ratio is between 16% and 18% (exclusive). There should be 100 rows.

(d) Find the mean nitric oxides concentration for homes whose median price is more than $25000 (the target variable). It should be around 0.492.


### Exercise 2 - Text data

Next we’ll look at some text data. We’ll be looking into Amazon reviews, and the steps needed to transform a raw dataset into one more suitable for prediction tasks.

Download the automotive 5-core dataset from here. Extract it to find the data in json format. You can also download one of the bigger ones, if you are feeling ambitious.

The `reviewText` -field contains the unstructured review text written by the user. When dealing with natural language, it is important to notice that while, for example, the words “Copper” and “copper.” are represented by two different strings, they have the same meaning. When applying statistical methods on this data, it is useful to ensure that words with the same meaning are represented by the same string.

To do this, we usually normalize the data, by for example removing punctuation and capitalization differences. A related issue is that, for example, while again the words “swims” and “swim” are distinct string, they both refer to swimming. **Stemming** refers to the process of mapping words in inflected form to their base form: swims -> swim, etc.

Finally, another popular approach is to remove so called **stop-words**, words that are very common and have little to do with the actual content matter. There’s plenty of openly available lists of stop-words for almost any language.

Do the following:

a) Open the `json` -file in your favorite environment

b) Access the `reviewText` field, downcase the contents

c) Remove all punctuation, as well as the stop-words. Find a list for English stop-words to apply here.

d) Apply a stemmer on the paragraphs, so that inflected forms are mapped to the base form. For example, for python the popular natural language toolkit `nltk` has an easy-to-use stemmer.

e) Filter the data by selecting reviews where the field `overall` is 4 or 5, and store the review texts in file `pos.txt`. Similarly, select reviews with rating 1 or 2 and store the reviews in file `neg.txt`. Each line in the two files should contain exactly one preprocessed review text without the rating.

Having created two collections of positive and negative reviews, respectively, you may wish to take a quick look to see how the review texts differ between them. We will be using this data later with machine learning methods.
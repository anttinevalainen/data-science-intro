# Week 4 - Big data & Deep learning

## Exercise 1 - Linear regression with feature selection

Download the TED Talks dataset from Kaggle. Your task is to predict the ratings and the number of views of a given TED talk.

Use the following ratings from column ratings: `Funny`, `Confusing`, `Inspiring`. Store these values into respective columns so that they are easier to access. Next, extract the tags from column `tags`. Count the number of occurrences of each tag and select the top-100 most common tags. Create a binary variable for each of these and include them in your data table, so that you can directly see whether a given tag is used in a given TED talk or not.

Construct a linear regression model to predict the number of views based on the data in the table, including the binary variables for the top-100 tags that you just created.

Do the same for the `Funny`, `Confusing`, and `Inspiring` ratings.

You will probably notice that most of the tags are not useful in predicting the views and the ratings. You should use some kind of variable selection to prune the set of tags that are included in the model. You can use for example `p-values` or more modern `Lasso` techniques. Which tags are the best predictors of each of the response variables?

Produce summaries of your results. Could you recommend good tags – or tags to avoid! – for speakers targeting plenty of views and/or certain ratings?

## Exercise 2 - HASYv2 part 2

Train a `random forest` -classifier on the previous week's data. Without tuning any parameters, how is the accuracy?

The amount of trees to use as a part of the random forest is an example of a hyperparameter, because it is a parameter that is set prior to the learning process. Train 20 classifiers, with varying amounts of decision trees starting from 10 up until 200, and plot the test accuracy as a function of the amount of classifiers. Does the accuracy keep increasing?

If we had picked the amount of decision trees by taking the value with the best test accuracy from the last plot, we would have overfit our hyperparameters to the test data. Can you see why it is a mistake to tune hyperparameters of your model by using the test data?

### Answer:

```bash
Tuning the hyperparameters does not affect the data values in any way, so it's completely random how the data shuffles and how the points of good/bad precision scatter around the plot. It would be a mistake to point out the points of good precision because they change constantly.
```

Reshuffle and resplit the data so that it is divided in 3 parts: `training (80%)`, `validation (10%)` and `test (10%)`. Repeatedly train a model of your choosing on the training data, and evaluate its performance on the validation set, while tuning the hyperparameters so that the accuracy on the validation set increases. Then, finally evaluate the performance of your model on the test data. What can you say in terms of the generalization of your model?

This process of picking a suitable model, evaluating its performance and tuning the hyperparameters is very time consuming. A new idea in machine learning is the concept of automating this by using an optimization algorithm to find the best model in the space of models and their hyperparameters. Have a look at TPOT, an automated ML solution that finds a good model and a good set of hyperparameters automatically. Try it on this data, it should outperform simple models like the ones we tried easily.
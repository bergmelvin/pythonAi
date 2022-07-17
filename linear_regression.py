from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import clear_output



import tensorflow as tf


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

# Pop out out label.
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Create feature_column array with numeric data (ex. male = 1, female = 2).
categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []
for feature_name in categorical_columns:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        key = feature_name,
        vocabulary_list = vocabulary
        ))
for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(
        key = feature_name,
        dtype = tf.float32
    ))

# Create a input function that converts our pandas dataframe into a t.data.Dataset object.
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function, which return a dataset object

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Create a linear estimator to utilize the linear regression algorithm
linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)

# Training the model
linear_est.train(input_fn = train_input_fn) # Train

# Evaluate our model by comparing testing data with training data
result = linear_est.evaluate(input_fn = eval_input_fn)
clear_output()
print(result['accuracy'])

# Make predictions
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0]) # Stats for person 1
print(result[0]['probabilities'][1]) # Chance of survival for person 1
# ----------------------- INFORMATION -------------------------------
    # Where regression was used to predict a numeric value, classification is used to seperate data points into classes of different labels.
    # Predciting classes: the probability that a specific data point is within all of the different classes it could be.



# ----------------------- CODE -------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
from tokenize import Number
import tensorflow as tf
import pandas as pd



csv_column_names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species']
species = ['setosa', 'versicolor', 'virginica'] # species : 0 = setosa, 1 = versicolor, 2 = virginica

# Load dataset.
train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

train = pd.read_csv(train_path, names=csv_column_names, header=0)
test = pd.read_csv(test_path, names=csv_column_names, header=0)

# Pop out out label.
y_train = train.pop('species')
y_test = test.pop('species')

# Create a input function
def input_fn(features, labels, training=True, batch_size= 256):
    # Converts the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
feature_columns = []
for feature_name in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=feature_name))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each (build our model).
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10],
    n_classes=3)

# Training the model
classifier.train(
    input_fn=lambda: input_fn(train, y_train, training=True),
    steps=5000)

# Evaluate our model by comparing testing data with training data
eval_result= classifier.evaluate(input_fn=lambda:input_fn(test, y_test, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Prediction
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        species[class_id], 100 * probability))
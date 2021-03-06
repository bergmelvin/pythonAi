''' To create a Hidden Markov Model (HMM) we need:
        * States
        * Observation Distribution
        * Transition Distribution
'''

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

'''Wheather Model
    1. Cold days are encoded by a 0 and hot days are encoded by a 1.
    2. The first day in our sequence has an 80% chance of being cold.
    3. A cold day has a 30% chance of being followed by a hot day.
    4. A hot day has a 20% chance of being followed by a cold day.
    5. On each day the temperature is normally distributed with mean and 
       standard deviation 0 and 5 on a cold day and mean and 
       standard deviation 15 and 10 on a hot day.
'''

initial_distribution = tfp.distributions.Categorical(probs=[0.8, 0.2])
transition_distribution = tfp.distributions.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfp.distributions.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# Create Model
model = tfp.distributions.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


class CriticNetwork(keras.Model):
    """
    The critic network is used to evaluate the value of the state-action pair - Q(s,a).
    """

    def __init__(
        self, fc1_dims=512, fc2_dims=512, name="critic", chkpt_dir="results/sac"
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_sac.h5"
        )

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ValueNetwork(keras.Model):
    """
    The value network is used to evaluate the value of the state - V(s).
    """

    def __init__(
        self, fc1_dims=512, fc2_dims=512, name="value", chkpt_dir="results/sac"
    ):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_sac.h5"
        )

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.v = Dense(1, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v


class ActorNetwork(keras.Model):
    """
    The actor network is used to determine the best action for a given state.
    """

    def __init__(
        self,
        max_action,
        n_actions=2,
        fc1_dims=512,
        fc2_dims=512,
        name="actor",
        chkpt_dir="results/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_td3.h5"
        )

        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        # mu is the mean of the probability distribution
        # if action bounds not +/- 1, can multiply by action space bounds
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample() + tf.random.normal(
                tf.shape(mu), stddev=self.noise, dtype=tf.float32
            )
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions) * self.max_action
        # log probability of the action using the actor network
        log_probs = probabilities.log_prob(actions)
        # calcualte the log prob of the policy
        # add noise to prevent log(0)
        log_probs = log_probs - tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

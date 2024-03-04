import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from nets import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        input_dims,
        alpha=0.001,
        beta=0.002,
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        tau=0.005,
        layer1_size=400,
        layer2_size=300,
        batch_size=64,
        noise=0.1,
    ):
        """
        Initialize the agent with the following parameters:
            input_dims: the dimensions of the input
            alpha: the learning rate for the actor network
            beta: the learning rate for the critic network
            env: the environment
            gamma: the discount factor
            n_actions: the number of actions
            max_size: the maximum size of the replay buffer
            tau: the soft update parameter
            layer1_size: the size of the first layer of the networks
            layer2_size: the size of the second layer of the networks
            batch_size: the size of the batch
            noise: the noise to add to the action
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="actor",
        )
        self.critic = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic"
        )
        self.target_actor = ActorNetwork(
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="target_actor",
        )
        self.target_critic = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic"
        )

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))

        # we do not use gradient decend on the target networks
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # update the weights of the target actor
        # by applying the formula:
        # target_weights = weights * tau + target_weights * (1 - tau)
        # soft update, similar to moving average
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            # add noise for exploration during training
            actions += tf.random.normal(
                shape=[self.n_actions], mean=0.0, stddev=self.noise
            )
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)

        # update the critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)

            # next_action_value
            critic_value_ = tf.squeeze(
                self.target_critic(new_states, target_actions), 1
            )

            # action_value
            critic_value = tf.squeeze(self.critic(states, actions), 1)

            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        self.critic.optimizer.apply_gradients(
            zip(critic_network_gradient, self.critic.trainable_variables)
        )

        # update the actor network
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)

            # critic outputs the value of the state-action pair
            # similar to the reward but with specific state-action pairs
            # we want to maximize the value of the state-action pair -> gradient ascent
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        self.actor.optimizer.apply_gradients(
            zip(actor_network_gradient, self.actor.trainable_variables)
        )

        self.update_network_parameters()

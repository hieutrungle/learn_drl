import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from nets import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(
        self,
        input_dims,
        alpha=0.0003,
        beta=0.0003,
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        tau=0.005,
        update_actor_interval=2,
        warmup=1000,
        layer1_size=400,
        layer2_size=300,
        batch_size=256,
        reward_scale=2,
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
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup
        self.reward_scale = reward_scale
        self.learn_step_cntr = 0
        self.time_step = 0
        self.max_action = env.action_space.high[0]
        self.max_action = tf.convert_to_tensor(self.max_action, dtype=tf.float32)
        self.min_action = env.action_space.low[0]
        self.min_action = tf.convert_to_tensor(self.min_action, dtype=tf.float32)

        self.actor = ActorNetwork(
            max_action=self.max_action,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="actor",
        )
        self.critic_1 = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic_2"
        )
        self.value = ValueNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="value"
        )
        self.target_value = ValueNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_value"
        )

        # self.target_actor = ActorNetwork(
        #     n_actions=n_actions,
        #     fc1_dims=layer1_size,
        #     fc2_dims=layer2_size,
        #     name="target_actor",
        # )
        # self.target_critic_1 = CriticNetwork(
        #     fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic_1"
        # )
        # self.target_critic_2 = CriticNetwork(
        #     fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic_2"
        # )

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        # # we do not use gradient decend on the target networks
        # self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        # self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        # self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # update the weights of the target actor
        # by applying the formula:
        # target_weights = weights * tau + target_weights * (1 - tau)
        # soft update, similar to moving average
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(tf.convert_to_tensor(weight * tau + targets[i] * (1 - tau)))
            # weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_value.set_weights(weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions, _ = self.actor.sample_normal(state, reparameterize=True)
        return actions.numpy()[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(
                states, reparameterize=True
            )
            log_probs = tf.squeeze(log_probs, 1)

            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(
            value_loss, self.value.trainable_variables
        )
        self.value.optimizer.apply_gradients(
            zip(value_network_gradient, self.value.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(
                states, reparameterize=True
            )
            log_probs = tf.squeeze(log_probs, 1)

            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables
        )
        self.actor.optimizer.apply_gradients(
            zip(actor_network_gradient, self.actor.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            next_value = tf.squeeze(self.target_value(new_states), 1)
            q_hat = self.reward_scale * rewards + self.gamma * next_value * (1 - done)
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2, q_hat)

        critic_network_1_gradient = tape.gradient(
            critic_1_loss, self.critic_1.trainable_variables
        )
        self.critic_1.optimizer.apply_gradients(
            zip(critic_network_1_gradient, self.critic_1.trainable_variables)
        )

        critic_network_2_gradient = tape.gradient(
            critic_2_loss, self.critic_2.trainable_variables
        )
        self.critic_2.optimizer.apply_gradients(
            zip(critic_network_2_gradient, self.critic_2.trainable_variables)
        )

        self.update_network_parameters()

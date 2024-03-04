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
        update_actor_interval=2,
        warmup=1000,
        layer1_size=400,
        layer2_size=300,
        batch_size=300,
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
        self.learn_step_cntr = 0
        self.time_step = 0
        self.max_action = env.action_space.high[0]
        self.max_action = tf.convert_to_tensor(self.max_action, dtype=tf.float32)
        self.min_action = env.action_space.low[0]
        self.min_action = tf.convert_to_tensor(self.min_action, dtype=tf.float32)

        self.actor = ActorNetwork(
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

        self.target_actor = ActorNetwork(
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="target_actor",
        )
        self.target_critic_1 = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic_1"
        )
        self.target_critic_2 = CriticNetwork(
            fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic_2"
        )

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss="mean")
        self.critic_1.compile(
            optimizer=Adam(learning_rate=beta), loss="mean_squared_error"
        )
        self.critic_2.compile(
            optimizer=Adam(learning_rate=beta), loss="mean_squared_error"
        )

        # we do not use gradient decend on the target networks
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

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
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        if self.time_step < self.warmup:
            action = tf.random.normal(
                shape=[self.n_actions], mean=0.0, stddev=self.noise
            )
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            action = self.actor(state)[0]
        if not evaluate:
            # add noise for exploration during training
            action += tf.random.normal(
                shape=[self.n_actions], mean=0.0, stddev=self.noise
            )
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        self.time_step += 1
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # update the critic network
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(new_states)
            target_actions = target_actions + tf.clip_by_value(
                tf.random.normal(shape=target_actions.shape, mean=0.0, stddev=0.2),
                -0.5,
                0.5,
            )
            target_actions = tf.clip_by_value(
                target_actions, self.min_action, self.max_action
            )

            # next_action_value
            # [batch_size, 1] --> [batch_size]
            next_q1 = tf.squeeze(self.target_critic_1(new_states, target_actions), 1)
            next_q2 = tf.squeeze(self.target_critic_2(new_states, target_actions), 1)

            # action_value
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            next_critic_value = tf.math.minimum(next_q1, next_q2)

            target = rewards + self.gamma * next_critic_value * (1 - done)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape.gradient(
            critic_1_loss, self.critic_1.trainable_variables
        )
        self.critic_1.optimizer.apply_gradients(
            zip(critic_1_gradient, self.critic_1.trainable_variables)
        )

        critic_2_gradient = tape.gradient(
            critic_2_loss, self.critic_2.trainable_variables
        )
        self.critic_2.optimizer.apply_gradients(
            zip(critic_2_gradient, self.critic_2.trainable_variables)
        )
        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval == 0:

            # update the actor network
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                # critic outputs the value of the state-action pair
                # similar to the reward but with specific state-action pairs
                # we want to maximize the value of the state-action pair -> gradient ascent
                q1 = -self.critic_1(states, new_actions)
                actor_loss = tf.math.reduce_mean(q1)

            actor_network_gradient = tape.gradient(
                actor_loss, self.actor.trainable_variables
            )
            self.actor.optimizer.apply_gradients(
                zip(actor_network_gradient, self.actor.trainable_variables)
            )

            self.update_network_parameters()

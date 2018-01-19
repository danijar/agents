# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dyna agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from agents import parts
from agents import tools
from agents.algorithms.dyna import utility


class Dyna(object):
  """A vectorized implementation of the Dyna algorithm by Richard Sutton."""

  def __init__(self, batch_env, step, is_training, should_log, config):
    """Create an instance of the Dyna algorithm.

    Args:
      batch_env: In-graph batch environment.
      step: Integer tensor holding the current training step.
      is_training: Boolean tensor for whether the algorithm should train.
      should_log: Boolean tensor for whether summaries should be returned.
      config: Object containing the agent configuration as attributes.
    """
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._observ_filter = parts.StreamingNormalize(
        self._batch_env.observ[0], center=True, scale=True, clip=5,
        name='normalize_observ')
    self._reward_filter = parts.StreamingNormalize(
        self._batch_env.reward[0], center=False, scale=True, clip=10,
        name='normalize_reward')
    self._use_gpu = self._config.use_gpu and utility.available_gpus()
    # TODO: Make sure the agent does not train on true experience. This can be
    # done by passign a variable for `is_training` that is false during
    # interaction with the true environment but true during interaction with the
    # dynamics model.
    self._inner_agent = self._config.inner_agent(
        self._batch_env, self._step, self._is_training, self._should_log,
        tools.AttrDict(self._config.inner_agent_config))
    self._initialize_memory()
    self._initialize_model()
    with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
      self._optimizer = self._config.optimizer(self._config.learning_rate)

  def begin_episode(self, agent_indices):
    """Reset the recurrent states and stored episode.

    Args:
      agent_indices: Tensor containing current batch indices.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('begin_episode/'):
      reset_inner_agent = self._inner_agent.begin_episode(agent_indices)
      reset_buffer = self._current_episodes.clear(agent_indices)
      with tf.control_dependencies([reset_buffer, reset_inner_agent]):
        return tf.constant('')

  def perform(self, agent_indices, observ):
    """Compute batch of actions and a summary for a batch of observation.

    Args:
      agent_indices: Tensor containing current batch indices.
      observ: Tensor of a batch of observations for all agents.

    Returns:
      Tuple of action batch tensor and summary tensor.
    """
    with tf.name_scope('perform/'):
      observ = self._observ_filter.transform(observ)
    return self._inner_agent.perform(agent_indices, observ)

  def experience(
      self, agent_indices, observ, action, reward, unused_done, unused_nextob):
    """Process the transition tuple of the current step.

    When training, add the current transition tuple to the memory and update
    the streaming statistics for observations and rewards. A summary string is
    returned if requested at this step.

    Args:
      agent_indices: Tensor containing current batch indices.
      observ: Batch tensor of observations.
      action: Batch tensor of actions.
      reward: Batch tensor of rewards.
      unused_done: Batch tensor of done flags.
      unused_nextob: Batch tensor of successor observations.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('experience/'):
      return tf.cond(
          self._is_training,
          # pylint: disable=g-long-lambda
          lambda: self._define_experience(
              agent_indices, observ, action, reward), str)

  def _define_experience(self, agent_indices, observ, action, reward):
    """Implement the branch of experience() entered during training."""
    update_filters = tf.summary.merge([
        self._observ_filter.update(observ),
        self._reward_filter.update(reward)])
    with tf.control_dependencies([update_filters]):
      batch = (observ, action, reward)
      append = self._current_episodes.append(batch, agent_indices)
    with tf.control_dependencies([append]):
      norm_observ = self._observ_filter.transform(observ)
      norm_reward = tf.reduce_mean(self._reward_filter.transform(reward))
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
          update_filters,
          self._observ_filter.summary(),
          self._reward_filter.summary(),
          tf.summary.scalar('memory_size', self._memory_index),
          tf.summary.histogram('normalized_observ', norm_observ),
          tf.summary.scalar('normalized_reward', norm_reward)]), str)
      return summary

  def end_episode(self, agent_indices):
    """Add episodes to the memory and perform update steps if memory is full.

    During training, add the collected episodes of the batch indices that
    finished their episode to the memory. If the memory is full, train on it,
    and then clear the memory. A summary string is returned if requested at
    this step.

    Args:
      agent_indices: Tensor containing current batch indices.

    Returns:
       Summary tensor.
    """
    with tf.name_scope('end_episode/'):
      return tf.cond(
          self._is_training,
          lambda: self._define_end_episode(agent_indices), str)

  def _initialize_memory(self):
    """Initialize temporary and permanent memory.

    Initializes the attributes `self._current_episodes`,
    `self._finished_episodes`, and `self._memory_index`. The first memory serves
    to collect multiple episodes in parallel. Finished episodes are copied into
    the next free slot of the second memory. The memory index points to the next
    free slot.
    """
    # We store observation, action, and reward.
    template = (
        self._batch_env.observ[0],
        self._batch_env.action[0],
        self._batch_env.reward[0])
    with tf.variable_scope('dyna_temporary'):
      self._current_episodes = parts.EpisodeMemory(
          template, len(self._batch_env), self._config.max_length, 'episodes')
    self._finished_episodes = parts.EpisodeMemory(
        template, self._config.update_every, self._config.max_length, 'memory')
    self._memory_index = tf.Variable(0, False)

  def _initialize_model(self):
    """Initialize the dynamics model."""
    model = functools.partial(self._config.model, self._config)
    self._model = tf.make_template('model', model)
    output = self._model(
        tf.zeros_like(self._batch_env.observ)[:, None],
        tf.zeros_like(self._batch_env.action)[:, None],
        tf.zeros_like(self._batch_env.action)[:, None, ..., 0],
        tf.ones(len(self._batch_env)))

  def _define_end_episode(self, agent_indices):
    """Implement the branch of end_episode() entered during training."""
    episodes, length = self._current_episodes.data(agent_indices)
    space_left = self._config.update_every - self._memory_index
    use_episodes = tf.range(tf.minimum(
        tf.shape(agent_indices)[0], space_left))
    episodes = tools.nested.map(lambda x: tf.gather(x, use_episodes), episodes)
    append = self._finished_episodes.replace(
        episodes, tf.gather(length, use_episodes),
        use_episodes + self._memory_index)
    with tf.control_dependencies([append]):
      inc_index = self._memory_index.assign_add(tf.shape(use_episodes)[0])
    with tf.control_dependencies([inc_index]):
      memory_full = self._memory_index >= self._config.update_every
      return tf.cond(memory_full, self._training, str)

  def _training(self):
    """Train both the model and the inner agent.

    Training on the episodes collected in the memory. Reset the memory
    afterwards. Always returns a summary string.

    Returns:
      Summary tensor.
    """
    # with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
    with tf.name_scope('training'):
      assert_full = tf.assert_equal(
          self._memory_index, self._config.update_every)
      with tf.control_dependencies([assert_full]):
        (observ, action, reward), length = self._finished_episodes.data()
      with tf.control_dependencies([tf.assert_greater(length, 0)]):
        length = tf.identity(length)
      observ = self._observ_filter.transform(observ)
      reward = self._reward_filter.transform(reward)
      model_summary = self._train_model(
          observ, action, reward, length)
      with tf.control_dependencies([model_summary]):
        clear_memory = tf.group(
            self._finished_episodes.clear(), self._memory_index.assign(0))
      with tf.control_dependencies([clear_memory]):
        agent_summary = self._train_inner_agent()
      with tf.control_dependencies([agent_summary]):
        weight_summary = utility.variable_summaries(
            tf.trainable_variables(), self._config.weight_summaries)
        return tf.summary.merge([
            model_summary, agent_summary, weight_summary])

  def _train_model(self, observ, action, reward, length):
    """Perform multiple update steps of the dynamics model.

    Args:
      observ: Sequences of observations.
      action: Sequences of actions.
      reward: Sequences of rewards.
      length: Batch of sequence lengths.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('train_model'):
      loss, summary = tf.scan(
          lambda _1, _2: self._model_update_step(
              observ, action, reward, length),
          tf.range(self._config.simulated_epochs),
          [0., ''], parallel_iterations=1)
      print_loss = tf.Print(0, [tf.reduce_mean(loss)], 'model loss: ')
      with tf.control_dependencies([loss, print_loss]):
        return summary[self._config.simulated_epochs // 2]

  def _model_update_step(self, observ, action, reward, length):
    """Perform one update step of the dynamics model.

    Args:
      observ: Sequences of observations.
      action: Sequences of actions.
      reward: Sequences of rewards.
      length: Batch of sequence lengths.

    Returns:
      Loss and summary tensors.
    """
    output = self._model(observ, action, reward, length)
    observ_loss = tf.reduce_mean((observ[:, 1:] - output.observ[:, :-1]) ** 2)
    action_loss = tf.reduce_mean((action[:, 1:] - output.action[:, :-1]) ** 2)
    reward_loss = tf.reduce_mean((reward[:, 1:] - output.reward[:, :-1]) ** 2)
    loss = observ_loss + action_loss + reward_loss
    optimize = self._optimizer.minimize(loss)
    with tf.control_dependencies([optimize]):
      return [tf.identity(loss), tf.constant('')]

  def _train_inner_agent(self):
    """

    Returns:
      Summary tensor.
    """
    with tf.name_scope('train_inner_agent'):
      steps = self._config.simulated_epochs * self._config.max_length
      score, summary = tf.scan(
          lambda _1, iteration: self._train_inner_agent_step(iteration),
          tf.range(steps), [tf.zeros((30)), ''], parallel_iterations=1)
      score = tf.reshape(score, [-1])
      mean_score = tf.reduce_mean(tf.boolean_mask(score, tf.is_finite(score)))
      print_score = tf.Print(0, [mean_score], 'inner agent score: ')
      with tf.control_dependencies([score, print_score]):
        return summary[steps // 2]

  def _train_inner_agent_step(self, iteration):
    """

    The returned score tensor is NaN for time steps that are not final.

    Returns:
      Score and summary tensors.
    """
    # TODO: Replace true batch env by learned model. This is just for testing.
    force_reset = tf.equal(iteration, 0)
    done, score, summary = tools.simulate(
        self._batch_env, self._inner_agent, False, force_reset)
    score = tf.where(done, score, np.nan * tf.ones_like(score))
    return [score, summary]

  def _mask(self, tensor, length, padding_value=0):
    """Set padding elements of a batch of sequences to a constant.

    Useful for setting padding elements to zero before summing along the time
    dimension, or for preventing infinite results in padding elements.

    Args:
      tensor: Tensor of sequences.
      length: Batch of sequence lengths.
      padding_value: Value to write into padding elemnts.

    Returns:
      Masked sequences.
    """
    with tf.name_scope('mask'):
      range_ = tf.range(tensor.shape[1].value)
      mask = range_[None, :] < length[:, None]
      if tensor.shape.ndims > 2:
        for _ in range(tensor.shape.ndims - 2):
          mask = mask[..., None]
        mask = tf.tile(mask, [1, 1] + tensor.shape[2:].as_list())
      masked = tf.where(mask, tensor, padding_value * tf.ones_like(tensor))
      return tf.check_numerics(masked, 'masked')

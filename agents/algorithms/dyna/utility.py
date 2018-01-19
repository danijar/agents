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

"""Utilities for the Dyna algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

import tensorflow as tf


def variable_summaries(vars_, groups=None, scope='weights'):
  """Create histogram summaries for the provided variables.

  Summaries can be grouped via regexes matching variables names.

  Args:
    vars_: List of variables to summarize.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for var in vars_:
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(var)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  # pylint: disable=redefined-argument-from-local
  for name, vars_ in grouped.items():
    vars_ = [tf.reshape(var, [-1]) for var in vars_]
    vars_ = tf.concat(vars_, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, vars_))
  return tf.summary.merge(summaries)

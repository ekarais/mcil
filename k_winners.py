# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from __future__ import print_function
import abc
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def maxEntropy(n,k):
  """
  The maximum enropy we could get with n units and k winners
  """

  s = float(k)/n
  if s > 0.0 and s < 1.0:
    entropy = - s * math.log(s,2) - (1 - s) * math.log(1 - s,2)
  else:
    entropy = 0

  return n*entropy


def binaryEntropy(x):
  """
  Calculate entropy for a list of binary random variables

  :param x: (torch tensor) the probability of the variable to be 1.
  :return: entropy: (torch tensor) entropy, sum(entropy)
  """
  entropy = - x*x.log2() - (1-x)*(1-x).log2()
  entropy[x*(1 - x) == 0] = 0
  return entropy, entropy.sum()


def plotDutyCycles(dutyCycle, filePath):
  """
  Create plot showing histogram of duty cycles

  :param dutyCycle: (torch tensor) the duty cycle of each unit
  :param filePath: (str) Full filename of image file
  """
  _,entropy = binaryEntropy(dutyCycle)
  bins = np.linspace(0.0, 0.3, 200)
  plt.hist(dutyCycle, bins, alpha=0.5, label='All cols')
  plt.title("Histogram of duty cycles, entropy=" + str(float(entropy)))
  plt.xlabel("Duty cycle")
  plt.ylabel("Number of units")
  plt.savefig(filePath)
  plt.close()



class k_winners(torch.autograd.Function):
  """
  A simple K-winner take all autograd function for creating layers with sparse
  output.

   .. note::
      Code adapted from this excellent tutorial:
      https://github.com/jcjohnson/pytorch-examples
  """


  @staticmethod
  def forward(ctx, x, dutyCycles, k, boostStrength):
    """
    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as: boostFactors = exp[ -
    boostStrength * (dutyCycle - targetDensity)] Intuitively this means that
    units that have been active (i.e. in the top-k) at the target activation
    level have a boost factor of 1, meaning their activity is not boosted.
    Columns whose duty cycle drops too much below that of their neighbors are
    boosted depending on how infrequently they have been active. Unit that has
    been active more than the target activation level have a boost factor below
    1, meaning their activity is suppressed and they are less likely to be in 
    the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the dutyCycle via an exponential function:

            boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> dutyCycle
                   |
              targetDensity

    :param ctx: 
      Place where we can store information we will need to compute the gradients
      for the backward pass.

    :param x: 
      Current activity of each unit.  

    :param dutyCycles: 
      The averaged duty cycle of each unit.

    :param k: 
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.
                
    :param boostStrength:     
      A boost strength of 0.0 has no effect on x.

    :return: 
      A tensor representing the activity of x after k-winner take all.
    """
    if boostStrength > 0.0:
      targetDensity = float(k) / x.size(1)
      boostFactors = torch.exp((targetDensity - dutyCycles) * boostStrength)
      boosted = x.detach() * boostFactors
    else:
      boosted = x.detach()

    # Take the boosted version of the input x, find the top k winners.
    # Compute an output that contains the values of x corresponding to the top k
    # boosted values
    res = torch.zeros_like(x)
    topk, indices = boosted.topk(k, sorted=False)
    for i in range(x.shape[0]):
      res[i, indices[i]] = x[i, indices[i]]

    ctx.save_for_backward(indices)
    return res


  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass, we set the gradient to 1 for the winning units, and 0
    for the others.
    """
    indices, = ctx.saved_tensors
    grad_x = torch.zeros_like(grad_output, requires_grad=True)

    # Probably a better way to do it, but this is not terrible as it only loops
    # over the batch size.
    for i in range(grad_output.size(0)):
      grad_x[i, indices[i]] = grad_output[i, indices[i]]

    return grad_x, None, None, None



class k_winners2d(torch.autograd.Function):
  """
  A K-winner take all autograd function for CNN 2D inputs (batch, Channel, H, W).

  .. seealso::
       Function :class:`k_winners`
  """


  @staticmethod
  def forward(ctx, x, dutyCycles, k, boostStrength):
    """
    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process. See :meth:`k_winners.forward` for more details.

    :param ctx:
      Place where we can store information we will need to compute the gradients
      for the backward pass.

    :param x:
      Current activity of each unit.

    :param dutyCycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.

    :param boostStrength:
      A boost strength of 0.0 has no effect on x.

    :return:
      A tensor representing the activity of x after k-winner take all.
    """
    batchSize = x.shape[0]
    if boostStrength > 0.0:
      targetDensity = float(k) / (x.shape[1] * x.shape[2] * x.shape[3])
      boostFactors = torch.exp((targetDensity - dutyCycles) * boostStrength)
      boosted = x.detach() * boostFactors
    else:
      boosted = x.detach()

    # Take the boosted version of the input x, find the top k winners.
    # Compute an output that only contains the values of x corresponding to the top k
    # boosted values. The rest of the elements in the output should be 0.
    boosted = boosted.reshape((batchSize, -1))
    xr = x.reshape((batchSize, -1))
    res = torch.zeros_like(boosted)
    topk, indices = boosted.topk(k, dim=1, sorted=False)
    res.scatter_(1, indices, xr.gather(1, indices))
    res = res.reshape(x.shape)

    ctx.save_for_backward(indices)
    return res


  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass, we set the gradient to 1 for the winning units, and 0
    for the others.
    """
    batchSize = grad_output.shape[0]
    indices, = ctx.saved_tensors

    g = grad_output.reshape((batchSize, -1))
    grad_x = torch.zeros_like(g, requires_grad=False)
    grad_x.scatter_(1, indices, g.gather(1, indices))
    grad_x = grad_x.reshape(grad_output.shape)

    return grad_x, None, None, None



def updateBoostStrength(m):
  """
  Function used to update KWinner modules boost strength after each epoch.

  Call using :meth:`torch.nn.Module.apply` after each epoch if required
  For example: ``m.apply(updateBoostStrength)``

  :param m: KWinner module
  """
  if isinstance(m, KWinnersBase):
    if m.training:
      m.boostStrength = m.boostStrength * m.boostStrengthFactor



class KWinnersBase(nn.Module):
  """
  Base KWinners class
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinnersBase, self).__init__()
    assert (boostStrength >= 0.0)

    self.n = n
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.learningIterations = 0

    # Boosting related parameters
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.dutyCyclePeriod = dutyCyclePeriod


  def getLearningIterations(self):
    return self.learningIterations


  @abc.abstractmethod
  def updateDutyCycle(self, x):
    """
     Updates our duty cycle estimates with the new value. Duty cycles are
     updated according to the following formula:

    .. math::
        dutyCycle = \\frac{dutyCycle \\times \\left( period - batchSize \\right)
                            + newValue}{period}
    :param x:
      Current activity of each unit
    """
    raise NotImplementedError


  def updateBoostStrength(self):
    """
    Update boost strength using given strength factor during training
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    if self.k < self.n:
      _, entropy = binaryEntropy(self.dutyCycle)
      return entropy
    else:
      return 0


  def maxEntropy(self):
    """
    Returns the maximum total entropy we can expect from this layer
    """
    return maxEntropy(self.n, self.k)



class KWinners(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

  """


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """

    super(KWinners, self).__init__(n=n, k=k,
                                   kInferenceFactor=kInferenceFactor,
                                   boostStrength=boostStrength,
                                   boostStrengthFactor=boostStrengthFactor,
                                   dutyCyclePeriod=dutyCyclePeriod)
    self.register_buffer("dutyCycle", torch.zeros(self.n))


  def forward(self, x):
    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if self.k >= self.n:
      return F.relu(x)

    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = k_winners.apply(x, self.dutyCycle, k, self.boostStrength)

    if self.training:
      self.updateDutyCycle(x)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
    self.dutyCycle.div_(period)



class KWinners2d(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

  """


  def __init__(self, n, k, channels, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """

    :param n:
      Number of units. Usually the output of the max pool or whichever layer
      preceding the KWinners2d layer.
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinners2d, self).__init__(n=n, k=k,
                                     kInferenceFactor=kInferenceFactor,
                                     boostStrength=boostStrength,
                                     boostStrengthFactor=boostStrengthFactor,
                                     dutyCyclePeriod=dutyCyclePeriod)

    self.channels = channels
    self.register_buffer("dutyCycle", torch.zeros((1, channels, 1, 1)))


  def forward(self, x):
    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if self.k >= self.n:
      return F.relu(x)

    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = k_winners2d.apply(x, self.dutyCycle, k, self.boostStrength)

    if self.training:
      self.updateDutyCycle(x)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize

    scaleFactor = float(x.shape[2] * x.shape[3])
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scaleFactor
    self.dutyCycle.reshape(-1).add_(s)
    self.dutyCycle.div_(period)


  def entropy(self):
    entropy = super(KWinners2d, self).entropy()
    return entropy * self.n / self.channels

import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class GRUModel(nn.Module):
	""" container module"""

	def __init__(self):
		super(GRUModel, self).__init__()
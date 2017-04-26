from torch import optim
import torch.nn as nn
from rasor_model import SquadModel

model = SquadModel(config)
loss_function = nn.CrossEntropyLoss()

for it in training_data:

	model.zero_grad()
	model.hidden = model.init_hidden()

	input_data = #get input data

	a_hats = model(input_data) 
	a_target = #get target answers

	optimizer = optim.Adam(model.parameters())
	loss = loss_function(a_hats, a_target)
	loss.backward()
	optimizer.step()

def _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len):
	# all arguments are concrete ints
	"""
	a way to understand this is-
	lets say that one data point has an answer from 
	3 4
	so it is represented by a unique index (3*30 + 4) = 94
	meaning, we skip over all the first 30 (position 0), next 30 (position 1), next 30 (position 2) 
	"""
	assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
	return ans_start_word_idx * max_ans_len + (ans_end_word_idx - ans_start_word_idx)

def _tt_ans_idx_to_ans_word_idxs(ans_idx, max_ans_len):
	# ans_idx theano int32 variable (batch_size,)
	# max_ans_len concrete int
	"""
	a way to understand this is-
	say we have an ans_idx of 94
	this means that 94/30 = 3 is the start_idx
	and 94%30 = 4, 
	4 + 3 = 7 is the end_idx
	so basically one number can represent both the first and end
	"""
	ans_start_word_idx = ans_idx // max_ans_len
	ans_end_word_idx = ans_start_word_idx + ans_idx % max_ans_len
	return ans_start_word_idx, ans_end_word_idx
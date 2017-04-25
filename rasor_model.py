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
	#check if this vocab_size contains unkown, START, END token as well?
	def __init__(self, vocab_size, emb_dim, pretrained_weight):
		super(GRUModel, self).__init__()

		embed = nn.Embedding(vocab_size, emb_dim)
		embed.weight.requires_grad = False	#do not propagate into the pre-trained word embeddings
		embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

		p_emb = emb[p]        # (max_p_len, batch_size, emb_dim)
		self.ff_align = nn.Linear(inp_dim, emb_dim)

		self.p_end_ff = nn.Linear(2*config.hidden_dim, config.ff_dim)
		self.p_start_ff = nn.Linear(2*config.hidden_dim, config.ff_dim)

		self.w_a_start = nn.Linear(config.ff_dim, 1, bias=False)
		self.w_a_end   = nn.Linear(config.ff_dim, 1, bias=False)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.cross_ent_loss = nn.CrossEntropyLoss()
		self.gru = nn.GRU(2*config.emb_dim, config.hidden_dim, 2, 0.1, bidirectional = True)

	def forward(self):
		p_emb = emb[p]	#(max_p_len, batch_size, emb_dim)
		q_emb = emb[q]	#(max_q_len, batch_size, emb_dim)

		p_star_parts = [p_emb]
		p_star_dim = config.emb_dim

		q_align_ff_p = self.sequence_linear_layer(self.ff_align, p_emb) #(max_p_len, batch_size, ff_dim)
		q_align_ff_q = self.sequence_linear_layer(self.ff_align, q_emb) #(max_q_len, batch_size, ff_dim)

		q_align_ff_p_shuffled = q_align_ff_p.permute(1,0,2)		#(batch_size, max_p_len, ff_dim)
		q_align_ff_q_shuffled = q_align_ff_q.permute(1,2,0)		#(batch_size, ff_dim, max_q_len)

		q_align_scores = torch.bmm(q_align_ff_p_shuffled, q_align_ff_q_shuffled)	#(batch_size, max_p_len, max_q_len)

		#float_p_mask has dimensions (max_p_len, batch_size)
		p_mask_shuffled = torch.unsqueeze(float_p_mask, 2)	#results in (max_p_len, batch_size, 1)
		p_mask_shuffled = p_mask_shuffled.permute(1, 0, 2)	#(batch_size, max_p_len, 1)

		q_mask_shuffled = torch.unsqueeze(float_q_mask, 2)	#results in (max_q_len, batch_size, 1)
		q_mask_shuffled = q_mask_shuffled.permute(1, 0, 2)	#(batch_size, max_q_len, 1)

		pq_mask = torch.bmm(p_mask_shuffled, q_mask_shuffled)	#(batch_size, max_p_len, max_q_len)
		q_align_mask_scores = q_align_scores * pq_mask	#elementwise matrix multiplication

		#this internal pytorch softmax automatically does max, min shifting to prevent overflows
		q_align_weights = self.softmax(q_align_mask_scores)	#(batch_size, max_p_len, max_q_len)
		q_emb_shuffled = q_emb.permute(1, 0, 2)			#(batch_size, max_q_len, emb_dim)

		q_align = torch.bmm(q_align_weights, q_emb_shuffled) #(batch_size, max_p_len, emb_dim)
		q_align_shuffled = q_align.permute(1, 0, 2)	#(max_p_len, batch_size, emd_dim)

		p_star_parts.append(q_align_shuffled)
		p_star_dim += config.emb_dim

		p_star = torch.cat(p_star_parts, 2)	#(max_p_len, batch_size, p_star_dim)

		p_level_h = self.gru(p_star)

		p_stt_ff = self.sequence_linear_layer(self.p_start_ff, p_level_h)	#(max_p_len, batch_size, ff_dim)
		p_end_ff = self.sequence_linear_layer(self.p_end_ff, p_level_h)		#(max_p_len, batch_size, ff_dim)

		word_start_scores = self.sequence_linear_layer(self.w_a_start, p_stt_ff)	#(max_p_len, batch_size)
		word_end_scores = self.sequence_linear_layer(self.w_a_end, p_end_ff)		#(max_p_len, batch_size)

		start_log_probs = self.cross_ent_loss(torch.transpose(word_start_scores, 0, 1))
		end_log_probs = self.cross_ent_loss(torch.transpose(word_end_scores, 0, 1))

		loss = -start_log_probs - end_log_probs
		loss = torch.mean(loss)

	def backward(self):


	#input has dimension (sequence_len, batch_size, input_dim)
	def sequence_linear_layer(self, layer, inp):
		dims = inp.size()
		out = []
		for i in dims[0]:
			inp_i = inp[i, :, :]
			out_i = self.relu(layer(inp_i))
			out.append(out_i)
		return torch.stack(out, 0)
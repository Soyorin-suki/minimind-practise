
from transformers import PretrainedConfig





class MyModelConfig(PretrainedConfig):
	model_type = "minimind"

	def __init__(
		self,
		dropout: float = 0.0,
		bos_token_id: int = 1,
		eos_token_id: int = 2,
		hidden_act: str = "silu",
		hidden_size: int = 512,
		intermediate_size: int = None,
		max_position_embeddings: int = 32768,
		num_attention_heads: int = 8,
		num_hidden_layers: int = 8,
		num_key_value_heads: int = 2,
		vocab_size: int = 6400,
		rms_norm_eps: float = 1e-05,
		rope_theta: int = 1000000,
		inference_rope_scaling: bool = False,
		flash_attention: bool = True,
		############ MoE ############
		use_moe: bool = False,
		num_experts_per_tok: int = 2,
		n_routed_experts: int = 4,
		n_shared_experts: int = 1,
		scoring_func: str = "softmax",
		aux_loss_alpha: float = 0.01,
		seq_aux: bool = True,
		norm_topk_prob: bool = True,
		**kwargs,
	):
		super().__init__(**kwargs)

		self.dropout = dropout
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.hidden_act = hidden_act
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.max_position_embeddings = max_position_embeddings
		self.num_attention_heads = num_attention_heads
		self.num_hidden_layers = num_hidden_layers
		self.num_key_value_heads = num_key_value_heads
		self.vocab_size = vocab_size
		self.rms_norm_eps = rms_norm_eps
		self.rope_theta = rope_theta
		self.inference_rope_scaling = inference_rope_scaling
		self.flash_attention = flash_attention
		self.use_moe = use_moe
		self.num_experts_per_tok = num_experts_per_tok
		self.n_routed_experts = n_routed_experts
		self.n_shared_experts = n_shared_experts
		self.seq_aux = seq_aux
		self.norm_topk_prob = norm_topk_prob
		self.aux_loss_alpha = aux_loss_alpha
		self.scoring_func = scoring_func

		self.rope_scaling = (
			{
				"beta_fast": 32,
				"beta_slow": 1,
				"factor": 16,
				"original_max_position_embeddings": 2048,
				"attention_factor": 1.0,
				"type": "yarn",
			}
			if self.inference_rope_scaling
			else None
		)



###############################
#			model             #
###############################
import math
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast



# RMSNorm 即 Root Mean Square Layer Normalization 均方层归一化
# 与 LayerNorm 相比省略了将均值归一化的过程
# Norm的主要作用是将数值归一化
# 1. 数值稳定性：防止每一层输出的数值越跑越大（梯度爆炸）或越跑越小（梯度消失）
# 2. 加速收敛：把数据限制在合理范围内可以帮助优化器（如Adam）更快找到最优解
# 3. 计算效率：比LN快一些
# RMS(x) = \sqrt{\frac{\sum{x_i^2}}{n}}
# \bar{a} = \frac{x}{RMS(x)}\dot g
# 其中 g 是缩放参数
class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.dim = dim
		# 一个小量，防止除零
		self.eps = eps
		# 此处为需要训练的缩放参数 g 形状 [dim]
		# 会广播作用于 [batch, seq, dim] 的张量上
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x: torch.Tensor):
		# 实现 RMSNorm 的公式
		# torch.rsqrt() 是计算平方根倒数的算子
		# .mean() 方法是求取均值
		# 其中 .mean(-1) 是求取每个向量内部元素的均值
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x: torch.Tensor):
		# 这里是进行前向传播
		# 使用type_as可以保证精度对齐
		return self.weight * self._norm(x.float()).type_as(x)


# RoPE Rotary Positional Embedding 旋转位置编码
# YaRN Yet a RoPE extensioN 一种专门用于**长文本外推**的针对RoPE的改进方案
# 对于高频，不变
# 对于中频，使用插值
# 对于低频，使用线性缩放
# 通过一个 温度系数 来修正Attention的分布，防止长文本下注意力被稀释
# 通过保留高维的原始频率，只在低频维度上进行缩放，使得模型在处理更长的文本时仍然可以保持良好的**局部注意力**
# 预处理位置编码
# cis 表示的是 cis(\theta) = cos(\theta)+i sin(\theta)
def precompute_freqs_cis(
	dim: int,					# 单头维度 head_dim 因为旋转是发生在头内部的
	end: int = 32 * 1024,		# 最大序列长度
	rope_base: float = 1e6,		# 即公式中的theta 基数 决定了不同旋转速度的"衰减率"
	rope_scaling: Optional[dict] = None,
):
	# freqs 为初始化 RoPE 频率
	# attn_factor是后面所用到的一个温差缩放
	freqs, attn_factor = (
		1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)),
		1.0,
	)

	# 如果给出了配置对象，则使用YaRN，否则使用RoPE
	if rope_scaling is not None:
		origen_max, factor, beta_fast, beta_slow = (
			rope_scaling.get("original_max_position_embeddings", 2048),
			rope_scaling.get("factor", 1.0),
			rope_scaling.get("beta_fast", 1.0),
			rope_scaling.get("beta_slow", 1.0),
		)

		# 推断的长度大于训练长度，使用缩放
		if end > origen_max:

			# YaRN : f'(x) = f(i)*((1-k)+k/s),  

			# 波长b对i的映射
			def inv_dim(b: float) -> float:
				return (dim * math.log(origen_max / (b * 2 * math.pi))) / (
					2 * math.log(rope_base)
				)
			# 划分高低维度
			# low: 不需要缩放的高频
			# high: 需要缩放的低频
			low, high = (
				max(math.floor(inv_dim(beta_fast)), 0),
				min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
			)

			# 计算缩放因子
			# low之前，ramp为0，在high部分，ramp为1
			# 在low和high之间线性过度
			ramp = torch.clamp(
				(torch.arange(dim // 2, device=freqs.device).float() - low)
				/ max(high - low, 0.001),
				0,
				1,
			)

			# 当ramp=0时（高频）： 系数为 1， 保持原频率不变
			# 当ramp=1时（低频）： 系数为 1/factor， 对原频率进行线性插值缩放
			# 当ramp在0-1之间时： 平滑过渡
			freqs = freqs * (1 - ramp + ramp / factor)

	# 根据end，生成位置索引t
	t = torch.arange(end, device=freqs.device).float()

	# 计算外积， 将t与频率部分相乘， 得到各个位置的旋转角度
	freqs = torch.outer(t, freqs).float()
	# 乘上缩放系数，保证注意力
	freq_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
	freq_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

	return freq_cos, freq_sin


# 应用 RoPE / YaRN 位置编码
def apply_rotary_pos_emb(
		q:torch.Tensor, 
		k:torch.Tensor, 
		cos:torch.Tensor, 
		sin:torch.Tensor, 
		postion_ids=None, 
		unsqueeze_dim=1):
	# [a,b]->[-b,a]
	def rotate_half(x: torch.Tensor):
		# x.shape[-1] 取最后一个维度的重点
		# x[..., x.shape[-1]//2 :] 取出后半部分
		return torch.cat(
			   (-x[..., x.shape[-1]//2 :],x[..., : x.shape[-1]//2]),
			   dim=-1
		)
	# x_rotate=x*cos+rotate_half(x)*sin
	# 考虑对sin和cos进行type_as方式混合精度计算报错
	q_embd=(q*cos.unsqueeze(unsqueeze_dim)) + (
		rotate_half(q)*sin.unsqueeze(unsqueeze_dim)
	)
	k_embd=(k*cos.unsqueeze(unsqueeze_dim)) + (
		rotate_half(k)*sin.unsqueeze(unsqueeze_dim)
	)

	return q_embd, k_embd
	


def repeat_kv(x:torch.Tensor, num_rep:int)->torch.Tensor:
	# bs: batch size
	# slen: sequence length
	bs, slen, num_key_value_heads, head_dim = x.shape
	if num_rep == 1:
		return x
	return (
		x[:,:,:,None,:]
		.expand(bs, slen, num_key_value_heads, num_rep, head_dim)
		.reshape(bs, slen, num_key_value_heads * num_rep, head_dim)
	)



class Attention(nn.Module):
	def __init__(self,args:MyModelConfig):
		super().__init__()

		# k/v的头数，这是一个静态的配置变量
		# 如果配置中存在num_key_value_heads，即k/v的头数，则使用分组查询（GQA）
		# 否则默认采用全量注意力头（MHA）
		self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

		# 检查 q的头数 是否是 k/v的头数量的整数倍
		assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be devided by num_key_value_heads"
		
		# q的头数
		self.n_local_heads = args.num_attention_heads
		# k/v的头数，这是一个运行时变量，如果是多卡计算的话可能会变，但是此处在数值上与 num_key_value_heads 相等
		self.n_local_kv_heads = args.num_key_value_heads
		# 重复倍数，应用于repeat_kv
		self.n_rep = self.n_local_heads//self.n_local_kv_heads
		# 每个头的维度
		self.head_dim = args.hidden_size//args.num_attention_heads

		# 设置 bias = False 可以减少参数计算量，能让训练更稳定，配合RMSNorm更好
		# q的投影矩阵，将q投影从隐藏层投影到多头注意力空间，准备进行QKV注意力机制的计算
		# 或者说是从 通用特征空间 投影到 查询特征空间
		# 隐藏层是输入层与输出层间的所有层
		self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
		# k的投影矩阵，将k投影从隐藏层投影到多头注意力空间，准备进行QKV计算
		self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
		# v的投影矩阵，同上
		self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
		# out输出投影矩阵，将多头注意力算出的矩阵再降维映射回原本的隐藏层size大小，并进行信息融合
		# 此处的两者恰好相等，因此主要作用是进行信息融合
		self.o_proj = nn.Linear(args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)
		
		# dropout，通过随机损失一些值，来防止过拟合
		# 在Softmax之后，与V相乘之前
		# 作用于注意力权重，防止模型过度依赖某些值，学习更加泛化的特征
		self.attn_dropout = nn.Dropout(args.dropout)
		# 在o_proj之后，在加回残差路径之前
		# 作用与输出投影，配合残差网络使用，防止某一层对最终输出的影响过大，起到正则化作用
		# 疑问：为什么防止某一层对最终输出影响过大，可以起到正则化作用（之后研究）
		# ans：1. 分担风险，防止模型过度依赖某一层
		# 2. 强制协作，迫使模型去通过残差路径去寻求其他层的帮助 （疑问：为什么残差路径与其它层的帮助有关，之后研究）
		# 3. 结果是使得模型不过度依赖任意一层
		self.resid_dropout = nn.Dropout(args.dropout)
		# 记录 dropout 概率的静态参数
		self.dropout = args.dropout
		
		# 对硬件相关进行设置，可能暂时忽略其具体作用
		self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

	def forward(
		self,
		x: torch.Tensor,
		position_embeddings: Tuple[torch.Tensor, torch.Tensor],		# 接受cos, sin
		past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
		use_cache: bool = False,
		attention_mask: Optional[torch.Tensor] = None
	):
		bsz, seq_len, _ = x.shape
		xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
		xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
		xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
		xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

		cos, sin = position_embeddings
		xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

		if past_key_values is not None:
			xk = torch.cat([past_key_values[0], xk], dim=1)
			xv = torch.cat([past_key_values[1], xv], dim=1)
		# use_cache 在推理时开启，训练时关闭
		past_kv = (xk,xv) if use_cache else None

		xq, xk, xv =(
			# 将 (batch, sequence, head, head_dim) 转换为 (batch, head, sequence, head_dim)
			# 以便于进行多头注意力计算
			xq.transpose(1,2),
			repeat_kv(xk, self.n_rep).transpose(1,2),
			repeat_kv(xv, self.n_rep).transpose(1,2)
		)

		if (self.flash 
	  		and (seq_len>1) 
			and (past_key_values is None) 
			and (attention_mask is None or torch.all(attention_mask==1))):
			# 如果有 CUDA 加速的话采用对应的算子
			output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
		else :
			# 计算相关性分数矩阵
			scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
			# 应用因果掩码
			scores[:,:,:, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

			if attention_mask is not None:
				extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
				extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
				scores = scores + extended_attention_mask
				
			# 开始应用 softmax 将分数转换为概率
			scores = F.softmax(scores.float(), dim=-1).type_as(xq)
			# 进行 attention 后的 dropout
			scores = self.attn_dropout(scores)
			# 应用 xv
			output = scores @ xv

		
		output = output.transpose(1,2).reshape(bsz, seq_len, -1)
		output = self.resid_dropout(self.o_proj(output))
		return output, past_kv


class FeedForward(nn.Module):

	def __init__(
		self,
		args: MyModelConfig
	):
		super().__init__()
		if args.intermediate_size is None:
			# 一般的工程实践中发现 8/3 倍比较好
			intermediate_size = int(args.hidden_size * 8 / 3)
			# 向上取整，设为 64 的整数倍
			args.intermediate_size = 64 * ((intermediate_size+64-1)//64)

		# 升维矩阵
		self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
		# 降维矩阵
		self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
		# 激活所用的门矩阵
		self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
		self.dropout = nn.Dropout(args.dropout)
		# 激活函数 activate function
		self.act_fn = ACT2FN[args.hidden_act]

	def forward(self, x: torch.Tensor):
		return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))




class MoEGate(nn.Module):

	def __init__(
		self,
		config: MyModelConfig
	):
		super().__init__()
		self.config = config
		# 设置选择的专家的数量
		self.top_k = config.num_experts_per_tok
		# 设置一共有多少个专家
		self.n_routed_experts = config.n_routed_experts

		# 设置计算每个 token 对各个 专家 的打分函数 （使用 dot-product/linear+softmax）
		self.scoring_func = config.scoring_func
		# 设置 辅助负载均衡损失的 参数 alpha
		# 用于控制该辅助损失的权重
		self.alpha = config.aux_loss_alpha
		# 设置是否采用 序列级辅助损失 否则使用 批级辅助损失
		self.seq_aux = config.seq_aux

		# 设置是否对选出的 top-k 专家的概率进行重新归一化（使其和为1）
		self.norm_topk_prob = config.norm_topk_prob
		# 路由 gating 输入的维度，应该等于隐藏层的维度
		self.gating_dim = config.hidden_size
		# 路由的矩阵，这里是形状为 (n_routed_experts, gating_dim)
		self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
		# 重置参数
		self.reset_parameters()
	
	def reset_parameters(self)->None:
		# 这里是调用 何凯明 的初始化矩阵方法，这里的 a=\sqrt 5 是一个经验参数
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))

	def forward(
		self,
		hidden_states: torch.Tensor	# 应该是 (batch, seq, hidden_size)
	):
		bsz, seq_len, h = hidden_states.shape
		# 将 hidden_states 的形状由 (batch, seq, hidden_size) 变成了 (batch*seq, hidden_size) 
		# 便于之后进行
		hidden_states = hidden_states.view(-1,h)
		# logits = x*W^T 计算每个 token 对每个 expert 的打分
		# 变成了 (batch*seq, num_expert)
		logits = F.linear(hidden_states, self.weight, None)
		if self.scoring_func == 'softmax':
			# 将原始的 logits 概率化
			scores = logits.softmax(dim=-1)
		else:
			raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
		
		# topk_weight 是 (batch*seq, k) 表示每一个bacth中的每一个token对所有n_routed_experts 的概率中前k个
		# topk_idx 是对应的下标
		topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
		
		# 是否需要归一化
		if self.top_k > 1 and self.norm_topk_prob:
			denominator = topk_weight.sum(dim=-1,keepdim=True)+ 1e-20
			topk_weight = topk_weight/denominator
		
		# 如果是训练模式
		if self.training and self.alpha>0.0:
			# (batch*seq, num_expert)
			scores_for_aux = scores
			# 计算 aux 时选择的 topk
			aux_topk = self.top_k
			# (batch, seq*num_expert)
			# 统计每个 batch 内，每个 token 被分到了哪个 expert
			topk_idx_for_aux_loss = topk_idx.view(bsz,-1)

			# 如果采用 序列级辅助损失函数
			if self.seq_aux:
				# (batch, seq, num_expert)
				scores_for_seq_aux = scores_for_aux.view(bsz,seq_len,-1)
				# (batch, num_expert) 
				# 计数器，计数每一个 batch 的每一个 num_expert 都被选择了几次
				ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
				# ce = ce * (e/(s*k))
				# 这里直接一步计算了，先是累加，之后原地除 seq*top_k/num_expert
				# 如果完全均匀，这里的每个值都应该是1
				ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len*aux_topk, device=hidden_states.device)).div_(
					seq_len*aux_topk/self.n_routed_experts
				)
				# loss = \alpha* \frac{1}{batch}*\sum_{batch}\sum{i} ce_{batch,i}*P_{batch,i}
				aux_loss = (ce*scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()*self.alpha
			else:
				# 为每个 batch 的每个 token 的每个 expert 创建一个 one-hot 向量
				# (batch*seq*num_expert, num_expert)
				mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
				# 计算每个 expert 被选中的比例
				# (batch*seq*num_expert, num_expert) -> (num_expert)
				ce = mask_ce.float().mean(0)
				# 计算每个 expert 被选中的全局比例
				Pi = scores_for_aux.mean(0)
				# 放大 ce ，使其均匀分布变为1
				fi = ce*self.n_routed_experts
				# loss = \alpha* \sum_{i}{P_i*f_i}
				aux_loss = (Pi*fi).sum()*self.alpha
		else :
			# 如果没有在训练，就返回一个 (batch*seq)
			aux_loss = scores.new_zeros(1).squeeze()
		# 这里的 aux_loss 是一个标量
		return topk_idx, topk_weight, aux_loss



class MoEFeedForward(nn.Module):
	def __init__(
		self,
		config:MyModelConfig
	):
		super().__init__()
		self.config = config
		self.experts = nn.ModuleList([
			FeedForward(config) for _ in range(config.n_routed_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts > 0:
			self.shared_experts = nn.ModuleList([
				FeedForward(config) for _ in range(config.n_shared_experts)
			])
	
	def forward(
		self,
		x: torch.Tensor
	):
		identity = x
		orig_shape = x.shape
		bsz, seq_len, h = x.shape
		
		topk_idx, topk_weight, aux_loss = self.gate(x)
		topk_idx: torch.Tensor
		topk_weight: torch.Tensor
		# (batch*seq, hidden)
		x = x.view(-1, x.shape[-1])
		# (batch*seq*k)
		flat_topk_idx = topk_idx.view(-1)
		if self.training :
			# (batch*seq*k, hidden)
			x = x.repeat_interleave(self.config.num_experts_per_tok,dim=0)
			# (batch*seq*k, hidden)
			y = torch.empty_like(x,dtype=x.dtype)

			for i,expert in enumerate(self.experts):
				# 选中的 token 数
				# (n_i, hidden)
				expert_out =  expert(x[flat_topk_idx == i])
				if expert_out.shape[0] > 0:
					# 把结果写回原位置
					y[flat_topk_idx == i] = expert_out.to(y.dtype)
				else:
					# 防止 expert 的梯度断掉
					y[flat_topk_idx == i] = expert_out.to(y.dtype)+0*sum(p.sum() for p in expert.parameters())

			# (batch*seq, hidden)*
			y = (y.view(*topk_weight.shape,-1)*topk_weight.unsqueeze(-1)).sum(dim= 1)
			y = y.view(*orig_shape)
		else :
			y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1,1)).view(
				*orig_shape
			)
		
		# 加上共享专家的
		if self.config.n_shared_experts > 0:
			for expert in self.shared_experts:
				y = y + expert(identity)
		self.aux_loss = aux_loss
		return y
	
	# MoE 推理方法
	@torch.no_grad()
	def moe_infer(
		self,
		x: torch.Tensor,
		flat_expert_indices: torch.Tensor,
		flat_expert_weight: torch.Tensor
	)->torch.Tensor:
		# (batch*seq*k,hidden)
		expert_cache = torch.zeros_like(x)
		# 分拣
		idxs = flat_expert_indices.argsort()
		tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
		token_idxs = idxs//self.config.num_experts_per_tok
		for i, end_idx in enumerate(tokens_per_expert):
			start_idx = 0 if i == 0 else tokens_per_expert[i-1]
			if start_idx == end_idx:
				continue
			expert = self.experts[i]
			expert_token_idx = token_idxs[start_idx:end_idx]
			expert_tokens = x[expert_token_idx]
			expert_out = expert(expert_tokens).to(expert_cache.dtype)
			expert_out: torch.Tensor
			expert_out.mul_(flat_expert_weight[idxs[start_idx:end_idx]])
			expert_cache.scatter_add_(
				0, expert_token_idx.view(-1,1).repeat(1,x.shape[-1]),expert_out
			)
		return expert_cache
		
		


class MyModelBlock(nn.Module):
	def __init__(
		self,
		layer_id: int,
		config: MyModelConfig
	):
		super().__init__()
		self.num_attention_heads = config.num_attention_heads
		self.hidden_size = config.hidden_size
		self.head_dim = config.hidden_size // self.num_attention_heads
		self.self_attn = Attention(config)

		self.layer_id = layer_id
		self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		# MLP multi-layer perceptron 多层感知机
		# 是 前馈神经网络的同义词
		self.mlp = (
			FeedForward(config)
			if not config.use_moe
			else MoEFeedForward(config)
		)

	def forward(
		self,
		hidden_status,
		position_embeddings,
		past_key_values = None,
		use_cache = False,
		attention_mask = None
	):
		residual = hidden_status
		hidden_status, present_key_value = self.self_attn(
			self.input_layernorm(hidden_status),
			position_embeddings,
			past_key_values,
			use_cache,
			attention_mask
		)
		hidden_status += residual
		hidden_status = hidden_status + self.mlp(self.post_attention_layernorm(hidden_status))
		return hidden_status, present_key_value


class MyModelModel(nn.Module):
	def __init__(
		self,
		config: MyModelConfig
	):
		self.config = config
		super().__init__()
		self.vocab_size, self.num_hidden_layer = (
			config.vocab_size,
			config.num_hidden_layers
		)

		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

		self.dropout = nn.Dropout(config.dropout)

		self.layers = nn.ModuleList(
			[MyModelBlock(i,config) for i in range(self.num_hidden_layer)]
		)

		self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

		# RoPE 预计算
		freqs_cos, freqs_sin = precompute_freqs_cis(
			dim= config.hidden_size // config.num_attention_heads,
			end= config.max_position_embeddings,
			rope_base= config.rope_theta,
			rope_scaling= config.rope_scaling,
		)

		# 这是一个不需要计算梯度等的常量
		# 直接将其存入内存中，会随GPU切换时切换
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)

	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		past_key_values: Optional[torch.Tensor] = None,
		use_cache: bool = False,
		**kwargs
	):
		batch_size, seq_len = input_ids.shape

		# 这是一个 hugging face 数据格式相关的代码，暂时不太需要管
		if hasattr(past_key_values, "layers"):
			past_key_values = None
		past_key_values = past_key_values or [None]*len(self.layers)
		
		start_pos = (
			past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
		)

		hidden_states = self.dropout(self.embed_tokens(input_ids))

		position_embeddings = (
			self.freqs_cos[start_pos:start_pos+seq_len],
			self.freqs_sin[start_pos:start_pos+seq_len]
		)

		presents = []
		for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
			hidden_states, present = layer(
				hidden_states,
				position_embeddings,
				past_key_values = past_key_value,
				use_cache = use_cache,
				attention_mask = attention_mask
			)
			presents.append(present)

		hidden_states = self.norm(hidden_states)

		aux_loss = sum(
			[
				layer.mlp.aux_loss 
				for layer in self.layers
				if isinstance(layer.mlp,MoEFeedForward)
			]
		)

		return hidden_states, presents, aux_loss




# Causal Langruage Model 因果语言模型
# PreTrainedModel 是 nn.Model 的增强版，继承之后拥有了 save_pretrained() 以及 from_pretrained() 方法
# 可以像加载官方 Llama 模型一样读写权重和配置
# GenerationMixin 给予了 generate() 方法，实现 forward() 方法后，自动处理 beam search 等
class MyModelForCausalLM(PreTrainedModel, GenerationMixin):
	config_class = MyModelConfig

	def __init__(
		self,
		config: MyModelConfig
	):
		self.config = config
		super().__init__(config)
		self.model = MyModelModel(config)

		# laugrage model head
		# 负责将隐藏层的特征向量映射到词表大小
		self.lm_head = nn.Linear(
			self.config.hidden_size,
			self.config.vocab_size,
			bias=False
		)

		# 权重共享
		# 输出层的权重与嵌入层的权重共享
		# 避免多计算一个 weight, 在计算时更加简单
		self.model.embed_tokens.weight = self.lm_head.weight
		
	
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
		past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
		use_cache: bool = False,
		logits_to_keep: Union[int, torch.Tensor] = 0,		# 保存多少位的 logits
		**args
	):
		
		hidden_states, past_key_values, aux_loss = self.model(
			input_ids = input_ids,
			attention_mask = attention_mask,
			past_key_values = past_key_values,
			use_cache = use_cache,
			**args
		)

		# logits 一般指的是 SoftMax 前的原始输出值
		# 如果 logits to keep 是整数，那就保留最后 n 个位置
		# 生成的时候只使用最后的 logits 来预测下一个 token
		slice_indices = (
			slice(-logits_to_keep, None)
			if isinstance(logits_to_keep, int)
			else logits_to_keep
		)
		logits = self.lm_head(hidden_states[:,slice_indices, :])
		loss = None
		if labels is not None:
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = F.cross_entropy(
				shift_logits.view(-1, shift_logits.size(-1)),
				shift_labels.view(-1),
				ignore_index=-100,
			)

		out_put = CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=past_key_values,
			hidden_states=hidden_states
		)
		out_put.aux_loss = aux_loss

		return out_put
		
		




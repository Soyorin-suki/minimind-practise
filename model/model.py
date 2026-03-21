
from transformers import PretrainedConfig






class MyModelConfig(PretrainedConfig):
	model_type = "mymodel"

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
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN



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


# 预处理 RoPE / YaRN位置编码
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
		bsz, seq_len = x.shape
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

		if self.flash and (seq_len>1) and (past_key_values is None) and (attention_mask is None or torch.all(attention_mask==1)):
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
			intermediate_size = int(args.hidden_size * 8 / 3)
			args.intermediate_size = 64 * ((intermediate_size+64-1)//64)

		self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
		self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
		self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
		self.dropout = nn.Dropout(args.dropout)
		self.act_fn = ACT2FN[args.hidden_act]

	def forward(self, x: torch.Tensor):
		return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))





		




# Author:妖怪鱼
# Date:2024/8/24
# Introduction:模拟实现 self attention

import torch
from torch.nn.functional import softmax

# note 准备词嵌入向量，这里代表了3个词
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

# note 随机初始化K,Q,V的权重矩阵
w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)


keys = x @ w_key
querys = x @ w_query
values = x @ w_value

# note 得到 Q K V

print("Keys: \n", keys)
print("Querys: \n", querys)
print("Values: \n", values)

# note 计算注意力分数
attn_scores = querys @ keys.T
print("的注意力分数为:\n ",attn_scores)
attn_scores_softmax = softmax(attn_scores, dim=-1)
print("softmax后的注意力分数为:\n ",attn_scores_softmax)

# # 为了使得后续方便，这里简略将计算后得到的分数赋予了一个新的值
# # For readability, approximate the above as follows
# attn_scores_softmax = [
#   [0.0, 0.5, 0.5],
#   [0.0, 1.0, 0.0],
#   [0.0, 0.9, 0.1]
# ]
# attn_scores_softmax = torch.tensor(attn_scores_softmax)
# print(attn_scores_softmax)

weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
print(weighted_values)
attention_output = weighted_values.sum(dim=0)
print(attention_output)  # note 最终输出
# Project
智能物联网课程作业
## 实验观察
文件体积对比： 记录导出模型和量化后的具体文件大小，并计算压缩比（例如：1750MB / 6800MB ≈ 25%）。
效果观察：
1. 量化后的模型在 CPU 上生成文字的速度如何？（主观感受即可，如：每秒大概几个字？）

   生成文字速度较快，稳定保持在每秒10-20字符左右

2. 量化后的模型是否依然能生成通顺的中文？请在报告中附上 2-3 轮对话的截图。

   可以正常生成中文<img width="1470" height="956" alt="截屏2026-01-15 16 35 24" src="https://github.com/user-attachments/assets/cf5afa63-99de-4bb8-9f8a-2cfde434ac08" />

   
   

## 思考题：
在任务三中，目前的推理逻辑每生成一个字都要重新计算整个序列。如果想要加速，应该引入什么机制？（提示：KV Cache）

1.缓存已计算的 K/V

Transformer 的自注意力中，每一层的 K（Key）和 V（Value）只依赖已生成的 token。
KV Cache 保存前面 token 的 K/V，下一个 token 生成时只计算新 token 的 K/V，然后和缓存做 Attention。

2.只传入新 token
第一次生成时传入整个 prompt，生成第一个 token。
后续每次只传入上一次生成的 token，同时传入缓存的 K/V。

3.更新缓存
每生成一个 token，将对应的新 K/V 拼接到缓存中，下一步使用。

4.实现方法
outputs = model(input_ids=new_token_ids, past_key_values=past, use_cache=True)
next_token_logits = outputs.logits[:, -1, :]
past = outputs.past_key_values

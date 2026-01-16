import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# ================= 配置 =================
MODEL_PATH = "qwen3_int8.onnx"
TOKENIZER_PATH = "./Qwen3-1.7B"
SEQ_LEN = 32          #  固定序列长度，量化模型要求
MAX_NEW_TOKENS = 20   # 每轮生成最大 token 数

# ================= 加载 tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

# ================= 加载 ONNX =================
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
print("ONNX inputs:", [i.name for i in sess.get_inputs()])

# ================= 自回归生成函数 =================
def generate(prompt: str):
    # 预处理输入
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[:SEQ_LEN]  # 截断过长
    print(f"Prompt token len: {len(input_ids)}")

    for step in range(MAX_NEW_TOKENS):
        #  padding 到固定长度 SEQ_LEN
        pad = np.zeros((1, SEQ_LEN), dtype=np.int64)
        pad[0, :len(input_ids)] = input_ids

        # ONNX 推理
        logits = sess.run(None, {"input_ids": pad})[0]  # shape: (1, SEQ_LEN, vocab_size)

        # 获取当前最后位置 token
        last_pos = len(input_ids) - 1
        next_token = int(np.argmax(logits[0, last_pos]))

        # EOS 判断
        if next_token == tokenizer.eos_token_id:
            break

        #  输出 token
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)

        #  更新 input_ids
        input_ids.append(next_token)

        # 保持固定长度，超出截断
        if len(input_ids) > SEQ_LEN:
            input_ids = input_ids[-SEQ_LEN:]

    print("\n")

# ================= 对话循环 =================
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Qwen: ", end="", flush=True)
        generate(user_input)

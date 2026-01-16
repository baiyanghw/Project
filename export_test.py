import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# ================= 配置 =================
MODEL_PATH = "qwen3_fp32.onnx"
MODEL_DIR = "./Qwen3-1.7B"
SEQ_LEN = 32
MAX_NEW_TOKENS = 20

# ================= tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)
tokenizer.padding_side = "left"

# ================= ONNX =================
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX inputs:", [i.name for i in sess.get_inputs()])


def generate(prompt: str):
    input_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]

    print("Prompt token len:", len(input_ids))

    for _ in range(MAX_NEW_TOKENS):
        if len(input_ids) >= SEQ_LEN:
            break

        cur_len = len(input_ids)

        # ===== 左 padding =====
        padded = np.zeros((1, SEQ_LEN), dtype=np.int64)
        padded[0, -cur_len:] = input_ids

        logits = sess.run(
            None,
            {"input_ids": padded}
        )[0]  # (1, seq, vocab)

        # ===== 关键修复点 =====
        next_token_logits = logits[0, -1]
        next_token = int(np.argmax(next_token_logits))

        input_ids.append(next_token)

    return tokenizer.decode(input_ids, skip_special_tokens=True)


# ================= 测试 =================
if __name__ == "__main__":
    prompt = "讲解一下人工智能"
    output = generate(prompt)

    print("\n===== 模型输出 =====")
    print(output)

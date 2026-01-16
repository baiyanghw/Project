import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# ============================================================
# 1️⃣ ONNX-safe causal mask 补丁
# ============================================================
# 目标：
# - 不依赖 attention_mask
# - 固定生成 (B, 1, S, S) 的 causal mask
# - 避免 ReduceMean / 广播错误
def mask_patch(*args, **kwargs):
    # 尝试从 kwargs 里取 input_shape
    input_shape = kwargs.get("input_shape", None)
    if input_shape is not None:
        bsz, seq_len = input_shape
    else:
        # fallback：从 input_ids 推断
        input_ids = kwargs.get("input_ids", None)
        if input_ids is not None:
            bsz, seq_len = input_ids.shape
        else:
            bsz, seq_len = 1, 32  # 最后兜底

    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))

    # causal mask：下三角为 0，上三角为 -inf
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=1)

    # reshape -> (B, 1, S, S)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(bsz, 1, seq_len, seq_len)

    return mask


# 应用补丁
transformers.masking_utils.create_causal_mask = mask_patch
print(">>> [Patch Applied] 已应用 ONNX-safe causal mask")

# ============================================================
# 2️⃣ 模型 Wrapper（关闭 cache，只返回 logits）
# ============================================================
class Qwen3ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
        )
        return outputs.logits


# ============================================================
# 3️⃣ 主程序
# ============================================================
model_path = "./Qwen3-1.7B"
output_file = "qwen3_fp32.onnx"

print("--- Loading Model ---")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="eager",  # 必须
)
base_model.eval()

model = Qwen3ONNXWrapper(base_model)

# ============================================================
# 4️⃣ 构造 dummy 输入
# ============================================================
dummy_input_ids = torch.ones((1, 32), dtype=torch.long)

print(f"--- Exporting to {output_file} ---")

# ============================================================
# 5️⃣ ONNX 导出（⚠️ 不使用 use_dynamo）
# ============================================================
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_input_ids,),
        output_file,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

print("✅ Export Success! ONNX 模型已生成:", output_file)

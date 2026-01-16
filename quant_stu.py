import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)
from transformers import AutoTokenizer
import numpy as np
import os

SEQ_LEN = 32

# ================= 校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer

        # 获取模型输入名，防止 mismatch
        session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_names = [inp.name for inp in session.get_inputs()]

        # 扩展校准数据
        self.texts = [
            "人工智能是计算机科学的一个分支。",
            "深度学习需要大量计算资源。",
            "自然语言处理正在快速发展。",
            "机器学习在金融领域有很多应用。",
            "计算机视觉是 AI 的核心方向之一。",
            "医疗影像分析可以借助深度学习提高诊断精度。",
            "自动驾驶汽车依赖于传感器和计算机视觉。",
            "强化学习在游戏 AI 中应用广泛。",
            "推荐系统利用用户历史行为进行预测。",
            "聊天机器人正在改变客户服务体验。",
            "Python 是流行的编程语言。",
            "深度神经网络可以解决复杂任务。",
            "卷积神经网络适合图像处理。",
            "循环神经网络适合序列数据。",
            "Transformer 模型在 NLP 中效果显著。",
            "大模型需要大量计算和存储资源。",
            "模型量化可以减小模型体积和加速推理。",
            "ONNX 格式便于跨平台部署模型。",
            "数据预处理对模型效果至关重要。",
            "训练集和验证集的分布应保持一致。",
            "过拟合是深度学习常见问题。",
            "正则化可以缓解过拟合。",
            "优化器如 Adam 可以加速训练。",
            "学习率调度可以提高收敛速度。",
            "GPU 可以大幅提升训练速度。",
            "多模态学习结合图像和文本信息。",
            "迁移学习可以利用预训练模型。",
            "模型蒸馏可以压缩大模型。",
            "激活函数决定非线性能力。",
            "批量归一化有助于训练稳定。",
            "注意力机制让模型关注重要信息。",
            "编码器-解码器架构适合序列到序列任务。",
            "Artificial Intelligence is transforming the world.",
            "Machine learning algorithms can predict stock prices.",
            "Data augmentation improves model robustness.",
            "Self-supervised learning reduces labeled data requirements.",
            "Computer graphics and vision often work together.",
            "Big data analytics requires scalable infrastructure.",
            "Robotics relies on sensors and actuators.",
            "Speech recognition models are widely deployed.",
            "GANs can generate realistic images from noise.",
            "Reinforcement learning has solved complex games.",
        ]
        self.data_iter = iter(self.texts)

    def get_next(self):
        text = next(self.data_iter, None)
        if text is None:
            return None

        # tokenizer -> numpy
        tokens = self.tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        # padding 到固定长度
        input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        length = min(len(tokens), SEQ_LEN)
        input_ids[0, :length] = tokens[:length]

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = input_ids

        return feed


# ================= 主程序 =================
model_fp32 = "qwen3_fp32.onnx"
model_int8 = "qwen3_int8.onnx"

if not os.path.exists(model_fp32):
    print("❌ 未找到 FP32 ONNX 模型")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen3-1.7B",
    trust_remote_code=True
)

dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print("--- Starting INT8 Quantization ---")

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    use_external_data_format=True,  # >2GB 大模型
)

print("✅ INT8 Quantization Complete!")
print(f"INT8 model saved to: {model_int8}")

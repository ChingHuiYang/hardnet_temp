import torch
import torch.quantization

# 加载模型
model = YourModel()

# 定义量化参数
bits = 8
scale = 1.0 / 255
zero_point = 0

# 遍历模型参数，对权重进行量化
for name, param in model.named_parameters():
    if 'weight' in name:
        quantized_weight = torch.quantization.quantize_linear(param, scale=scale, zero_point=zero_point, dtype=torch.qint8)
        setattr(model, name, quantized_weight)

# 保存量化后的模型
torch.save(model.state_dict(), 'quantized_model.pth')

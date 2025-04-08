import torch
import torch.nn as nn

import sgl_kernel
import sgl_kernel.cpu

#   "quantization_config": {
#     "activation_scheme": "dynamic",
#     "fmt": "e4m3",
#     "quant_method": "fp8",
#     "weight_block_size": [
#       128,
#       128
#     ]
#   },

class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)

# BF16 input
# FP8 weight
# weight_scale_inv
# weight_block_size
# bias

# Ref: FP8 weight to BF16 weight, F.linear
# Optimized: sgl-kernel


block_size = 2

scales_block_size = [block_size, block_size]

# has_bias = True
has_bias = False
M, K, N = 32, 32, 16
fp8_max = 448.0

model = Mod(K, N, has_bias).eval()
data = torch.rand(M, K).bfloat16() * 0.1
weight = model.linear.weight  # (N, K)

# Prepare compute dtype
compute_dtype = torch.bfloat16

# Step 1: reshape to 8x8 grid of (128x128) blocks
weight_blocks = weight.view(N // block_size, block_size, K // block_size, block_size)
weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()  # shape: (8, 8, 128, 128)

# Step 2: compute per-block max abs values → scale
abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)  # (8, 8, 1, 1)
scales = abs_max / fp8_max
scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # avoid division by zero

# Step 3: quantize → FP8
q_blocks = (weight_blocks / scales).to(torch.float8_e4m3fn)
q_blocks_reshape = q_blocks.permute(0, 2, 1, 3).contiguous()
q_blocks_reshape = q_blocks_reshape.view(N, K)

# Step 4: dequantize
dq_blocks = q_blocks.float() * scales  # back to float32

# Step 5: reshape back to (N, K)
dq_blocks = dq_blocks.permute(0, 2, 1, 3).contiguous()  # (8, 128, 8, 128)
w_dq = dq_blocks.view(N, K).to(compute_dtype)

# Step 6: forward pass
if has_bias:
    bias = model.linear.bias
    output1 = torch.matmul(data.to(compute_dtype), w_dq.T) + bias
else:
    output1 = torch.matmul(data.to(compute_dtype), w_dq.T)

# TODO: add prepack
# TODO: test case with tail
scales_squeeze = scales.view(N // block_size, K // block_size)
output2 = sgl_kernel.cpu.fp8_scaled_mm(
    data, q_blocks_reshape, scales_squeeze, scales_block_size, bias if has_bias else None, data.dtype, is_vnni=False
)
print(output1)
print(output2)

# assert output1 == output2
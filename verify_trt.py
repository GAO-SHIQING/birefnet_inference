"""TensorRT Engine 精度验证脚本 (使用 PyTorch 管理 GPU 内存)"""
import sys
sys.path.insert(0, '.')
import torch, torch.nn.functional as F, time, numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import tensorrt as trt

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ENGINE_PATH = "birefnet_fp16_fixed.engine"

# 加载 TRT Engine
with open(ENGINE_PATH, 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()
print(f"[TRT] Engine 加载完成: {len(engine_data)/1e6:.1f} MB")
print(f"[TRT] I/O: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}")

# 分配显存 (PyTorch CUDA tensors)
H, W = 1024, 1024
input_shape = (1, 3, H, W)
output_shape = (1, 1, H, W)

d_input = torch.empty(input_shape, dtype=torch.float32, device='cuda')
d_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')

context.set_input_shape("input_image", input_shape)
context.set_tensor_address("input_image", d_input.data_ptr())
context.set_tensor_address("output_logits", d_output.data_ptr())

# 准备测试数据
img = Image.new('RGB', (1024, 1024), color=(0, 255, 0))
draw = ImageDraw.Draw(img)
draw.ellipse([200, 200, 800, 800], fill=(255, 0, 0))

transform_fn = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tensor_np = transform_fn(img).unsqueeze(0).numpy().astype(np.float32)

# PyTorch 基准
from transformers import AutoModelForImageSegmentation
model_pt = AutoModelForImageSegmentation.from_pretrained(
    'zhengpeng7/BiRefNet', trust_remote_code=True
).float().eval()
with torch.no_grad():
    out_pt = model_pt(torch.from_numpy(tensor_np))[-1].sigmoid()

# TRT 推理 + 速度测试
stream = torch.cuda.Stream()
d_input.copy_(torch.from_numpy(tensor_np))
context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
out_trt = torch.sigmoid(d_output.cpu())  # 先获取一次输出用于精度对比

for _ in range(10):  # warmup
    d_input.copy_(torch.from_numpy(tensor_np))
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()

N = 50
t0 = time.time()
for _ in range(N):
    d_input.copy_(torch.from_numpy(tensor_np))
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
t_trt = (time.time() - t0) / N * 1000

# 精度对比
diff = (out_pt - out_trt).abs()
print(f"\n===== TRT vs PyTorch 精度 =====")
print(f"  mean_diff: {diff.mean().item():.6e}")
print(f"  max_diff:  {diff.max().item():.6e}")

fg = out_trt[0, 0, 400:600, 400:600].mean().item()
bg = out_trt[0, 0, 50:100, 50:100].mean().item()
print(f"  前景 mean: {fg:.4f}")
print(f"  背景 mean: {bg:.4f}")

# 速度对比
dummy = torch.randn(1, 3, 1024, 1024)
model_pt_fp16 = model_pt.float().cuda().eval()
for _ in range(10):
    with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
        _ = model_pt_fp16(dummy.cuda())[-1].sigmoid()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
        _ = model_pt_fp16(dummy.cuda())[-1].sigmoid()
    torch.cuda.synchronize()
t_pt = (time.time() - t0) / 50 * 1000

print(f"\n===== 速度对比 (RTX 3060, 1024x1024) =====")
print(f"  PyTorch FP16: {t_pt:.1f} ms")
print(f"  TensorRT FP16: {t_trt:.1f} ms")
print(f"  加速比: {t_pt/t_trt:.1f}x")

status = "✅" if diff.max() < 1e-2 else "⚠️"
print(f"\n{status} TRT 精度验证: max_diff={diff.max():.6f}")

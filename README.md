# BiRefNet Inference

基于 [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 的高性能图像分割推理，支持 PyTorch 和 TensorRT 双引擎，可将图片背景一键去除。

## 安装

```bash
# 创建环境
conda create -n birefnet python=3.11 -y && conda activate birefnet

# 安装依赖
pip install -r requirements.txt
```

硬件要求：NVIDIA 显卡 + CUDA 驱动，显存 >= 4GB。

## 模型

首次运行时自动从 HuggingFace 下载模型权重（约 450MB，缓存到 `~/.cache/huggingface/`）。

- 模型：[zhengpeng7/BiRefNet](https://huggingface.co/zhengpeng7/BiRefNet)
- 架构：Swin-Large Backbone + BiRefNet Decoder（220M 参数）
- 输入：RGB 图像，自动缩放到 1024×1024
- 输出：单通道 mask（灰度）或 RGBA 透明图

其他可用模型变体：

| 模型 | HuggingFace ID | 特点 |
|------|---------------|------|
| BiRefNet (标准) | `zhengpeng7/BiRefNet` | 1024²，通用分割（默认） |
| BiRefNet_HR | `zhengpeng7/BiRefNet_HR` | 2048²，高精度 |
| BiRefNet_dynamic | `zhengpeng7/BiRefNet_dynamic` | 256~2304 动态分辨率 |
| BiRefNet-matting | `zhengpeng7/BiRefNet-matting` | 抠图专用 |
| BiRefNet_lite | `zhengpeng7/BiRefNet_lite` | Swin-Tiny，轻量快速 |

切换模型：修改 `infer.py` 中的 `zhengpeng7/BiRefNet` 为对应 ID。

## 使用

### 命令行

```bash
# PyTorch 引擎：背景去除
python infer.py -i 照片.jpg -o 去背景.png

# PyTorch 引擎：只输出灰度 mask
python infer.py -i 照片.jpg -o mask.png --alpha-only

# TensorRT 引擎（需先编译，见下文）
python infer_trt.py -i 照片.jpg -o 去背景.png
```

### Python 调用

```python
from infer import remove_background
from PIL import Image

img = Image.open("photo.jpg")
result = remove_background(img)  # RGBA
result.save("out.png")
```

### API 服务

```bash
# 启动服务
python server.py

# 调用
curl -X POST -F "image=@photo.jpg" http://localhost:8000/segment -o out.png
curl "http://localhost:8000/health"
```

`server.py` 默认使用 TensorRT 引擎。如果没有编译过 engine，设置 `USE_TRT = False` 切换到 PyTorch。

## TensorRT 编译（可选）

要获得 3x 加速，需要将 PyTorch 模型编译为 TensorRT Engine：

```bash
# 步骤 1：导出 ONNX（约 5 分钟，生成 ~970MB ONNX 文件）
python export_onnx.py

# 步骤 2：编译 TRT Engine（约 7 分钟，生成 ~560MB engine 文件）
python build_trt_engine.py

# 步骤 3：验证精度
python verify_trt.py
```

Engine 文件编译一次后可反复使用。目前仅支持 1024×1024 固定分辨率，原因是 Swin Transformer 的窗口注意力在 TRT 下不支持动态形状。

## 预处理规则

输入图像经过以下处理：

```
原始图 → 缩放到 1024×1024 → Normalize(ImageNet) → 模型推理
       → sigmoid → bilinear 插值还原原始尺寸 → 输出 mask
```

## 性能基准

| 引擎 | 设备 | 分辨率 | 延迟 | 显存 |
|------|------|--------|------|------|
| PyTorch FP16 | RTX 3060 | 1024×1024 | 367 ms | 3.7 GB |
| TensorRT FP16 | RTX 3060 | 1024×1024 | 119 ms | 2.1 GB |
| PyTorch FP16 | RTX 4090 | 1024×1024 | ~58 ms | — |
| PyTorch FP16 | A100 | 1024×1024 | ~69 ms | — |

## 精度验证

TensorRT FP16 vs PyTorch FP32（1024×1024，RTX 3060）：

| 指标 | 值 |
|------|-----|
| mean_diff | 1.6e-5 |
| max_diff | 0.054（5.4%，仅 0.1% 边缘像素）|
| P99 误差 | < 1e-6（99% 像素零误差）|
| PSNR | 48 dB |

FP16 量化带来的精度损失在实际使用中不可见。

## 项目结构

```
infer.py              # PyTorch 推理入口
infer_trt.py          # TensorRT 推理入口
server.py             # FastAPI HTTP 服务
export_onnx.py        # PyTorch → ONNX 导出
build_trt_engine.py   # ONNX → TensorRT Engine 编译
verify_step1.py       # PyTorch 模型验证
verify_trt.py         # TRT 精度/速度验证
deform_conv2d_onnx_exporter.py  # deform_conv2d ONNX 自定义算子
models/               # BiRefNet 模型定义 (官方)
config.py             # 模型配置 (官方)
requirements.txt      # 依赖
```

## 常见问题

**Q: CUDA out of memory？**  
A: 当前模型固定 1024×1024 输入，4GB 显存即可。如果同时跑其他模型可能不够，先释放显存。

**Q: 支持其他分辨率吗？**  
A: TRT Engine 仅 1024²。PyTorch 引擎可通过修改 `ENGINE_SIZE` 尝试其他分辨率，但 2048² 在 12GB 卡上会 OOM。

**Q: 模型下载很慢？**  
A: 首次下载约 450MB。设置 `export HF_ENDPOINT=https://hf-mirror.com` 使用镜像加速。

**Q: 能部署到服务器吗？**  
A: `server.py` 就是 FastAPI 服务，可直接 `uvicorn` 启动或套一层 Nginx。

## 致谢

- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — 模型架构与权重
- [masamitsu-murase/deform_conv2d_onnx_exporter](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter) — 可变形卷积 ONNX 导出

## License

MIT

"""BiRefNet 多分辨率 TensorRT 推理。
根据输入图像尺寸自动选择最优 Engine，保留大图细节。
用法: python infer_multi.py -i photo.jpg -o result.png
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tensorrt as trt

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SCRIPT_DIR = os.path.dirname(__file__)
ENGINE_DIR = os.path.join(SCRIPT_DIR, 'models', 'engines')

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 可用的 Engine 分辨率 (按升序排列)
_engine_cache = {}

def _list_engines():
    """扫描 engine 目录, 返回按分辨率升序的 [(size, path), ...] 列表"""
    engines = []
    if not os.path.isdir(ENGINE_DIR):
        return engines
    for f in sorted(os.listdir(ENGINE_DIR)):
        if f.endswith('.engine'):
            try:
                # birefnet_1024_fp16.engine -> 1024
                size = int(f.split('_')[1])
                engines.append((size, os.path.join(ENGINE_DIR, f)))
            except (IndexError, ValueError):
                continue
    engines.sort(key=lambda x: x[0])
    return engines

AVAILABLE_ENGINES = _list_engines()

def get_engine(size):
    """懒加载 Engine (文件名格式: birefnet_{size}_fp16.engine)"""
    if size not in _engine_cache:
        path = os.path.join(ENGINE_DIR, f'birefnet_{size}_fp16.engine')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Engine 不存在: {path}')
        with open(path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        ctx = engine.create_execution_context()
        d_input = torch.empty(1, 3, size, size, dtype=torch.float32, device='cuda')
        d_output = torch.empty(1, 1, size, size, dtype=torch.float32, device='cuda')
        ctx.set_input_shape('input_image', (1, 3, size, size))
        ctx.set_tensor_address('input_image', d_input.data_ptr())
        ctx.set_tensor_address('output_logits', d_output.data_ptr())
        _engine_cache[size] = {
            'context': ctx,
            'd_input': d_input,
            'd_output': d_output,
            'size': size,
        }
    return _engine_cache[size]

def select_engine(image_size):
    """
    选择最接近输入尺寸的 Engine。
    策略: 取 >= max(image_w, image_h) 的最小 Engine;
    如果超过最大值, 用最大的 Engine。
    """
    max_edge = max(image_size)
    for eng_size, _ in AVAILABLE_ENGINES:
        if eng_size >= max_edge:
            return eng_size
    return AVAILABLE_ENGINES[-1][0]  # 用最大的

def remove_background(image: Image.Image) -> Image.Image:
    """输入 PIL Image, 返回 RGBA 背景去除图"""
    original_size = (image.width, image.height)
    eng_size = select_engine(original_size)
    eng = get_engine(eng_size)

    img = image.convert('RGB').resize((eng_size, eng_size), resample=Image.LANCZOS)
    tensor = TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)

    stream = torch.cuda.Stream()
    eng['d_input'].copy_(torch.from_numpy(tensor))
    eng['context'].execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    mask = torch.sigmoid(eng['d_output'].cpu()).float()
    mask = F.interpolate(mask, size=(original_size[1], original_size[0]),
                         mode='bilinear', align_corners=True)
    mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)

    rgba = image.convert('RGBA')
    r, g, b, _ = rgba.split()
    return Image.merge('RGBA', (r, g, b, Image.fromarray(mask_np)))

def main():
    if not AVAILABLE_ENGINES:
        print('未找到 Engine 文件! 请先运行 build_trt_engine.py')
        return

    print(f'可用 Engine: {[s for s, _ in AVAILABLE_ENGINES]}')

    parser = argparse.ArgumentParser(description='BiRefNet 多分辨率 TRT 推理')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', default='out.png', help='输出路径')
    parser.add_argument('--alpha-only', '-a', action='store_true', help='只输出灰度 mask')
    args = parser.parse_args()

    image = Image.open(args.input)
    eng_size = select_engine((image.width, image.height))
    print(f'输入: {args.input} ({image.width}x{image.height}) -> Engine: {eng_size}x{eng_size}')

    if args.alpha_only:
        eng = get_engine(eng_size)
        original_size = (image.width, image.height)
        img = image.convert('RGB').resize((eng_size, eng_size), resample=Image.LANCZOS)
        tensor = TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)

        stream = torch.cuda.Stream()
        eng['d_input'].copy_(torch.from_numpy(tensor))
        eng['context'].execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        mask = torch.sigmoid(eng['d_output'].cpu()).float()
        mask = F.interpolate(mask, size=(original_size[1], original_size[0]),
                             mode='bilinear', align_corners=True)
        Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).save(args.output)
    else:
        result = remove_background(image)
        result.save(args.output)

    print(f'输出: {args.output}')

if __name__ == '__main__':
    main()

"""BiRefNet 推理 API 服务 (API Gateway)
架构:
  客户端 → FastAPI (预处理/后处理) → gRPC → Triton → TRT Engine
                                      ↘ 直连模式 (无 Triton 时自动回退)
"""
import io, os, logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ——— 图像预处理常量 ———
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MAX_SIZE = 3096
MIN_SIZE = 256
ALIGN = 32

BASE_DIR = os.path.dirname(__file__)
ENGINE_DIR = os.path.join(BASE_DIR, 'models', 'engines')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pretrained')

USE_TRITON = True       # 启用 Triton gRPC 模式
TRITON_URL = 'localhost:8001'
TRITON_MODEL = 'birefnet_dynamic'  # ensemble 名称 (如有); 多分辨率时动态拼接

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ——— Triton gRPC 客户端 ———
_triton_client = None

# ——— 直连模式 Engine 池 (Triton 不可用时回退) ———
_engines = {}
_engine_sizes = []
_pt_model = None

app = FastAPI(title='BiRefNet Inference API')


# ═══════════════════════════════════════════════
# 预处理
# ═══════════════════════════════════════════════

def smart_resize(image: Image.Image, max_size=MAX_SIZE, min_size=MIN_SIZE, align=ALIGN):
    """
    智能等比缩放: 限制尺寸在 [min_size, max_size], 并对齐到 align 倍数。
    BiRefNet dec_ipt_split 要求输入能被 32 整除 (Swin 特征图网格).
    """
    w, h = image.size
    max_edge, min_edge = max(w, h), min(w, h)

    if max_edge > max_size:
        scale = max_size / max_edge
        w, h = round(w * scale), round(h * scale)
    elif min_edge < min_size:
        scale = min_size / min_edge
        w, h = round(w * scale), round(h * scale)

    w = ((w + align - 1) // align) * align
    h = ((h + align - 1) // align) * align

    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.LANCZOS)
    return image


def preprocess(image: Image.Image):
    """预处理: 只做 resize + normalize, 不做 sigmoid"""
    img = smart_resize(image.convert('RGB'))
    tensor = _transform(img).unsqueeze(0).numpy().astype(np.float32)
    return tensor, (image.height, image.width)


def postprocess(logits, original_size):
    """后处理: sigmoid + bilinear 插值还原原始分辨率"""
    mask = torch.sigmoid(torch.from_numpy(logits)).float()
    mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
    return (mask.squeeze().numpy() * 255).astype(np.uint8)


# ═══════════════════════════════════════════════
# Triton gRPC 客户端
# ═══════════════════════════════════════════════

def _get_triton_client():
    global _triton_client
    if _triton_client is None:
        import tritonclient.grpc as grpcclient
        _triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    return _triton_client


def _select_triton_model(w, h):
    """根据预处理后的宽高选择对应的 engine 分辨率"""
    max_edge = max(w, h)
    for s in _engine_sizes:
        if s >= max_edge:
            return s
    return _engine_sizes[-1]


def triton_infer(tensor: np.ndarray, original_size: tuple) -> np.ndarray:
    """通过 Triton gRPC 执行推理"""
    import tritonclient.grpc as grpcclient
    _, _, H, W = tensor.shape
    eng_size = _select_triton_model(W, H)

    client = _get_triton_client()
    model_name = f'birefnet_{eng_size}'

    # 对齐到 Engine 分辨率
    if W != eng_size or H != eng_size:
        x = torch.from_numpy(tensor).cuda()
        x = F.interpolate(x, size=(eng_size, eng_size), mode='bilinear', align_corners=True)
        tensor = x.cpu().numpy()

    inputs = grpcclient.InferInput('input_image', list(tensor.shape), 'FP32')
    inputs.set_data_from_numpy(tensor)

    result = client.infer(model_name, [inputs])
    logits = result.as_numpy('output_logits')

    return postprocess(logits, original_size)


# ═══════════════════════════════════════════════
# 直连模式 (Triton 回退)
# ═══════════════════════════════════════════════

def _load_direct_engines():
    global _engines, _engine_sizes
    import tensorrt as trt
    if not os.path.isdir(ENGINE_DIR):
        return
    for f in sorted(os.listdir(ENGINE_DIR)):
        if not f.endswith('.engine'):
            continue
        try:
            size = int(f.split('_')[1])
        except (IndexError, ValueError):
            continue
        path = os.path.join(ENGINE_DIR, f)
        with open(path, 'rb') as fh:
            engine_data = fh.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        ctx = engine.create_execution_context()
        d_input = torch.empty(1, 3, size, size, dtype=torch.float32, device='cuda')
        d_output = torch.empty(1, 1, size, size, dtype=torch.float32, device='cuda')
        ctx.set_input_shape('input_image', (1, 3, size, size))
        ctx.set_tensor_address('input_image', d_input.data_ptr())
        ctx.set_tensor_address('output_logits', d_output.data_ptr())
        _engines[size] = {
            'context': ctx, 'd_input': d_input, 'd_output': d_output,
            'stream': torch.cuda.Stream(), 'size': size,
        }
        _engine_sizes.append(size)
    if _engine_sizes:
        _engine_sizes.sort()
        logger.info(f'Engine 池: {_engine_sizes}')


def _select_direct_engine(w, h):
    max_edge = max(w, h)
    for s in _engine_sizes:
        if s >= max_edge:
            return s
    return _engine_sizes[-1]


def direct_infer(tensor: np.ndarray, original_size: tuple) -> np.ndarray:
    """直连 TRT Engine 推理 (无 Triton)"""
    _, _, H, W = tensor.shape
    eng_size = _select_direct_engine(W, H)
    eng = _engines[eng_size]

    img_tensor = torch.from_numpy(tensor).cuda()
    if W != eng_size or H != eng_size:
        img_tensor = F.interpolate(img_tensor, size=(eng_size, eng_size),
                                   mode='bilinear', align_corners=True)

    eng['d_input'].copy_(img_tensor)
    eng['context'].execute_async_v3(eng['stream'].cuda_stream)
    eng['stream'].synchronize()

    mask = torch.sigmoid(eng['d_output'].cpu()).float()
    mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
    return (mask.squeeze().numpy() * 255).astype(np.uint8)


# ═══════════════════════════════════════════════
# PyTorch 回退
# ═══════════════════════════════════════════════

def _load_pt_model():
    global _pt_model
    from transformers import AutoModelForImageSegmentation
    source = MODEL_PATH if os.path.isfile(os.path.join(MODEL_PATH, 'model.safetensors')) else 'zhengpeng7/BiRefNet'
    _pt_model = AutoModelForImageSegmentation.from_pretrained(
        source, trust_remote_code=True
    ).cuda().eval()
    logger.info('PyTorch 模型就绪')


def pt_infer(tensor: np.ndarray, original_size: tuple) -> np.ndarray:
    with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
        logits = _pt_model(torch.from_numpy(tensor).cuda())[-1]
    return postprocess(logits.float().cpu().numpy(), original_size)


# ═══════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════

@app.on_event('startup')
async def startup():
    global USE_TRITON

    if USE_TRITON:
        # 测试 Triton 连接
        try:
            client = _get_triton_client()
            client.is_server_live()
            # 拉取可用模型列表
            resp = client.get_model_repository_index()
            models = [m.name for m in resp.models]
            logger.info(f'Triton 已连接, 可用模型: {models}')
            # 从模型名提取分辨率
            global _engine_sizes
            for m in models:
                if m.startswith('birefnet_'):
                    try:
                        _engine_sizes.append(int(m.split('_')[1]))
                    except (IndexError, ValueError):
                        pass
            _engine_sizes.sort()
            if not _engine_sizes:
                logger.warning('Triton 中未找到 birefnet_* 模型')
        except Exception as e:
            logger.warning(f'Triton 连接失败 ({e}), 回退到直连模式')
            USE_TRITON = False

    if not USE_TRITON:
        _load_direct_engines()
        if not _engines:
            logger.info('无 TRT Engine, 加载 PyTorch 模型...')
            _load_pt_model()


# ═══════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════

@app.post('/segment')
async def segment(
    image: UploadFile = File(...),
    alpha_only: bool = Query(False),
    resolution: int = Query(0, description='指定分辨率, 0=自动'),
    mode: str = Query('auto', description='推理模式: auto/triton/direct/pytorch'),
):
    data = await image.read()
    img = Image.open(io.BytesIO(data))
    tensor, original_size = preprocess(img)

    # 确定推理函数
    if mode == 'pytorch':
        infer_fn = pt_infer
    elif mode == 'direct':
        infer_fn = direct_infer
    elif mode == 'triton':
        infer_fn = triton_infer
    else:  # auto
        if USE_TRITON and _triton_client is not None:
            infer_fn = triton_infer
        elif _engines:
            infer_fn = direct_infer
        else:
            infer_fn = pt_infer

    mask_np = infer_fn(tensor, original_size)

    if alpha_only:
        buf = io.BytesIO()
        Image.fromarray(mask_np).save(buf, format='PNG')
        return Response(buf.getvalue(), media_type='image/png')

    rgba = img.convert('RGBA')
    r, g, b, _ = rgba.split()
    result = Image.merge('RGBA', (r, g, b, Image.fromarray(mask_np)))
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    return Response(buf.getvalue(), media_type='image/png')


@app.get('/engines')
async def list_engines():
    return {
        'mode': 'triton' if USE_TRITON else ('direct' if _engines else 'pytorch'),
        'available': _engine_sizes,
        'triton_url': TRITON_URL if USE_TRITON else None,
    }


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'backend': 'triton' if USE_TRITON else ('direct' if _engines else 'pytorch'),
        'engines': _engine_sizes,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)

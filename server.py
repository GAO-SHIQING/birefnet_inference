"""BiRefNet 推理 API 服务 (FastAPI, 不依赖 Triton)"""
import io
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ENGINE_PATH = "birefnet_fp16_fixed.engine"
ENGINE_SIZE = (1024, 1024)
USE_TRT = True  # 设为 False 则使用 PyTorch

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_model = None
_stream = None
_d_input = None
_d_output = None
_context = None

app = FastAPI(title="BiRefNet Inference API")


@app.on_event("startup")
async def startup():
    global _model, _stream, _d_input, _d_output, _context, USE_TRT

    if USE_TRT:
        import tensorrt as trt
        with open(ENGINE_PATH, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        _context = engine.create_execution_context()
        _d_input = torch.empty(1, 3, *ENGINE_SIZE, dtype=torch.float32, device='cuda')
        _d_output = torch.empty(1, 1, *ENGINE_SIZE, dtype=torch.float32, device='cuda')
        _context.set_input_shape("input_image", (1, 3, *ENGINE_SIZE))
        _context.set_tensor_address("input_image", _d_input.data_ptr())
        _context.set_tensor_address("output_logits", _d_output.data_ptr())
        _stream = torch.cuda.Stream()
        print(f"TRT Engine 加载完成 ({ENGINE_PATH})")
    else:
        from transformers import AutoModelForImageSegmentation
        _model = AutoModelForImageSegmentation.from_pretrained(
            'zhengpeng7/BiRefNet', trust_remote_code=True
        ).cuda().eval()
        print("PyTorch 模型加载完成")


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    alpha_only: bool = Query(False),
):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    original_size = (img.height, img.width)

    img_resized = img.resize(ENGINE_SIZE, resample=Image.LANCZOS)
    tensor = _transform(img_resized).unsqueeze(0).numpy().astype(np.float32)

    if USE_TRT:
        _d_input.copy_(torch.from_numpy(tensor))
        _context.execute_async_v3(_stream.cuda_stream)
        _stream.synchronize()
        mask = torch.sigmoid(_d_output.cpu()).float()
    else:
        with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
            logits = _model(torch.from_numpy(tensor).cuda())[-1]
        mask = torch.sigmoid(logits).float().cpu()

    mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
    mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)

    if alpha_only:
        buf = io.BytesIO()
        Image.fromarray(mask_np).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    rgba = img.convert("RGBA")
    r, g, b, _ = rgba.split()
    result = Image.merge("RGBA", (r, g, b, Image.fromarray(mask_np)))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "trt" if USE_TRT else "pytorch"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

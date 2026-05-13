"""BiRefNet 单图推理脚本。用法: python infer.py --input in.jpg --output mask.png"""
import argparse, sys, os
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MODEL_SIZE = (1024, 1024)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pretrained')
HF_REPO = 'zhengpeng7/BiRefNet'

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_cache = {}

def _resolve_model_source():
    if os.path.isfile(os.path.join(MODEL_PATH, 'model.safetensors')):
        return MODEL_PATH
    print('本地模型不存在，从 HuggingFace 下载...')
    return HF_REPO

def get_model():
    if 'model' not in _cache:
        _cache['model'] = AutoModelForImageSegmentation.from_pretrained(
            _resolve_model_source(), trust_remote_code=True
        ).cuda().eval()
    return _cache['model']

def remove_background(image: Image.Image) -> Image.Image:
    """输入 PIL Image (RGB)，返回去除背景的 RGBA 图。"""
    model = get_model()
    original_size = (image.height, image.width)

    # 预处理: 统一缩放到 1024x1024
    img = image.convert('RGB').resize(MODEL_SIZE, resample=Image.LANCZOS)
    tensor = TRANSFORM(img).unsqueeze(0).cuda()

    # 推理
    with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
        logits = model(tensor)[-1]
        mask = torch.sigmoid(logits).float()

    # 后处理: 还原原始分辨率
    mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
    mask = mask.squeeze().cpu().numpy()

    # 合成 RGBA
    rgba = image.convert('RGBA')
    r, g, b, _ = rgba.split()
    alpha = Image.fromarray((mask * 255).astype('uint8'))
    return Image.merge('RGBA', (r, g, b, alpha))

def main():
    parser = argparse.ArgumentParser(description='BiRefNet 图像分割推理')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', default='mask.png', help='输出图片路径 (默认 mask.png)')
    parser.add_argument('--alpha-only', '-a', action='store_true',
                        help='只输出灰度 mask，不做背景去除')
    args = parser.parse_args()

    image = Image.open(args.input)
    print(f'输入: {args.input} ({image.size[0]}x{image.size[1]})')

    if args.alpha_only:
        # 只输出灰度 mask
        model = get_model()
        original_size = (image.height, image.width)
        img = image.convert('RGB').resize(MODEL_SIZE, resample=Image.LANCZOS)
        tensor = TRANSFORM(img).unsqueeze(0).cuda()
        with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
            logits = model(tensor)[-1]
            mask = torch.sigmoid(logits).float()
        mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
        Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype('uint8')).save(args.output)
    else:
        # 背景去除 (RGBA)
        result = remove_background(image)
        result.save(args.output)

    print(f'输出: {args.output}')

if __name__ == '__main__':
    main()

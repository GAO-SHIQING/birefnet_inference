"""BiRefNet 单图推理脚本。用法: python infer.py --input in.jpg --output mask.png"""
import argparse, sys
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageSegmentation

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MAX_SIZE = 3096
MIN_SIZE = 256
ALIGN = 32

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_cache = {}  # 单例模型，避免重复加载

def get_model():
    if 'model' not in _cache:
        _cache['model'] = AutoModelForImageSegmentation.from_pretrained(
            'zhengpeng7/BiRefNet', trust_remote_code=True
        ).cuda().eval()
    return _cache['model']

def smart_resize(image, max_size=MAX_SIZE, min_size=MIN_SIZE, align=ALIGN):
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

def remove_background(image: Image.Image) -> Image.Image:
    """输入 PIL Image (RGB)，返回去除背景的 RGBA 图。"""
    model = get_model()
    original_size = (image.height, image.width)

    # 预处理
    img = smart_resize(image.convert('RGB'))
    tensor = TRANSFORM(img).unsqueeze(0).cuda()

    # 推理
    with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
        logits = model(tensor)[-1]
        mask = torch.sigmoid(logits).float()

    # 后处理：还原原始分辨率
    mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
    mask = mask.squeeze().cpu().numpy()  # (H, W), 值域 [0, 1]

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
        img = smart_resize(image.convert('RGB'))
        tensor = TRANSFORM(img).unsqueeze(0).cuda()
        with torch.amp.autocast('cuda', dtype=torch.float16), torch.no_grad():
            logits = model(tensor)[-1]
            mask = torch.sigmoid(logits).float()
        mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(mask_np).save(args.output)
    else:
        # 背景去除 (RGBA)
        result = remove_background(image)
        result.save(args.output)

    print(f'输出: {args.output}')

if __name__ == '__main__':
    main()

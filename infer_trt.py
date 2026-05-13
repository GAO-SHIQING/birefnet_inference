"""BiRefNet TensorRT 推理入口。需要先执行 build_trt_engine.py 生成 engine 文件。"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tensorrt as trt

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ENGINE_PATH = os.path.join(os.path.dirname(__file__), "models", "pretrained", "birefnet_fp16_fixed.engine")
ENGINE_SIZE = (1024, 1024)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class BiRefNetTRT:
    def __init__(self, engine_path=ENGINE_PATH):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self.d_input = torch.empty(1, 3, *ENGINE_SIZE, dtype=torch.float32, device='cuda')
        self.d_output = torch.empty(1, 1, *ENGINE_SIZE, dtype=torch.float32, device='cuda')
        self.context.set_input_shape("input_image", (1, 3, *ENGINE_SIZE))
        self.context.set_tensor_address("input_image", self.d_input.data_ptr())
        self.context.set_tensor_address("output_logits", self.d_output.data_ptr())
        self.stream = torch.cuda.Stream()

    def infer(self, image: Image.Image) -> Image.Image:
        original_size = (image.height, image.width)
        img = image.convert('RGB').resize(ENGINE_SIZE, resample=Image.LANCZOS)
        tensor = TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)

        self.d_input.copy_(torch.from_numpy(tensor))
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        mask = torch.sigmoid(self.d_output.cpu()).float()
        mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
        mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)

        rgba = image.convert('RGBA')
        r, g, b, _ = rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(mask_np)))


def main():
    parser = argparse.ArgumentParser(description='BiRefNet TensorRT 推理')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', default='out.png', help='输出路径')
    parser.add_argument('--alpha-only', '-a', action='store_true', help='只输出灰度 mask')
    parser.add_argument('--engine', default=ENGINE_PATH, help='TRT engine 路径')
    args = parser.parse_args()

    model = BiRefNetTRT(args.engine)
    image = Image.open(args.input)

    if args.alpha_only:
        original_size = (image.height, image.width)
        img = image.convert('RGB').resize(ENGINE_SIZE, resample=Image.LANCZOS)
        tensor = TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)
        model.d_input.copy_(torch.from_numpy(tensor))
        model.context.execute_async_v3(model.stream.cuda_stream)
        model.stream.synchronize()
        mask = torch.sigmoid(model.d_output.cpu()).float()
        mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
        Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8)).save(args.output)
    else:
        result = model.infer(image)
        result.save(args.output)

    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()

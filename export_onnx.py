"""BiRefNet PyTorch → ONNX 导出脚本 —— 步骤 2+6"""
import sys, os
sys.path.insert(0, '.')
import torch

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
print("[1] deform_conv2d ONNX 算子已注册")

SCRIPT_DIR = os.path.dirname(__file__)

MODELS = {
    'birefnet': {
        'model_path': os.path.join(SCRIPT_DIR, 'models', 'pretrained'),
        'onnx_path': os.path.join(SCRIPT_DIR, 'birefnet_1024_fixed.onnx'),
    },
    'birefnet_dynamic': {
        'model_path': os.path.join(SCRIPT_DIR, 'models', 'dynamic'),
        'onnx_path': os.path.join(SCRIPT_DIR, 'birefnet_dynamic_1024_fixed.onnx'),
    },
}


def export_onnx(model_path, resolution, out_path, dynamic=False):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.float().eval()
    H, W = resolution
    print(f"[2] {model_path}: FP32 CPU, size={H}x{W}")

    dummy = torch.randn(1, 3, H, W)
    kwargs = dict(
        verbose=False, opset_version=17,
        input_names=['input_image'], output_names=['output_logits'],
    )
    if dynamic:
        kwargs['dynamic_axes'] = {
            'input_image': {0: 'batch', 2: 'height', 3: 'width'},
            'output_logits': {0: 'batch', 2: 'height', 3: 'width'},
        }

    torch.onnx.export(model, dummy, out_path, **kwargs)
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print(f"[3] {out_path}: {len(m.graph.node)} ops, {os.path.getsize(out_path)/1e6:.1f}MB")
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export BiRefNet to ONNX')
    parser.add_argument('--model', '-m', choices=list(MODELS.keys()), default='birefnet',
                        help='Which model to export')
    args = parser.parse_args()

    cfg = MODELS[args.model]
    export_onnx(cfg['model_path'], (1024, 1024), cfg['onnx_path'], dynamic=False)

    print("\n---")
    print("TRT 引擎仅支持固定 1024x1024 分辨率 (Swin Transformer 窗格限制)。")
    print("换模型: 修改 --model 参数即可。")

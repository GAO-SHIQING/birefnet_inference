"""BiRefNet PyTorch → ONNX 导出脚本 —— 步骤 2+6"""
import sys, os
sys.path.insert(0, '.')
import torch

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
print("[1] deform_conv2d ONNX 算子已注册")


def export_onnx(model_name, resolution, out_path, dynamic=True):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = model.float().eval()
    H, W = resolution
    print(f"[2] {model_name}: FP32 CPU, size={H}x{W}")

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
    # 动态轴 (用于 ORT 对比)
    export_onnx('zhengpeng7/BiRefNet', (1024, 1024), 'birefnet_dynamic.onnx', dynamic=True)

    # 固定 1024² (用于 TRT, 规避 Swin 窗格动态 reshape)
    export_onnx('zhengpeng7/BiRefNet', (1024, 1024), 'birefnet_1024_fixed.onnx', dynamic=False)

    print("\n---")
    print("2048² 导出在 12GB 卡上 FP32 CPU 导出也 OOM，需 24GB+ 机器执行。")
    print("BiRefNet_dynamic/HR 的 birefnet.py MD5 与标准版一致，架构相同，TRT 动态形状问题通用。")
    print("换模型: 改 model_name 参数即可。")

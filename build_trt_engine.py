"""BiRefNet ONNX → TensorRT Engine 编译脚本 —— 实施步骤 3"""
import sys, os, time
sys.path.insert(0, '.')
import tensorrt as trt
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 模型配置: (ONNX文件名, Engine输出路径, 分辨率)
MODELS = {
    'birefnet': (
        os.path.join(SCRIPT_DIR, 'birefnet_1024_fixed.onnx'),
        os.path.join(SCRIPT_DIR, 'models', 'pretrained', 'birefnet_fp16_fixed.engine'),
        (1024, 1024),
    ),
    'birefnet_dynamic': (
        os.path.join(SCRIPT_DIR, 'birefnet_dynamic_1024_fixed.onnx'),
        os.path.join(SCRIPT_DIR, 'models', 'dynamic', 'birefnet_dynamic_fp16_fixed.engine'),
        (1024, 1024),
    ),
}

def build_engine(onnx_path, engine_path, resolution, fp16=True):
    H, W = resolution
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[配置] FP16: 启用")
        else:
            print("[配置] FP16: 硬件不支持，回退 FP32")

    # 解析 ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")

    # 固定形状 (避免 Swin Transformer 动态窗格 reshape 冲突)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_image",
                      (1, 3, H, W),     # min
                      (1, 3, H, W),     # opt
                      (1, 3, H, W))     # max
    config.add_optimization_profile(profile)

    print(f"[构建] 开始构建 TensorRT Engine (FP{'16' if fp16 else '32'})...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine is None:
        raise RuntimeError("Engine 构建失败 (engine is None)")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine)

    print(f"[构建] 完成: {engine_path}")
    print(f"[构建] 耗时: {elapsed:.1f}s")
    print(f"[构建] 大小: {engine.nbytes / 1e6:.1f} MB")
    print(f"[构建] 输入: {network.get_input(0).name} {network.get_input(0).shape}")
    print(f"[构建] 输出: {network.get_output(0).name} {network.get_output(0).shape}")
    return engine_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build TensorRT Engine from ONNX')
    parser.add_argument('--model', '-m', choices=list(MODELS.keys()), default='birefnet',
                        help='Which model to build')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    args = parser.parse_args()

    onnx_path, engine_path, resolution = MODELS[args.model]
    if not os.path.exists(onnx_path):
        print(f"ONNX 文件不存在: {onnx_path}")
        print("请先运行 export_onnx.py 导出 ONNX 模型")
        sys.exit(1)
    build_engine(onnx_path, engine_path, resolution, fp16=not args.fp32)

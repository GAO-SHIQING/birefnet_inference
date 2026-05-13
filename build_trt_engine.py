"""BiRefNet ONNX → TensorRT Engine 编译脚本 —— 实施步骤 3"""
import sys, os, time
sys.path.insert(0, '.')
import tensorrt as trt
import numpy as np

ONNX_PATH = "birefnet_fixed.onnx"  # 固定形状版本（规避 Swin 窗格动态 reshape）
ENGINE_PATH = os.path.join(os.path.dirname(__file__), "models", "pretrained", "birefnet_fp16_fixed.engine")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
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
                      (1, 3, 1024, 1024),     # min
                      (1, 3, 1024, 1024),     # opt
                      (1, 3, 1024, 1024))     # max
    config.add_optimization_profile(profile)

    print(f"[构建] 开始构建 TensorRT Engine (FP{'16' if fp16 else '32'})...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine is None:
        raise RuntimeError("Engine 构建失败 (engine is None)")

    with open(engine_path, 'wb') as f:
        f.write(engine)

    print(f"[构建] 完成: {engine_path}")
    print(f"[构建] 耗时: {elapsed:.1f}s")
    print(f"[构建] 大小: {engine.nbytes / 1e6:.1f} MB")
    print(f"[构建] 输入: {network.get_input(0).name} {network.get_input(0).shape}")
    print(f"[构建] 输出: {network.get_output(0).name} {network.get_output(0).shape}")
    return engine_path


if __name__ == '__main__':
    build_engine(ONNX_PATH, ENGINE_PATH, fp16=True)

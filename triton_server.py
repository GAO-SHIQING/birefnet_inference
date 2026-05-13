"""轻量 Triton 兼容 gRPC 推理服务。
实现 GRPCInferenceService 协议，支持多 context 并发。

用法:
    python triton_server.py --port 8001 --contexts-per-engine 3
"""
import os, sys, time, logging, threading
import numpy as np
import torch
from concurrent import futures
import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc, model_config_pb2

SCRIPT_DIR = os.path.dirname(__file__)
ENGINE_DIR = os.path.join(SCRIPT_DIR, 'models', 'engines')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('TritonServer')


# 默认每分辨率 context 数: 小图多用, 大图少用 (基于 48GB GPU)
DEFAULT_CONTEXTS = {768: 8, 1024: 8, 1536: 3, 2048: 2}


def parse_contexts_arg(arg: str) -> dict:
    """解析 '768=8,1024=8,1536=3,2048=2' 格式"""
    result = {}
    for kv in arg.split(','):
        k, v = kv.split('=')
        result[int(k)] = int(v)
    return result


def load_engines(contexts_map=None):
    """加载 Engine, 每个创建多个独立 execution context 以支持并发"""
    import tensorrt as trt
    if contexts_map is None:
        contexts_map = DEFAULT_CONTEXTS

    engines = {}
    if not os.path.isdir(ENGINE_DIR):
        return engines

    for f in sorted(os.listdir(ENGINE_DIR)):
        if not f.endswith('.engine'):
            continue
        try:
            size = int(f.split('_')[1])
        except (IndexError, ValueError):
            continue

        n_ctx = contexts_map.get(size, 1)
        path = os.path.join(ENGINE_DIR, f)
        with open(path, 'rb') as fh:
            engine_data = fh.read()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        name = f'birefnet_{size}'
        contexts = []
        for _ in range(n_ctx):
            ctx = engine.create_execution_context()
            d_input = torch.empty(1, 3, size, size, dtype=torch.float32, device='cuda')
            d_output = torch.empty(1, 1, size, size, dtype=torch.float32, device='cuda')
            ctx.set_input_shape('input_image', (1, 3, size, size))
            ctx.set_tensor_address('input_image', d_input.data_ptr())
            ctx.set_tensor_address('output_logits', d_output.data_ptr())
            stream = torch.cuda.Stream()
            contexts.append({
                'context': ctx,
                'd_input': d_input,
                'd_output': d_output,
                'stream': stream,
                'lock': threading.Lock(),
            })
        engines[name] = {
            'contexts': contexts,
            'size': size,
            'next_idx': 0,
        }
        logger.info(f'  已加载: {name} ({size}²), {n_ctx} contexts')

    return engines


class InferenceServicer(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, engines):
        self._engines = engines

    def _acquire_context(self, name):
        """轮询获取一个空闲 context (带负载均衡)"""
        eng = self._engines[name]
        n = len(eng['contexts'])
        for _ in range(n):
            idx = eng['next_idx']
            eng['next_idx'] = (idx + 1) % n
            ctx = eng['contexts'][idx]
            if ctx['lock'].acquire(blocking=False):
                return ctx
        # 全忙, 阻塞等待第一个
        ctx = eng['contexts'][0]
        ctx['lock'].acquire()
        return ctx

    # —— 生命周期 ——
    def ServerLive(self, request, context):
        return service_pb2.ServerLiveResponse(live=True)

    def ServerReady(self, request, context):
        return service_pb2.ServerReadyResponse(ready=True)

    def ModelReady(self, request, context):
        return service_pb2.ModelReadyResponse(ready=request.name in self._engines)

    def ServerMetadata(self, request, context):
        return service_pb2.ServerMetadataResponse(name='triton-server', version='2.0.0')

    def ModelMetadata(self, request, context):
        eng = self._engines.get(request.name)
        if eng is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return service_pb2.ModelMetadataResponse()
        size = eng['size']
        return service_pb2.ModelMetadataResponse(
            name=request.name,
            inputs=[model_config_pb2.ModelInput(
                name='input_image',
                data_type=model_config_pb2.DataType('TYPE_FP32'),
                dims=[1, 3, size, size],
            )],
            outputs=[model_config_pb2.ModelOutput(
                name='output_logits',
                data_type=model_config_pb2.DataType('TYPE_FP32'),
                dims=[1, 1, size, size],
            )],
        )

    def RepositoryIndex(self, request, context):
        models = [service_pb2.RepositoryIndexResponse.ModelIndex(
            name=n, version='1', state='READY', reason='') for n in self._engines]
        return service_pb2.RepositoryIndexResponse(models=models)

    # —— 推理 ——
    def ModelInfer(self, request, context):
        name = request.model_name
        eng = self._engines.get(name)
        if eng is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'Model {name} not found')
            return service_pb2.ModelInferResponse()

        try:
            raw = request.raw_input_contents[0]
            shape = tuple(request.inputs[0].shape)
            tensor = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f'Failed to parse input: {e}')
            return service_pb2.ModelInferResponse()

        ctx = self._acquire_context(name)
        try:
            x = torch.from_numpy(tensor).cuda()
            ctx['d_input'].copy_(x)
            ctx['context'].execute_async_v3(ctx['stream'].cuda_stream)
            ctx['stream'].synchronize()
            output = ctx['d_output'].cpu().numpy()
        finally:
            ctx['lock'].release()

        return service_pb2.ModelInferResponse(
            model_name=name,
            model_version='1',
            outputs=[service_pb2.ModelInferResponse.InferOutputTensor(
                name='output_logits', datatype='FP32', shape=list(output.shape))],
            raw_output_contents=[output.tobytes()],
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Triton-compatible gRPC inference server')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--max-workers', type=int, default=20)
    parser.add_argument('--contexts', type=str, default=None,
                        help='每分辨率 context 数, 如 "768=8,1024=8,1536=3,2048=2"')
    parser.add_argument('--contexts-per-engine', type=int, default=None,
                        help='(兼容旧版) 所有 Engine 统一 context 数')
    args = parser.parse_args()

    if args.contexts:
        contexts_map = parse_contexts_arg(args.contexts)
    elif args.contexts_per_engine:
        contexts_map = {s: args.contexts_per_engine for s in [768, 1024, 1536, 2048]}
    else:
        contexts_map = DEFAULT_CONTEXTS

    logger.info(f'加载 TRT Engines (context 分配: {contexts_map})...')
    engines = load_engines(contexts_map=contexts_map)
    if not engines:
        logger.error('没有找到任何 Engine!')
        sys.exit(1)

    max_msg = 256 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ('grpc.max_send_message_length', max_msg),
            ('grpc.max_receive_message_length', max_msg),
        ],
    )
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        InferenceServicer(engines), server
    )
    server.add_insecure_port(f'0.0.0.0:{args.port}')
    server.start()

    total_contexts = sum(len(e['contexts']) for e in engines.values())
    logger.info(f'gRPC 服务已启动: 0.0.0.0:{args.port}')
    logger.info(f'模型: {list(engines.keys())}, 并发能力: {total_contexts}')
    logger.info('按 Ctrl+C 停止')

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info('正在停止...')
        server.stop(0)


if __name__ == '__main__':
    main()

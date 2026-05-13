# BiRefNet 推理服务 — 后端接入文档

## 1. 架构

```
后端服务 ──HTTP POST──▶ API Gateway (:8080) ──gRPC──▶ TRT 推理服务 (:8001)
                             │                              │
                        预处理/后处理                  4 Engine 池 (768/1024/1536/2048)
                        · smart_resize                21 路并发推理
                        · normalize
                        · engine 选择
                        · sigmoid + 插值还原
```

后端只需调一个 HTTP 端点，无需关注模型加载、显存管理、分辨率适配。

---

## 2. 启动服务

```bash
# 终端 1: 启动推理服务 (gRPC)
python triton_server.py --port 8001

# 终端 2: 启动 API 网关 (HTTP)
python server.py
```

`server.py` 默认监听 `0.0.0.0:8080`。如需改端口，编辑末行 `uvicorn.run(app, host='0.0.0.0', port=8080)`。

### 启动后验证

```bash
curl http://<host>:8080/health
# → {"status":"ok","backend":"triton","engines":[768,1024,1536,2048]}
```

---

## 3. API 端点

### 3.1 背景去除 (最常用)

```http
POST /segment
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 图片文件 (jpg/png/webp) |
| `alpha_only` | bool | 否 | `true` 返回灰度 mask，默认 `false` 返回 RGBA |
| `resolution` | int | 否 | 指定推理分辨率 (768/1024/1536/2048)，`0` 或不传=自动选择 |

**示例:**

```bash
# 背景去除 → RGBA 透明 PNG
curl -X POST -F "image=@photo.jpg" http://<host>:8080/segment -o result.png

# 只取灰度 mask
curl -X POST -F "image=@photo.jpg" http://<host>:8080/segment?alpha_only=true -o mask.png

# 强制 2048 分辨率 (保留大图细节)
curl -X POST -F "image=@photo.jpg" "http://<host>:8080/segment?resolution=2048" -o result.png
```

**后端代码示例:**

```python
import requests

def remove_background(image_path: str, output_path: str):
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://<host>:8080/segment",
            files={"image": f},
            timeout=60,  # 大图可能较慢
        )
    resp.raise_for_status()
    with open(output_path, "wb") as out:
        out.write(resp.content)
```

```java
// Java (OkHttp)
OkHttpClient client = new OkHttpClient();
RequestBody body = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("image", "photo.jpg",
        RequestBody.create(new File("photo.jpg"), MediaType.parse("image/*")))
    .build();
Request request = new Request.Builder()
    .url("http://<host>:8080/segment")
    .post(body)
    .build();
Response response = client.newCall(request).execute();
// response.body().bytes() → PNG RGBA
```

```javascript
// Node.js (fetch / axios)
const form = new FormData();
form.append("image", fs.createReadStream("photo.jpg"));
const resp = await fetch("http://<host>:8080/segment", { method: "POST", body: form });
const buffer = await resp.arrayBuffer();
fs.writeFileSync("result.png", Buffer.from(buffer));
```

```go
// Go
file, _ := os.Open("photo.jpg")
defer file.Close()
body := &bytes.Buffer{}
writer := multipart.NewWriter(body)
part, _ := writer.CreateFormFile("image", "photo.jpg")
io.Copy(part, file)
writer.Close()
resp, _ := http.Post("http://<host>:8080/segment", writer.FormDataContentType(), body)
defer resp.Body.Close()
out, _ := os.Create("result.png")
io.Copy(out, resp.Body)
```

### 3.2 健康检查

```http
GET /health
```

```json
{
    "status": "ok",
    "backend": "triton",
    "engines": [768, 1024, 1536, 2048]
}
```

### 3.3 引擎列表

```http
GET /engines
```

```json
{
    "mode": "triton",
    "available": [768, 1024, 1536, 2048],
    "triton_url": "localhost:8001"
}
```

---

## 4. 预处理规则

后端无需关心。API Gateway 自动执行：

| 步骤 | 规则 |
|------|------|
| 缩放上限 | `max(w,h) > 3096` → 等比缩到最长边 3096 |
| 缩放下限 | `min(w,h) < 256` → 等比放到最短边 256 |
| 对齐 | 宽高对齐到 32 倍数 (LANCZOS 插值) |
| 归一化 | ImageNet 均值/标准差 |
| 引擎选择 | 取 `>= max_edge` 的最小分辨率 Engine |
| 还原 | 推理后 bilinear 插值还原原始分辨率 |

---

## 5. 返回格式

| 模式 | Content-Type | 内容 |
|------|-------------|------|
| 默认 | `image/png` | RGBA 4 通道，背景透明 |
| `alpha_only=true` | `image/png` | 灰度图，值 0-255 |

HTTP 状态码:
- `200`: 成功
- `500`: 服务异常 (显存不足、模型未加载等)
- `422`: 参数错误 (未上传 image 字段)

---

## 6. 性能参考

测试环境: RTX 5880 Ada (48GB), FP16

| 分辨率 | 单次延迟 | 并发能力 |
|--------|----------|----------|
| 768×768 | ~17ms | 8 路 |
| 1024×1024 | ~33ms | 8 路 |
| 1536×1536 | ~73ms | 3 路 |
| 2048×2048 | ~144ms | 2 路 |

端到端（含 HTTP 开销 + 预处理 + 后处理）约增加 5-15ms。

**并发建议**: 后端用连接池，并发数控制在 10-20。超时建议 60s（大图 2048 首次推理会触发 CUDA kernel 编译）。

---

## 7. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `curl: connection refused` | 服务未启动 | 检查 8080/8001 端口 |
| `500 Internal Server Error` | 推理异常 | 查看 `server.py` 日志 |
| `CUDA out of memory` | 显存不足 | 减少 `--contexts` 参数 |
| 首次请求超时 | TRT kernel 编译 | 已配置 warmup，通常 2-3s 内完成 |
| 结果异常 (全黑/全白) | 预处理或后处理异常 | 检查输入是否为 RGB 格式 |

---

## 8. 直连 Python 调用 (不走 HTTP)

如果后端是 Python，可以跳过 HTTP 直接调用推理函数，零网络开销：

```python
import sys
sys.path.insert(0, '/path/to/birefnet_inference')
from infer_multi import remove_background
from PIL import Image

img = Image.open("photo.jpg")
result = remove_background(img)     # RGBA PIL Image
result.save("result.png")
```

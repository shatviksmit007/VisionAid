import os
import time
import numpy as np
import onnxruntime as ort
import torch

ort.preload_dlls(directory='')

base = os.path.expanduser('~/.paddlex/official_models')
dp = os.path.join(base, 'PP-OCRv6_tiny_det_onnx', 'inference.onnx')
rp = os.path.join(base, 'PP-OCRv6_tiny_rec_onnx', 'inference.onnx')

det = ort.InferenceSession(dp, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
rec = ort.InferenceSession(rp, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

dn = det.get_inputs()[0].name
rn = rec.get_inputs()[0].name

dx = np.zeros((1,3,640,640), np.float32)
rx = np.zeros((23,3,48,154), np.float32)

print('DET:', det.get_providers())
print('REC:', rec.get_providers())

for _ in range(30):
    det.run(None, {dn: dx})
    torch.cuda.synchronize()
    rec.run(None, {rn: rx})
    torch.cuda.synchronize()

det_t = []
rec_t = []
total_t = []

for _ in range(100):
    t0 = time.perf_counter()

    det.run(None, {dn: dx})
    torch.cuda.synchronize()

    t1 = time.perf_counter()

    rec.run(None, {rn: rx})
    torch.cuda.synchronize()

    t2 = time.perf_counter()

    det_t.append((t1-t0)*1000)
    rec_t.append((t2-t1)*1000)
    total_t.append((t2-t0)*1000)

def report(name, x):
    print(
        f'{name}: '
        f'P50={np.percentile(x,50):.3f} ms  '
        f'P95={np.percentile(x,95):.3f} ms  '
        f'P99={np.percentile(x,99):.3f} ms  '
        f'Mean={np.mean(x):.3f} ms  '
        f'Max={np.max(x):.3f} ms'
    )

print()
report('DET', det_t)
report('REC', rec_t)
report('TOTAL', total_t)

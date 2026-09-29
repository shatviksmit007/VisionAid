import os,time,statistics,cv2,numpy as np,onnxruntime as ort,yaml

ort.preload_dlls(directory='')

BASE=os.path.expanduser('~/.paddlex/official_models')
DET=os.path.join(BASE,'PP-OCRv6_tiny_det_onnx','inference.onnx')
REC=os.path.join(BASE,'PP-OCRv6_tiny_rec_onnx','inference.onnx')
META=os.path.join(BASE,'PP-OCRv6_tiny_rec_onnx','inference.yml')
IMAGE='/home/shatviksmit/VisionAid/ultralytics/coco100/val2017/000000187734.jpg'

providers=['CUDAExecutionProvider','CPUExecutionProvider']
det=ort.InferenceSession(DET,providers=providers)
rec=ort.InferenceSession(REC,providers=providers)

print('DET providers:',det.get_providers())
print('REC providers:',rec.get_providers())
print('DET input:',det.get_inputs()[0].name,det.get_inputs()[0].shape)
print('REC input:',rec.get_inputs()[0].name,rec.get_inputs()[0].shape)

img=cv2.imread(IMAGE)
if img is None: raise RuntimeError(IMAGE)

# ------------------------------------------------------------
# Recognition preprocessing
# ------------------------------------------------------------
def rec_preprocess(crop):
    h,w=crop.shape[:2]
    new_h=48
    new_w=max(8,int(round(w*new_h/h)))
    crop=cv2.resize(crop,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    crop=crop.astype(np.float32)/255.0
    crop=(crop-0.5)/0.5
    crop=crop.transpose(2,0,1)
    return crop

# The exact model metadata says RecResizeImg -> [3,48,320].
# For the benchmark we use a representative 48x154 crop,
# matching the standalone test already measured.
crop=img[100:148,100:254]
rec_input=rec_preprocess(crop)
rec_input=rec_input[None].astype(np.float32)

# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------
print('\nWarmup...')
for _ in range(30):
    det.run(None,{det.get_inputs()[0].name:np.zeros((1,3,640,640),dtype=np.float32)})
    rec.run(None,{rec.get_inputs()[0].name:rec_input})


def stats(xs):
    xs=sorted(xs)
    def q(p): return xs[min(len(xs)-1,int(len(xs)*p/100))]
    return {
        'min':xs[0],
        'p50':q(50),
        'p95':q(95),
        'p99':q(99),
        'mean':statistics.mean(xs),
        'max':xs[-1]
    }

def show(name,xs):
    s=stats(xs)
    print('\n'+name)
    print(f"Min : {s['min']:.3f} ms")
    print(f"P50 : {s['p50']:.3f} ms")
    print(f"P95 : {s['p95']:.3f} ms")
    print(f"P99 : {s['p99']:.3f} ms")
    print(f"Mean: {s['mean']:.3f} ms")
    print(f"Max : {s['max']:.3f} ms")
    print(f"FPS : {1000/s['mean']:.2f}")

N=500

det_x=[]
rec_x=[]
pre_x=[]
rec_total_x=[]

print(f'\nBenchmarking {N} iterations...')

for _ in range(N):
    t=time.perf_counter()
    det.run(None,{det.get_inputs()[0].name:np.zeros((1,3,640,640),dtype=np.float32)})
    det_x.append((time.perf_counter()-t)*1000)

for _ in range(N):
    t=time.perf_counter()
    x=rec_preprocess(crop)
    pre_x.append((time.perf_counter()-t)*1000)
    x=x[None].astype(np.float32)
    t2=time.perf_counter()
    rec.run(None,{rec.get_inputs()[0].name:x})
    rec_x.append((time.perf_counter()-t2)*1000)
    rec_total_x.append((time.perf_counter()-t)*1000)

show('PP-OCRv6 Tiny DET — raw ONNX CUDA',det_x)
show('Recognition preprocessing',pre_x)
show('PP-OCRv6 Tiny REC — raw ONNX CUDA',rec_x)
show('REC preprocess + ONNX',rec_total_x)

print('\nRepresentative recognition input:',rec_input.shape)
print('Image:',img.shape)

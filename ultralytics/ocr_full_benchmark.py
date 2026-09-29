import os,time,glob,statistics,cv2,numpy as np,onnxruntime as ort

ort.preload_dlls(directory='')

BASE=os.path.expanduser('~/.paddlex/official_models')
DET=os.path.join(BASE,'PP-OCRv6_tiny_det_onnx','inference.onnx')
REC=os.path.join(BASE,'PP-OCRv6_tiny_rec_onnx','inference.onnx')
IMGDIR='/home/shatviksmit/VisionAid/ultralytics/coco100/val2017'

det=ort.InferenceSession(DET,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
rec=ort.InferenceSession(REC,providers=['CUDAExecutionProvider','CPUExecutionProvider'])

det_name=det.get_inputs()[0].name
rec_name=rec.get_inputs()[0].name

print('DET providers:',det.get_providers())
print('REC providers:',rec.get_providers())
print('DET input:',det.get_inputs()[0].shape)
print('REC input:',rec.get_inputs()[0].shape)

imgs=sorted(glob.glob(os.path.join(IMGDIR,'*.jpg')))[:100]
print('Images:',len(imgs))

# Warmup
det_x=np.zeros((1,3,640,640),np.float32)
rec_x=np.zeros((23,3,48,154),np.float32)

for _ in range(30):
    det.run(None,{det_name:det_x})
    rec.run(None,{rec_name:rec_x})

print('Warmup complete')

DET_T=[]
DET_PRE=[]
CROP_T=[]
REC_T=[]
TOTAL_T=[]
BOXES=[]

def pct(a,p):
    return float(np.percentile(a,p))

def report(name,a):
    print(
        f'{name:25s} '
        f'P50={pct(a,50):8.3f} ms  '
        f'P95={pct(a,95):8.3f} ms  '
        f'P99={pct(a,99):8.3f} ms  '
        f'Mean={statistics.mean(a):8.3f} ms  '
        f'Max={max(a):8.3f} ms'
    )

for i,path in enumerate(imgs,1):

    img=cv2.imread(path)
    if img is None:
        continue

    total_start=time.perf_counter()

    # -------------------------
    # DETECTOR PREPROCESS
    # -------------------------
    t=time.perf_counter()

    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resized=cv2.resize(rgb,(640,640),interpolation=cv2.INTER_LINEAR)
    arr=resized.astype(np.float32)/255.0
    arr=np.transpose(arr,(2,0,1))[None]

    DET_PRE.append((time.perf_counter()-t)*1000)

    # -------------------------
    # DETECTOR
    # -------------------------
    t=time.perf_counter()

    det.run(None,{det_name:arr})

    DET_T.append((time.perf_counter()-t)*1000)

    # -------------------------
    # REPRESENTATIVE REAL CROPS
    # -------------------------
    t=time.perf_counter()

    h,w=img.shape[:2]
    boxes=[]

    # Generate representative text-line sized regions
    # from the REAL image.
    for yy in range(0,max(1,h-48),max(48,h//6)):
        for xx in range(0,max(1,w-154),max(154,w//4)):
            x2=min(w,xx+154)
            y2=min(h,yy+48)

            if x2-xx >= 20 and y2-yy >= 10:
                boxes.append((xx,yy,x2,y2))

    boxes=boxes[:23]

    crops=[]

    for x1,y1,x2,y2 in boxes:

        crop=img[y1:y2,x1:x2]

        if crop.size==0:
            continue

        crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        crop=cv2.resize(crop,(154,48),interpolation=cv2.INTER_LINEAR)

        crop=crop.astype(np.float32)/255.0
        crop=np.transpose(crop,(2,0,1))

        crops.append(crop)

    if crops:
        batch=np.stack(crops,axis=0).astype(np.float32)
    else:
        batch=np.zeros((1,3,48,154),np.float32)

    CROP_T.append((time.perf_counter()-t)*1000)

    # -------------------------
    # BATCHED RECOGNITION
    # -------------------------
    t=time.perf_counter()

    rec.run(None,{rec_name:batch})

    REC_T.append((time.perf_counter()-t)*1000)

    TOTAL_T.append((time.perf_counter()-total_start)*1000)
    BOXES.append(len(boxes))

    if i%10==0:
        print(
            f'[{i:3d}/{len(imgs)}] '
            f'boxes={len(boxes):2d} '
            f'DET={DET_T[-1]:.2f} ms '
            f'REC={REC_T[-1]:.2f} ms '
            f'TOTAL={TOTAL_T[-1]:.2f} ms'
        )

print()
print('==============================================')
print(' PP-OCRv6 TINY — REAL IMAGE PIPELINE')
print('==============================================')

report('Detector preprocess',DET_PRE)
report('DET ONNX',DET_T)
report('Crop + preprocess',CROP_T)
report('REC ONNX',REC_T)
report('TOTAL',TOTAL_T)

print()
print(f'Boxes/image: mean={statistics.mean(BOXES):.1f} min={min(BOXES)} max={max(BOXES)}')
print(f'Total FPS from P50: {1000/pct(TOTAL_T,50):.2f}')
print(f'Total FPS from P99: {1000/pct(TOTAL_T,99):.2f}')

print()
print('NOTE:')
print('This benchmark uses real COCO images and real batched REC inputs.')
print('It does NOT reproduce PaddleOCR DB detector postprocessing.')
print('The raw DET ONNX execution is measured directly.')

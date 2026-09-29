import os
import time
import threading
import glob
import cv2
import numpy as np
import statistics
import onnxruntime as ort
import torch

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

IMGDIR = '/home/shatviksmit/VisionAid/ultralytics/coco100/val2017'
INPUT_FPS = 30.0
RUN_SECONDS = 60.0
OCR_INTERVAL = 1.0

BASE = os.path.expanduser('~/.paddlex/official_models')
DET_PATH = os.path.join(BASE, 'PP-OCRv6_tiny_det_onnx', 'inference.onnx')
REC_PATH = os.path.join(BASE, 'PP-OCRv6_tiny_rec_onnx', 'inference.onnx')

# ------------------------------------------------------------
# ORT / MODELS
# ------------------------------------------------------------

ort.preload_dlls(directory='')

ocr_det = ort.InferenceSession(
    DET_PATH,
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
)

ocr_rec = ort.InferenceSession(
    REC_PATH,
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
)

det_name = ocr_det.get_inputs()[0].name
rec_name = ocr_rec.get_inputs()[0].name

print('OCR DET:', ocr_det.get_providers())
print('OCR REC:', ocr_rec.get_providers())

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

images = sorted(glob.glob(os.path.join(IMGDIR, '*.jpg')))

if not images:
    raise RuntimeError('No COCO images found')

print('Images:', len(images))

# ------------------------------------------------------------
# SHARED STATE
# ------------------------------------------------------------

lock = threading.Lock()

state = {
    'frame': None,
    'frame_id': 0,
    'frame_ts': 0.0,

    'yolo_ts': 0.0,
    'yolo_frame_id': 0,

    'depth_ts': 0.0,
    'depth_frame_id': 0,

    'fusion_ts': 0.0,

    'stop': False,

    'ocr_running': False,
    'ocr_last_frame_id': 0,
    'ocr_last_ts': 0.0,
}

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

camera_to_yolo = []
yolo_to_fusion = []
camera_to_fusion = []
depth_age = []

ocr_total = []
ocr_queue_age = []
ocr_count = []

# ------------------------------------------------------------
# YOLO SIMULATION
# ------------------------------------------------------------

# We intentionally use the same measured YOLO workload from your
# existing safety benchmark rather than pretending OCR is the only
# GPU workload. This performs a representative CUDA workload.

yolo_path = '/home/shatviksmit/VisionAid/ultralytics/yolo11n.pt'

try:
    from ultralytics import YOLO
    yolo = YOLO(yolo_path)
    print('YOLO loaded')
except Exception as e:
    print('YOLO load failed:', e)
    yolo = None

# ------------------------------------------------------------
# DEPTH SIMULATION
# ------------------------------------------------------------

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(
        'depth-anything/depth-anything-v2-small-hf'
    )

    depth_model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/depth-anything-v2-small-hf'
    ).cuda().half()

    depth_model.eval()

    print('Depth loaded')

except Exception as e:
    print('Depth load failed:', e)
    processor = None
    depth_model = None

# ------------------------------------------------------------
# WARMUP
# ------------------------------------------------------------

dummy_det = np.zeros((1,3,640,640), np.float32)
dummy_rec = np.zeros((23,3,48,154), np.float32)

for _ in range(30):
    ocr_det.run(None, {det_name: dummy_det})
    torch.cuda.synchronize()

    ocr_rec.run(None, {rec_name: dummy_rec})
    torch.cuda.synchronize()

if yolo is not None:
    dummy = np.zeros((480,640,3), np.uint8)

    for _ in range(10):
        yolo.predict(
            dummy,
            imgsz=640,
            conf=0.25,
            verbose=False,
            device=0
        )

if depth_model is not None:
    dummy = np.zeros((480,640,3), np.uint8)

    inputs = processor(
        images=dummy,
        return_tensors='pt'
    )

    inputs = {
        k: v.cuda()
        for k,v in inputs.items()
        if torch.is_tensor(v)
    }

    with torch.inference_mode():
        for _ in range(5):
            depth_model(**inputs)

    torch.cuda.synchronize()

print('Warmup complete')

# ------------------------------------------------------------
# YOLO WORKER
# ------------------------------------------------------------

def yolo_worker():

    while True:

        with lock:
            if state['stop']:
                return

            frame = state['frame']
            fid = state['frame_id']
            fts = state['frame_ts']

        if frame is None:
            time.sleep(0.001)
            continue

        start = time.perf_counter()

        if yolo is not None:
            yolo.predict(
                frame,
                imgsz=640,
                conf=0.25,
                verbose=False,
                device=0
            )

        torch.cuda.synchronize()

        end = time.perf_counter()

        with lock:
            state['yolo_ts'] = end
            state['yolo_frame_id'] = fid

            camera_to_yolo.append(
                (end - fts) * 1000
            )

        time.sleep(0.0005)

# ------------------------------------------------------------
# DEPTH WORKER
# ------------------------------------------------------------

def depth_worker():

    period = 1.0 / 12.0
    next_time = time.perf_counter()

    while True:

        next_time += period

        with lock:
            if state['stop']:
                return

            frame = state['frame']
            fid = state['frame_id']
            fts = state['frame_ts']

        if frame is not None and depth_model is not None:

            inputs = processor(
                images=frame,
                return_tensors='pt'
            )

            inputs = {
                k: v.cuda()
                for k,v in inputs.items()
                if torch.is_tensor(v)
            }

            with torch.inference_mode():
                depth_model(**inputs)

            torch.cuda.synchronize()

            now = time.perf_counter()

            with lock:
                state['depth_ts'] = now
                state['depth_frame_id'] = fid

        sleep = next_time - time.perf_counter()

        if sleep > 0:
            time.sleep(sleep)

# ------------------------------------------------------------
# OCR WORKER
# ------------------------------------------------------------

def ocr_worker():

    while True:

        time.sleep(OCR_INTERVAL)

        with lock:

            if state['stop']:
                return

            if state['frame'] is None:
                continue

            if state['ocr_running']:
                continue

            frame = state['frame'].copy()
            fid = state['frame_id']
            frame_ts = state['frame_ts']

            state['ocr_running'] = True

        start = time.perf_counter()

        try:

            # ------------------------------------------------
            # OCR DET PREPROCESS
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            resized = cv2.resize(
                rgb,
                (640,640),
                interpolation=cv2.INTER_LINEAR
            )

            arr = resized.astype(
                np.float32
            ) / 255.0

            arr = np.transpose(
                arr,
                (2,0,1)
            )[None]

            # ------------------------------------------------
            # OCR DET
            # ------------------------------------------------

            ocr_det.run(
                None,
                {det_name: arr}
            )

            torch.cuda.synchronize()

            # ------------------------------------------------
            # REPRESENTATIVE TEXT REGIONS
            # ------------------------------------------------

            h,w = frame.shape[:2]

            crops = []

            for yy in range(
                0,
                max(1,h-48),
                max(48,h//6)
            ):

                for xx in range(
                    0,
                    max(1,w-154),
                    max(154,w//4)
                ):

                    x2 = min(w,xx+154)
                    y2 = min(h,yy+48)

                    crop = frame[
                        yy:y2,
                        xx:x2
                    ]

                    if crop.size == 0:
                        continue

                    crop = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2RGB
                    )

                    crop = cv2.resize(
                        crop,
                        (154,48),
                        interpolation=cv2.INTER_LINEAR
                    )

                    crop = crop.astype(
                        np.float32
                    ) / 255.0

                    crop = np.transpose(
                        crop,
                        (2,0,1)
                    )

                    crops.append(crop)

                    if len(crops) >= 23:
                        break

                if len(crops) >= 23:
                    break

            if crops:
                batch = np.stack(
                    crops,
                    axis=0
                ).astype(np.float32)
            else:
                batch = np.zeros(
                    (1,3,48,154),
                    np.float32
                )

            # ------------------------------------------------
            # BATCH REC
            # ------------------------------------------------

            ocr_rec.run(
                None,
                {rec_name: batch}
            )

            torch.cuda.synchronize()

            end = time.perf_counter()

            with lock:

                ocr_total.append(
                    (end-start)*1000
                )

                ocr_queue_age.append(
                    (start-frame_ts)*1000
                )

                ocr_count.append(
                    len(crops)
                )

                state['ocr_last_frame_id'] = fid
                state['ocr_last_ts'] = end

        finally:

            with lock:
                state['ocr_running'] = False

# ------------------------------------------------------------
# FUSION LOOP
# ------------------------------------------------------------

def fusion_loop():

    last_fusion = -1

    while True:

        with lock:

            if state['stop']:
                return

            fid = state['frame_id']
            yfid = state['yolo_frame_id']
            yts = state['yolo_ts']
            fts = state['frame_ts']
            dts = state['depth_ts']

        if yfid != last_fusion and yts > 0:

            now = time.perf_counter()

            with lock:

                state['fusion_ts'] = now

                camera_to_fusion.append(
                    (now-fts)*1000
                )

                yolo_to_fusion.append(
                    (now-yts)*1000
                )

                if dts > 0:
                    depth_age.append(
                        (now-dts)*1000
                    )

            last_fusion = yfid

        time.sleep(0.001)

# ------------------------------------------------------------
# START WORKERS
# ------------------------------------------------------------

threads = [
    threading.Thread(target=yolo_worker,daemon=True),
    threading.Thread(target=depth_worker,daemon=True),
    threading.Thread(target=ocr_worker,daemon=True),
    threading.Thread(target=fusion_loop,daemon=True),
]

for t in threads:
    t.start()

print()
print('==============================================')
print(' ASYNC OCR + YOLO + DEPTH BENCHMARK')
print('==============================================')
print(f'Virtual camera: {INPUT_FPS:.1f} FPS')
print(f'Runtime: {RUN_SECONDS:.0f} seconds')
print(f'OCR interval: {OCR_INTERVAL:.1f} seconds')
print()

# ------------------------------------------------------------
# VIRTUAL CAMERA
# ------------------------------------------------------------

start = time.perf_counter()
next_frame = start
fid = 0

while time.perf_counter()-start < RUN_SECONDS:

    next_frame += 1.0 / INPUT_FPS

    path = images[fid % len(images)]

    frame = cv2.imread(path)

    ts = time.perf_counter()

    with lock:
        state['frame'] = frame
        state['frame_id'] = fid
        state['frame_ts'] = ts

    fid += 1

    sleep = next_frame-time.perf_counter()

    if sleep > 0:
        time.sleep(sleep)

# ------------------------------------------------------------
# STOP
# ------------------------------------------------------------

with lock:
    state['stop'] = True

for t in threads:
    t.join(timeout=3)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

def report(name,x):

    if not x:
        print(name, ': NO DATA')
        return

    print(
        f'{name:25s} '
        f'P50={np.percentile(x,50):8.3f} ms  '
        f'P95={np.percentile(x,95):8.3f} ms  '
        f'P99={np.percentile(x,99):8.3f} ms  '
        f'Mean={np.mean(x):8.3f} ms  '
        f'Max={np.max(x):8.3f} ms'
    )

print()
print('==============================================')
print(' RESULTS')
print('==============================================')

report('Camera -> YOLO',camera_to_yolo)
report('YOLO -> Fusion',yolo_to_fusion)
report('Camera -> Fusion',camera_to_fusion)
report('Depth age',depth_age)

print()

report('OCR total',ocr_total)
report('OCR queue age',ocr_queue_age)

print()
print('Frames generated:',fid)
print('OCR jobs:',len(ocr_total))

if ocr_count:
    print(
        'OCR regions/job:',
        f'P50={np.percentile(ocr_count,50):.1f}',
        f'P99={np.percentile(ocr_count,99):.1f}'
    )

print()
print('This benchmark measures asynchronous OCR interference with')
print('the YOLO + Depth safety loop. OCR never blocks fusion.')

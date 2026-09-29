import time
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Force model onto GPU
model.to("cuda")

# Dummy input: batch=1, channels=3, 640x640
x = torch.randn(
    1, 3, 640, 640,
    device="cuda"
)

# Warmup
print("Warming up...")

for _ in range(100):
    model.model(x)

torch.cuda.synchronize()

# Benchmark
iterations = 1000
times = []

print("Benchmarking...")

for _ in range(iterations):

    torch.cuda.synchronize()
    start = time.perf_counter()

    model.model(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

times.sort()

def percentile(data, p):
    index = int(len(data) * p / 100)
    return data[min(index, len(data) - 1)]

print("\n========== RAW YOLO11n ==========")

print(f"Min : {min(times):.3f} ms")
print(f"P50 : {percentile(times, 50):.3f} ms")
print(f"P95 : {percentile(times, 95):.3f} ms")
print(f"P99 : {percentile(times, 99):.3f} ms")
print(f"Max : {max(times):.3f} ms")

print(f"\nApprox FPS: {1000 / percentile(times, 50):.1f}")

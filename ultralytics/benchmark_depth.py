import time
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

device = torch.device("cuda")

print("Loading model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL_NAME)


# --------------------------------------------------
# Test image
# --------------------------------------------------

image = Image.open("bus.jpg").convert("RGB")

inputs = processor(images=image, return_tensors="pt")

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


# --------------------------------------------------
# Warmup
# --------------------------------------------------

print("\nWarming up...")

with torch.inference_mode():

    for _ in range(30):
        model(**inputs)

torch.cuda.synchronize()


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

iterations = 200
times = []

print(f"Benchmarking {iterations} iterations...")

with torch.inference_mode():

    for _ in range(iterations):

        torch.cuda.synchronize()

        start = time.perf_counter()

        outputs = model(**inputs)

        torch.cuda.synchronize()

        end = time.perf_counter()

        times.append((end - start) * 1000)


times.sort()


def percentile(data, p):

    index = int(len(data) * p / 100)

    return data[min(index, len(data) - 1)]


print("\n========== DEPTH ANYTHING V2 SMALL ==========")

print(f"Input image: {image.size}")

print(f"\nMin : {min(times):.3f} ms")
print(f"P50 : {percentile(times, 50):.3f} ms")
print(f"P95 : {percentile(times, 95):.3f} ms")
print(f"P99 : {percentile(times, 99):.3f} ms")
print(f"Max : {max(times):.3f} ms")

print(
    f"\nApprox FPS from P50: "
    f"{1000 / percentile(times, 50):.1f}"
)

print(
    f"Approx FPS from P99: "
    f"{1000 / percentile(times, 99):.1f}"
)

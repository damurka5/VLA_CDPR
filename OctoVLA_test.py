#!/usr/bin/env python3
import time
import jax
import jax.numpy as jnp
import numpy as np
from octo.model.octo_model import OctoModel
from PIL import Image

# Configuration
IMAGE_PATH = "data/put_eggplant_into_pot--clutter.png"
PROMPT = "put eggplant into pot"
MODEL_NAME = "hf://rail-berkeley/octo-small-1.5"
NUM_WARMUP = 10
NUM_ITERATIONS = 100

def main():
    # Load model and prepare inputs
    print("Loading model...")
    model = OctoModel.load_pretrained(MODEL_NAME)
    
    # Load image and create observation
    img = np.array(Image.open(IMAGE_PATH).resize((256, 256)))
    img = img[np.newaxis, np.newaxis, ...]  # add batch + time dims
    observation = {
        "image_primary": img,
        "timestep_pad_mask": np.array([[True]])
    }
    task = model.create_tasks(texts=[PROMPT])
    rng = jax.random.PRNGKey(0)
    
    # Warmup runs
    print(f"Running {NUM_WARMUP} warmup iterations...")
    for _ in range(NUM_WARMUP):
        _ = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
            rng=rng
        )
    
    # Benchmark runs
    print(f"Running {NUM_ITERATIONS} benchmark iterations...")
    timings = []
    for _ in range(NUM_ITERATIONS):
        start_time = time.perf_counter()
        
        _ = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
            rng=rng
        )
        
        # Block until computation is done (JAX is async by default)
        jax.block_until_ready(_)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    # Calculate statistics
    timings = np.array(timings)
    avg_time = np.mean(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    std_dev = np.std(timings)
    fps = 1 / avg_time
    
    print("\nBenchmark Results:")
    print(f"- Average inference time: {avg_time:.4f} ± {std_dev:.4f} s")
    print(f"- Best case: {min_time:.4f} s")
    print(f"- Worst case: {max_time:.4f} s")
    print(f"- Throughput: {fps:.2f} FPS")
    
    # JAX memory stats (if using GPU)
    try:
        from jax.lib import xla_bridge
        if xla_bridge.get_backend().platform == 'gpu':
            from jax.lib import xla_client
            mem_stats = xla_client.get_memory_stats()
            peak_mem = mem_stats['peak_bytes'] / (1024 ** 2)  # MB
            print(f"- Peak GPU memory used: {peak_mem:.2f} MB")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
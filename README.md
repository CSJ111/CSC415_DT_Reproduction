

# ⚡ Optimized Decision Transformer (Single-Notebook A100)

### *A High-Performance, Single-File Reproduction of Decision Transformer*

This notebook contains a highly optimized, self-contained implementation of the **Decision Transformer** (Chen et al., 2021). It is designed to run on high-end GPUs (A100/H100) directly within a Jupyter/Colab environment.

By leveraging **GPU-resident datasets**, **Torch Compile (JIT)**, and **Automatic Mixed Precision (AMP)**, this notebook trains a model to **~80.6 normalized score** on Hopper-Medium in under **3 minutes**.

## 🏆 Results

| Environment | Dataset | Paper Baseline | **This Notebook** |
| --- | --- | --- | --- |
| **Hopper** | Medium | 67.6 ± 1.0 | **80.6 ± 25.9** |

*Note: The high variance () reflects the model's ability to "stitch" suboptimal trajectories into expert ones ( score) while occasionally falling back to the dataset's average failure modes ( score).*

## 🚀 Key Optimizations

* **⚡ Zero-Copy Data Loading:** The entire dataset is loaded into GPU VRAM at startup. The `__getitem__` operation is a simple tensor slice, eliminating CPU bottlenecks entirely.
* **🔥 Torch 2.0 Compilation:** Uses `torch.compile()` to fuse Transformer kernels, reducing overhead by ~30%.
* **🧠 Automatic Mixed Precision (AMP):** Full BF16/FP16 support for maximum throughput on NVIDIA Ampere+ architectures.
* **🛠️ Auto-Repair Logic:** Automatically detects and fixes broken episode boundaries in D4RL `v0` datasets (e.g., `hopper_medium.hdf5`) without needing external libraries.

## 📦 Requirements

This notebook is designed for:

* **Hardware:** NVIDIA A100 or H100 (High VRAM required for large batches). You can also decrease the batch size for smaller GPUs, but expect longer training times.
* **Environment:** Google Colab Pro, Kaggle Kernels, or a local Jupyter Lab instance.

### Dependencies

Run the first cell to install the necessary libraries:

```python
!pip install gymnasium[mujoco] mujoco h5py numpy torch

```

## 📖 Notebook Structure

The notebook is divided into 4 logical cells for modularity:

1. **Configuration (Cell 1):**
* Sets up the **Massive Batch Size (4096)**.
* Configures the A100-specific hyperparameters (`torch.set_float32_matmul_precision('high')`).


2. **Dataset & Model (Cell 2):**
* `GPUDataset`: A custom class that loads the HDF5 file directly onto the GPU.
* `DecisionTransformer`: The GPT-style architecture with interleaved embeddings ().


3. **Training Loop (Cell 3):**
* `train_lightspeed`: The optimized loop using `torch.amp.autocast` and `torch.compile`.
* Saves the best model to `best_dt_a100.pt`.


4. **Final Evaluation (Cell 4):**
* Loads the best checkpoint.
* Runs 20 evaluation episodes to calculate the final robust score.



## 🏎️ How to Run

1. **Open the Notebook:** Load `DecisionTransformer_A100.ipynb` in your environment.
2. **Run All Cells:** Execute cells 1 through 4 sequentially.
3. **Wait for Compilation:** The first training step will take ~30 seconds as PyTorch JIT compiles the kernels.
4. **Watch it Fly:** Subsequent steps will run at ~10-20 steps/second. Training completes in < 3 minutes.

## 📊 Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `batch_size` | **4096** | Massive batch size to saturate A100 cores. |
| `embed_dim` | **512** | Increased capacity (vs 128 in paper) for better stitching. |
| `layers` | 4 | Deeper network for complex temporal dependencies. |
| `context_len` | 20 | Standard context window. |
| `steps` | 2,500 | Equivalent to ~150k steps at batch size 64. |

## 📜 Citation

If you use this code, please cite the original paper:

```bibtex
@article{chen2021decision,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  journal={arXiv preprint arXiv:2106.01345},
  year={2021}
}

```
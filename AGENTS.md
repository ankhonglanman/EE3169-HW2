# AGENTS.md

Guidance for coding agents working in `EE3169-HW2` (HCMUT EE3169 Homework 2 — Fully Connected Neural Networks in NumPy & TensorFlow).

## Project Overview

This repository implements multi-layer fully connected neural networks (`[100, 100, 100]` hidden units + Softmax output) for two classification tasks described in `assignment2.pdf`:
1. **Bat dataset (`data/bat.dat`)**: 3-class 2D synthetic classification (`train_x`: `(12000, 2)`, `test_x`: `(3000, 2)`).
2. **Fashion MNIST (`data/fashion-mnist/*.gz`)**: 10-class `28x28` image classification (`train_x`: `(50000, 784)`, `val_x`: `(10000, 784)`, `test_x`: `(10000, 784)` with `sampling_step=1`).

## Repository Structure

| File | Purpose | Edit Policy |
| :--- | :--- | :--- |
| `assignment2.pdf` | Official Vietnamese assignment spec and grading rubric (`[TODO 1.1]`–`[TODO 1.7]`). | Read-only reference. |
| `util.py` | Data loaders (`get_bat_data`, `get_mnist_data`), preprocessing (`normalize`, `create_one_hot`, `add_one`), plotting (`visualize_point`, `plot_loss`). | **Do not modify** (per assignment instructions). |
| `gradient_check.py` | Numerical differentiation helpers (`eval_numerical_gradient`, `rel_error`). | **Do not break signatures**; keep read-only unless explicitly asked. |
| `activation_np.py` | `[TODO 1.1]` Activations (`sigmoid`, `reLU`, `tanh`, `softmax`, `softmax_minus_max`) and gradients (`sigmoid_grad`, `reLU_grad`, `tanh_grad`). | Editable implementation file. |
| `dnn_np.py` | `[TODO 1.2]`–`[TODO 1.6]` NumPy `Config`, `Layer`, `NeuralNet`, `minibatch_train`, `batch_train`, `unit_test_layer`, `test`, `bat_classification`, `mnist_classification`. | Editable implementation file. |
| `dnn_tf.py` | `[TODO 1.7]` TensorFlow Classifier for Bat & Fashion MNIST (`bat_classification`, `mnist_classification`, aliases `problem1`, `problem2`), plus TF 2.16+ Estimator compatibility shim. | Editable implementation file. |
| `dnn_walkthrough.ipynb` | Step-by-step interactive walkthrough notebook explaining `dnn_np.py` and `dnn_tf.py`. | Documentation / learning artifact. |
| `data/` | Binary/compressed datasets (`bat.dat`, `fashion-mnist/*.gz`). | Read-only data. |

## Environment & Execution

- **Preferred Python / Conda Environment**:
  - `/home/an-vu/miniconda3/envs/ml_env_linux/bin/python` (Python 3.14, TensorFlow 2.22 with CUDA, NumPy, Matplotlib, Jupyter/nbformat).
  - System `/usr/bin/python3` has NumPy and Matplotlib, **not** TensorFlow.
- **Running the scripts**:
  ```bash
  # NumPy implementation (numerical gradient check + Bat + Fashion MNIST)
  /home/an-vu/miniconda3/envs/ml_env_linux/bin/python dnn_np.py

  # TensorFlow implementation (Bat + Fashion MNIST)
  /home/an-vu/miniconda3/envs/ml_env_linux/bin/python dnn_tf.py
  ```
- **Headless / non-GUI testing**:
  - Set `MPLBACKEND=Agg` (and optionally `MPLCONFIGDIR=/tmp/matplotlib`) when running automated checks without a display.
  - `dnn_np.py` automatically switches `qt*` Matplotlib backends (`PySide6`) to `TkAgg` at startup to avoid Qt C++ event-loop segfaults during `plt.pause(0.01)` redraw loops.

## Architecture & Mathematical Conventions

1. **Matrix Shapes & Bias Handling**:
   - Inputs have shape `(N, D_in)`, weights `Layer.w` have shape `(D_in, D_out)`, pre-activations `z = np.dot(x, self.w)` have shape `(N, D_out)`.
   - `dnn_np.py` uses the **bias trick** via `add_one()` (appending a `1.0` column to `train_x`, `val_x`, `test_x`).
   - `dnn_tf.py` does **not** call `add_one()` because `DNNClassifier` / `tf.keras.layers.Dense` maintains explicit bias vectors.
2. **Loss & L2 Regularization**:
   - Data loss is mean categorical cross-entropy over `m = y.shape[0]` samples: $-\frac{1}{m}\sum y_{ik} \log \hat{y}_{ik}$ (probabilities clipped by `1e-12`).
   - L2 penalty in `NeuralNet.compute_loss` is $0.5 \cdot \sum_l \text{reg}_l \|W^{(l)}\|_F^2$.
   - Regularization gradient term added in `Layer.backward` and `NeuralNet.backward` (`grad_last`) is $+\text{reg}_l \cdot W^{(l)}$.
   - Output softmax+cross-entropy delta is $\delta^{(L)} = (\hat{y} - y) / m$.
3. **TensorFlow Compatibility (`dnn_tf.py`)**:
   - Native `tf.estimator` was removed in TensorFlow 2.16+.
   - `dnn_tf.py` aliases `tf = tf.compat.v1` and installs a Keras-backed shim (`_EstimatorInputs`, `_DNNClassifier`, `_EstimatorModule`) **only when** `tf.estimator.DNNClassifier` is absent, keeping submission compatibility with older TF 1.x / <= 2.15 grading environments.

## Agent Guidelines & Guardrails

- **Preserve `[TODO ...]` comments**: Never delete `# [TODO 1.1]` through `# [TODO 1.7]` or `# TODO: [YC1.7]` markers; grading scripts and instructors may scan for them.
- **Avoid blocking interactive prompts**: Keep `pdb.set_trace()` guarded by `sys.stdin.isatty()` (as in `dnn_np.py`) and never re-introduce unguarded `pdb.set_trace()` in `dnn_tf.py`.
- **Verify changes with numerical gradient checking**: After touching `activation_np.py` or `Layer`/`NeuralNet` math in `dnn_np.py`, verify both `unit_test_layer` (`rel_error < 1e-4`, target `< 1e-7`) and full-network finite-difference gradient checks with `reg > 0`.
- **Known watchout (`activation_np.py`)**: `reLU(x)` must be elementwise (`np.maximum(0, x)`), not a reduction like `np.max(x, axis=0)`, which collapses batch dimension `(N, D)` to `(D,)` and breaks downstream `softmax_minus_max(..., axis=1)`.

# WatchET8: Real-Time Edge Object Detection and Crowd Counting via Quantized ExecuTorch on Arm Cortex-A

**Track**: Problem Statement 2: AI for Safer Communities: Edge-Based Object Detection and Counting  
**Target Hardware**: Raspberry Pi 4 & 5 (64-bit Quad-Core Arm Cortex-A72 / A76, Armv8/v8.2-A NEON SIMD)  
**Core Framework**: PyTorch 2.x and ExecuTorch Runtime with XNNPACK Delegate  
**Team Name**: WatchET8  
**Affiliation**: Indian Institute of Technology Madras (IIT Madras)  
**Authors**: 
* **Sonu Kumar** (Team Leader) | Email: `25f2100288@es.study.iitm.ac.in`
* **Krishn Singh** | Email: `24f3100088@es.study.iitm.ac.in`

## Abstract
Continuous situational awareness in municipal pedestrian zones and traffic junctions requires localized, deterministic spatial analysis. Conventional cloud-offloaded deep neural networks face severe operational bottlenecks: high latency jitter under fluctuating wide-area networks, bandwidth degradation, and civil data-privacy liabilities from continuous upstream video transmission. We present **WatchET8**, an air-gapped, zero-cloud Edge AI perception pipeline specifically engineered for lightweight spatial object detection and localized crowd census on low-power Arm Cortex-A single-board computers (Raspberry Pi 4 and 5). Built natively on PyTorch's on-device framework, **ExecuTorch**, our system captures an anchor-free lightweight vision backbone (YOLO11-Nano) into an optimized intermediate graph representation. Through post-training quantization (INT8 PTQ) and operator-level lowering via the XNNPACK delegate, 2D convolutions and matrix multiplications are directly targeted to Arm NEON 128-bit vector execution units (`SDOT`/`UDOT` primitives). WatchET8 provides a mathematically rigorous, energy-efficient methodology for zero-cloud public safety monitoring under strict embedded thermal (5 to 12W) and memory constraints.

**Keywords**: Edge AI, ExecuTorch, Arm Cortex-A, NEON SIMD, XNNPACK, Post-Training Quantization, Embedded Vision, Zero-Cloud Inference.

## 1. Introduction & Societal Problem
Real-time monitoring of pedestrian density and vehicle movement is essential for municipal traffic management, public safety, and rapid incident response. Existing commercial approaches rely heavily on continuous video streaming to remote cloud data centers. This cloud-centric architecture exhibits three core points of failure:
1. **Network Degradation and Outages**: Suburban and high-density municipal corridors experience severe bandwidth throttling and WAN dropouts, rendering cloud systems inoperable during connectivity loss.
2. **Latency Inelasticity and Jitter**: Network round-trip times (300 to 2500 ms) prevent real-time, deterministic local awareness.
3. **Data Sovereignty and Civil Privacy**: Continuous cloud streaming of public visual feeds violates civil privacy mandates. Processing data locally in volatile memory ensures no raw video frames ever leave the physical device.

To resolve these challenges, **WatchET8** demonstrates an edge-native methodology deploying quantized neural graphs strictly on local Arm silicon without cloud dependencies or dedicated external accelerators.

## 2. Problem Formulation & Edge Operational Constraints
We formulate spatial detection and localized counting as a constrained edge optimization problem.

### 2.1 Edge Boundary Conditions
The runtime environment on the Raspberry Pi target is bounded by three hardware constraints:
* **Thermal Dissipation Envelope**: System thermal design power $P_{\text{TDP}} \le 5\text{ to }12\text{ W}$, demanding high arithmetic energy efficiency.
* **Volatile Memory Budget**: Peak Resident Set Size (RSS) $\mathcal{M}_{\text{peak}} \le 1.0\text{ GB}$ within shared physical system memory.
* **Zero-Cloud Isolation**: Network transmission rate $\mathcal{R}_{\text{net}} \equiv 0\text{ KB/s}$ (strictly air-gapped).

### 2.2 Mathematical Objective
Let an input visual frame be denoted as a multidimensional tensor $\mathbf{X}_t \in \mathbb{R}^{H \times W \times 3}$. The edge system executes a deterministic spatial transformation $\mathcal{T}_{\text{edge}}$ followed by a quantized forward operator $\mathcal{M}_{\text{PTE}}$:

$$\hat{\mathbf{Y}}_t = \mathcal{M}_{\text{PTE}}\left(\mathcal{T}_{\text{edge}}(\mathbf{X}_t); \mathbf{W}_{\text{INT8}}\right) = \left\{ (\mathbf{b}_i, c_i, s_i) \right\}_{i=1}^{N_t}$$

Where:
* $\mathbf{b}_i = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in \mathbb{R}^4$ specifies normalized bounding box coordinates.
* $c_i \in \mathcal{C}$ defines the semantic category index, where $\mathcal{C} = \{\text{Pedestrian}, \text{Vehicle}\}$.
* $s_i \in [0, 1]$ represents the class posterior confidence score subject to an acceptance threshold $s_i \ge \tau_{\text{conf}}$.

The localized census count $\mathcal{K}_k(t)$ for each semantic class $k \in \mathcal{C}$ at timestamp $t$ is computed locally without network telemetry:

$$\mathcal{K}_k(t) = \sum_{i=1}^{N_t} \mathbb{I}(c_i = k) \cdot \mathbb{I}(s_i \ge \tau_{\text{conf}})$$

## 3. Proposed Technical Methodology & Architecture

### 3.1 Model Topology Selection
We select **YOLO11-Nano (`yolo11n`)** as the primary detection backbone. With approximately 2.6M parameters, it introduces an optimized C3k2 feature aggregation architecture and an anchor-free decoupled detection head, maximizing feature reuse while minimizing arithmetic compute operations (6.5 GFLOPs at $640 \times 640$).

### 3.2 Compilation & Lowering Pipeline
The deployment architecture decouples host compilation from the embedded Arm runtime:

```
[ PyTorch 2.x Pretrained YOLO11n Backbone ]
        │
        ▼
[ Graph Capture: torch.export.export() ] ──> Generates ExportedProgram IR with static tensor shapes
        │
        ▼
[ Edge Lowering: exir.to_edge() ] ──> Hardware-agnostic dialect with ATen-to-Edge canonicalization
        │
        ▼
[ Post-Training Quantization (PTQ) ] ──> 8-bit Uniform Symmetric Quantization
        │
        ▼
[ Delegate Partitioning: XNNPACKPartitioner ] ──> Fuses Conv2D/BatchNorm/ReLU into Arm NEON kernels
        │
        ▼
[ Binary Emission: .to_executorch() ] ──> Emits standalone watchet8_yolo_int8.pte
        │
        ▼
[ Target Deployment on Raspberry Pi ] ──> In-memory zero-copy execution via C++/Python runtime
```

### 3.3 Symmetric INT8 Quantization Formalism
Model weights are quantized using uniform symmetric per-channel quantization:

$$q_w = \text{clamp}\left( \left\lfloor \frac{w}{S_w} \right\rceil, -128, 127 \right), \quad S_w = \frac{\max(|w|)}{127}$$

Activations are mapped via uniform affine per-tensor quantization:

$$q_a = \text{clamp}\left( \left\lfloor \frac{a}{S_a} \right\rceil + Z_a, 0, 255 \right)$$

### 3.4 Hardware-Level SIMD Acceleration via Arm NEON
At runtime, ExecuTorch routes tensor operations through the **XNNPACK delegate**. This triggers specialized Arm NEON micro-kernels executing `SDOT` (Signed Dot Product) and `UDOT` (Unsigned Dot Product) instructions. In a single clock cycle, each 128-bit vector lane computes four 8-bit integer multiplications and accumulates the result into a 32-bit vector register:

$$\text{acc}_{32} \leftarrow \text{acc}_{32} + \sum_{k=0}^{3} a_k^{(8)} \times w_k^{(8)}$$

This quadruples arithmetic compute density relative to standard scalar FP32 execution while operating within the target thermal budget.

### 3.5 Advanced Systems, Compiler, and Microarchitectural Optimization Stack
To maximize inference throughput and eliminate latency jitter on Arm Cortex-A silicon, WatchET8 integrates an exhaustive, multi-tier systems optimization stack:

1. **Graph Surgery and Class Pruning**:
   * Pretrained YOLO11 models compute logits for 80 MS-COCO classes (output tensor $[B, 84, 8400]$).
   * We apply offline head pruning to slice the final convolutional prediction tensor down to 2 target classes ($\mathcal{C} = \{\text{Pedestrian}, \text{Vehicle}\}$):
     $$\mathbf{W}_{\text{head}} \in \mathbb{R}^{84 \times C_{\text{in}}} \xrightarrow{\text{slice}} \mathbf{W}_{\text{pruned}} \in \mathbb{R}^{6 \times C_{\text{in}}}$$
   * This contracts the output tensor dimension from $[B, 84, 8400]$ down to $[B, 6, 8400]$, cutting post-decode memory traversal by $>70\%$.

2. **Vectorized Non-Maximum Suppression (NMS)**:
   * Redundant candidate detections are suppressed using vectorized Intersection-over-Union (IoU) filtering:
     $$\text{IoU}(\mathbf{b}_i, \mathbf{b}_j) = \frac{\text{Area}(\mathbf{b}_i \cap \mathbf{b}_j)}{\text{Area}(\mathbf{b}_i \cup \mathbf{b}_j)} < \tau_{\text{IoU}}$$
   * Branch prediction hints (`__builtin_expect(score > conf_threshold, 0)`) prevent Cortex-A branch misprediction stalls on non-object background anchors.

3. **Profile-Guided Optimization (PGO) and Link-Time Optimization (LTO)**:
   * The ExecuTorch C++ runtime is built via a two-pass profile-guided workflow (`-fprofile-generate` followed by `-fprofile-use`), reorganizing instruction layout to maximize L1 instruction-cache hit rates.
   * Dead-code kernel stripping (`-flto -Wl,--gc-sections`) purges unused operator kernels (e.g., recurrent and transformer layers), compressing the executable binary to under 1.5 MB.

4. **16 KB Memory Page Alignment and Cache Prefetching**:
   * Tuned to Raspberry Pi OS 64-bit native 16 KB virtual memory page size, reducing Translation Lookaside Buffer (TLB) misses by $4\times$ when streaming weight tensors through RAM.
   * Software prefetching intrinsics (`__builtin_prefetch(&tensor_data[i + 64], 0, 3)`) signal the Cortex-A76 prefetcher to stage upcoming cache lines into L2/L3 before the execution core demands them.

5. **Vectorized Fast-Math and Sigmoid Look-Up Table (LUT)**:
   * Replaced expensive transcendental exponential calculations (`exp()`) with a static 256-entry Look-Up Table for bounded activation mappings ($[-6.0, +6.0]$):
     $$\sigma(x) \approx \text{LUT}[\text{quantize}(x)]$$
   * Coordinate decoders are compiled with `-ffast-math -fno-math-errno` to utilize native single-cycle NEON vector approximation instructions (`VRECPE`/`VRSQRTE`).

6. **Zero-Copy Pipeline and VideoCore Hardware Offload**:
   * Video decoding is offloaded from the CPU to the Raspberry Pi hardware video pipeline via V4L2 and FFmpeg DRM PRIME, sparing all four Cortex-A cores exclusively for neural network execution.
   * Direct DMA-BUF memory pointers are passed to ExecuTorch via `from_blob()` zero-copy wrappers, and input evaluation frames are staged directly in shared memory (`/dev/shm`), bypassing slow MicroSD card read latencies.

7. **Static Memory Arena and OS Determinism**:
   * ExecuTorch pre-allocates a single contiguous memory arena at initialization. Every subsequent frame reuses this static buffer, eliminating runtime `malloc()`/`free()` heap fragmentation.
   * Core isolation (`isolcpus`), thread affinity (`taskset`), and zero swap thrashing (`vm.swappiness=0`) guarantee microsecond-level latency determinism.

## 4. Overall Implementation Plan

Our deployment plan is organized into four structured phases:

* **Phase 1: Ingestion Harness and Preprocessing Pipeline**:
  * Set up deterministic frame ingestion for organizer-provided test video clips (10 to 20s) and static images.
  * Implement aspect-preserving letterbox scaling and normalized tensor transformations:
    $$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X}_{\text{letterbox}}}{255.0} \in [0.0, 1.0]^{3 \times 640 \times 640}$$
  * Apply zero-copy memory mapping directly into contiguous model input buffers via `/dev/shm`.
* **Phase 2: ExecuTorch Runtime Deployment and Threshold Calibration**:
  * Load the pre-compiled `watchet8_yolo_int8.pte` model into the static ExecuTorch memory arena.
  * Verify hardware NEON `SDOT`/`UDOT` kernel dispatch on the Raspberry Pi.
  * Calibrate confidence ($\tau_{\text{conf}}$) and Intersection-over-Union ($\tau_{\text{IoU}}$) thresholds to maximize pedestrian and vehicle detection precision under varying illumination.
* **Phase 3: Automated Validation and Metrics Logging**:
  * Execute sequential batch inference across all organizer-supplied evaluation inputs.
  * Compute localized counting precision against organizer ground truth $\mathcal{K}_k^*(t)$:
    $$\text{Accuracy} = \left(1 - \frac{|\mathcal{K}_k(t) - \mathcal{K}_k^*(t)|}{\mathcal{K}_k^*(t)}\right) \times 100\%$$
  * Log detected bounding coordinates, class predictions, object counts, per-frame inference latency ($\tau_{\text{infer}}$), and resident memory footprint ($\mathcal{M}_{\text{RSS}}$) into structured CSV/JSON telemetry.
* **Phase 4: Output Delivery and Demonstration**:
  * Present a clean terminal CLI demonstration displaying real-time census counts, formatted results tables, and annotated sample detection outputs.
  * Validate that execution operates 100% offline with zero network connectivity.

## 5. Conclusion
**WatchET8** provides a robust, research-grade systems approach for edge-based spatial object detection and counting on low-cost Arm hardware. By systematically pairing an anchor-free single-stage backbone (YOLO11-Nano) with ExecuTorch's INT8 quantization and XNNPACK/NEON acceleration, the proposed methodology delivers real-time, air-gapped performance within the strict thermal and memory budgets of the Raspberry Pi. This framework offers a scalable, privacy-preserving foundation for decentralized municipal infrastructure and smart-community monitoring.

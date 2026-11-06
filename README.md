<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; color: #1f2328; line-height: 1.6;">

# <span style="color: #0969da;">WatchET8: Watching Every Clock</span>

<div style="background-color: #f6f8fa; border-left: 4px solid #0969da; padding: 12px 16px; margin-bottom: 20px; border-radius: 4px;">
  <p style="margin: 0 0 6px 0;"><strong>Track:</strong> <span style="color: #1f2328;">Problem Statement 2: AI for Safer Communities: Edge-Based Object Detection and Counting</span></p>
  <p style="margin: 0 0 6px 0;"><strong>Target Hardware:</strong> <span style="color: #0550ae; font-weight: 600;">Raspberry Pi 4 & 5 (64-bit Quad-Core Arm Cortex-A72 / A76, Armv8/v8.2-A NEON SIMD)</span></p>
  <p style="margin: 0 0 6px 0;"><strong>Core Framework:</strong> <span style="color: #0550ae; font-weight: 600;">PyTorch 2.x and ExecuTorch Runtime with XNNPACK Delegate</span></p>
  <p style="margin: 0 0 6px 0;"><strong>Team Name:</strong> <span style="color: #8250df; font-weight: 600;">WatchET8</span> | <strong>Affiliation:</strong> <span style="color: #1a7f37; font-weight: 600;">Indian Institute of Technology Madras (IIT Madras)</span></p>
  <p style="margin: 0;"><strong>Authors:</strong> Sonu Kumar (Team Leader, <code>25f2100288@es.study.iitm.ac.in</code>) | Krishn Singh (<code>24f3100088@es.study.iitm.ac.in</code>)</p>
</div>

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">Abstract</span>

Continuous situational awareness in municipal pedestrian zones and traffic junctions requires localized, deterministic spatial analysis. Conventional cloud-offloaded deep neural networks face severe operational bottlenecks: high latency jitter under fluctuating wide-area networks, bandwidth degradation, and civil data-privacy liabilities from continuous upstream video transmission. We present <strong style="color: #8250df;">WatchET8</strong>, an air-gapped, zero-cloud Edge AI perception pipeline specifically engineered for lightweight spatial object detection and localized crowd census on low-power Arm Cortex-A single-board computers (Raspberry Pi 4 and 5). Built natively on PyTorch's on-device framework, <strong style="color: #0969da;">ExecuTorch</strong>, our system captures an anchor-free lightweight vision backbone (YOLO11-Nano) into an optimized intermediate graph representation. Through post-training quantization (INT8 PTQ) and operator-level lowering via the XNNPACK delegate, 2D convolutions and matrix multiplications are directly targeted to Arm NEON 128-bit vector execution units (<code>SDOT</code>/<code>UDOT</code> primitives). WatchET8 provides a mathematically rigorous, energy-efficient methodology for zero-cloud public safety monitoring under strict embedded thermal (5 to 12W) and memory constraints.

<p style="background: #ddf4ff; color: #0969da; padding: 6px 12px; border-radius: 6px; font-size: 13px;"><strong>Keywords:</strong> Edge AI, ExecuTorch, Arm Cortex-A, NEON SIMD, XNNPACK, Post-Training Quantization, Embedded Vision, Zero-Cloud Inference.</p>

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">1. Introduction & Societal Problem</span>

Real-time monitoring of pedestrian density and vehicle movement is essential for municipal traffic management, public safety, and rapid incident response. Existing commercial approaches rely heavily on continuous video streaming to remote cloud data centers. This cloud-centric architecture exhibits three core points of failure:
1. <strong style="color: #cf222e;">Network Degradation and Outages:</strong> Suburban and high-density municipal corridors experience severe bandwidth throttling and WAN dropouts, rendering cloud systems inoperable during connectivity loss.
2. <strong style="color: #9a6700;">Latency Inelasticity and Jitter:</strong> Network round-trip times (300 to 2500 ms) prevent real-time, deterministic local awareness.
3. <strong style="color: #1a7f37;">Data Sovereignty and Civil Privacy:</strong> Continuous cloud streaming of public visual feeds violates civil privacy mandates. Processing data locally in volatile memory ensures no raw video frames ever leave the physical device.

To resolve these challenges, **WatchET8** demonstrates an edge-native methodology deploying quantized neural graphs strictly on local Arm silicon without cloud dependencies or dedicated external accelerators.

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">2. Problem Formulation & Edge Operational Constraints</span>

We formulate spatial detection and localized counting as a constrained edge optimization problem.

### 2.1 Edge Boundary Conditions
The runtime environment on the Raspberry Pi target is bounded by three hardware constraints:
* **Thermal Dissipation Envelope:** System thermal design power $P_{\text{TDP}} \le 5\text{ to }12\text{ W}$, demanding high arithmetic energy efficiency.
* **Volatile Memory Budget:** Peak Resident Set Size (RSS) $\mathcal{M}_{\text{peak}} \le 1.0\text{ GB}$ within shared physical system memory.
* **Zero-Cloud Isolation:** Network transmission rate $\mathcal{R}_{\text{net}} \equiv 0\text{ KB/s}$ (strictly air-gapped).

### 2.2 Mathematical Objective
Let an input visual frame be denoted as a multidimensional tensor $\mathbf{X}_t \in \mathbb{R}^{H \times W \times 3}$. The edge system executes a deterministic spatial transformation $\mathcal{T}_{\text{edge}}$ followed by a quantized forward operator $\mathcal{M}_{\text{PTE}}$:

$$\hat{\mathbf{Y}}_t = \mathcal{M}_{\text{PTE}}\left(\mathcal{T}_{\text{edge}}(\mathbf{X}_t); \mathbf{W}_{\text{INT8}}\right) = \left\{ (\mathbf{b}_i, c_i, s_i) \right\}_{i=1}^{N_t}$$

Where:
* $\mathbf{b}_i = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in \mathbb{R}^4$ specifies normalized bounding box coordinates.
* $c_i \in \mathcal{C}$ defines the semantic category index, where $\mathcal{C} = \{\text{Pedestrian}, \text{Vehicle}\}$.
* $s_i \in [0, 1]$ represents the class posterior confidence score subject to an acceptance threshold $s_i \ge \tau_{\text{conf}}$.

The localized census count $\mathcal{K}_k(t)$ for each semantic class $k \in \mathcal{C}$ at timestamp $t$ is computed locally without network telemetry:

$$\mathcal{K}_k(t) = \sum_{i=1}^{N_t} \mathbb{I}(c_i = k) \cdot \mathbb{I}(s_i \ge \tau_{\text{conf}})$$

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">3. Proposed Technical Methodology & Architecture</span>

### 3.1 Model Topology Selection
We select <strong style="color: #0969da;">YOLO11-Nano (<code>yolo11n</code>)</strong> as the primary detection backbone. With approximately 2.6M parameters, it introduces an optimized C3k2 feature aggregation architecture and an anchor-free decoupled detection head, maximizing feature reuse while minimizing arithmetic compute operations (6.5 GFLOPs at $640 \times 640$).

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

1. <strong style="color: #0969da;">Graph Surgery and Class Pruning:</strong>
   * Pretrained YOLO11 models compute logits for 80 MS-COCO classes (output tensor $[B, 84, 8400]$).
   * We apply offline head pruning to slice the final convolutional prediction tensor down to 2 target classes ($\mathcal{C} = \{\text{Pedestrian}, \text{Vehicle}\}$):
     $$\mathbf{W}_{\text{head}} \in \mathbb{R}^{84 \times C_{\text{in}}} \xrightarrow{\text{slice}} \mathbf{W}_{\text{pruned}} \in \mathbb{R}^{6 \times C_{\text{in}}}$$
   * This contracts the output tensor dimension from $[B, 84, 8400]$ down to $[B, 6, 8400]$, cutting post-decode memory traversal by $>70\%$.

2. <strong style="color: #0969da;">Vectorized Non-Maximum Suppression (NMS):</strong>
   * Redundant candidate detections are suppressed using vectorized Intersection-over-Union (IoU) filtering:
     $$\text{IoU}(\mathbf{b}_i, \mathbf{b}_j) = \frac{\text{Area}(\mathbf{b}_i \cap \mathbf{b}_j)}{\text{Area}(\mathbf{b}_i \cup \mathbf{b}_j)} < \tau_{\text{IoU}}$$
   * Branch prediction hints (<code>__builtin_expect(score > conf_threshold, 0)</code>) prevent Cortex-A branch misprediction stalls on non-object background anchors.

3. <strong style="color: #0969da;">Profile-Guided Optimization (PGO) and Link-Time Optimization (LTO):</strong>
   * The ExecuTorch C++ runtime is built via a two-pass profile-guided workflow (<code>-fprofile-generate</code> followed by <code>-fprofile-use</code>), reorganizing instruction layout to maximize L1 instruction-cache hit rates.
   * Dead-code kernel stripping (<code>-flto -Wl,--gc-sections</code>) purges unused operator kernels, compressing the executable binary to under 1.5 MB.

4. <strong style="color: #0969da;">16 KB Memory Page Alignment and Cache Prefetching:</strong>
   * Tuned to Raspberry Pi OS 64-bit native 16 KB virtual memory page size, reducing Translation Lookaside Buffer (TLB) misses by $4\times$ when streaming weight tensors through RAM.
   * Software prefetching intrinsics (<code>__builtin_prefetch(&tensor_data[i + 64], 0, 3)</code>) signal the Cortex-A76 prefetcher to stage upcoming cache lines into L2/L3 before the execution core demands them.

5. <strong style="color: #0969da;">Vectorized Fast-Math and Sigmoid Look-Up Table (LUT):</strong>
   * Replaced expensive transcendental exponential calculations (<code>exp()</code>) with a static 256-entry Look-Up Table for bounded activation mappings ($[-6.0, +6.0]$):
     $$\sigma(x) \approx \text{LUT}[\text{quantize}(x)]$$
   * Coordinate decoders are compiled with <code>-ffast-math -fno-math-errno</code> to utilize native single-cycle NEON vector approximation instructions (<code>VRECPE</code>/<code>VRSQRTE</code>).

6. <strong style="color: #0969da;">Zero-Copy Pipeline and VideoCore Hardware Offload:</strong>
   * Video decoding is offloaded from the CPU to the Raspberry Pi hardware video pipeline via V4L2 and FFmpeg DRM PRIME, sparing all four Cortex-A cores exclusively for neural network execution.
   * Direct DMA-BUF memory pointers are passed to ExecuTorch via <code>from_blob()</code> zero-copy wrappers, and input evaluation frames are staged directly in shared memory (<code>/dev/shm</code>), bypassing slow MicroSD card read latencies.

7. <strong style="color: #0969da;">Static Memory Arena and OS Determinism:</strong>
   * ExecuTorch pre-allocates a single contiguous memory arena at initialization. Every subsequent frame reuses this static buffer, eliminating runtime <code>malloc()</code>/<code>free()</code> heap fragmentation.
   * Core isolation (<code>isolcpus</code>), thread affinity (<code>taskset</code>), and zero swap thrashing (<code>vm.swappiness=0</code>) guarantee microsecond-level latency determinism.

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">4. Overall Implementation Plan</span>

Our deployment plan is organized into four structured phases:

* <strong style="color: #0969da;">Phase 1: Ingestion Harness and Preprocessing Pipeline</strong>
  * Set up deterministic frame ingestion for organizer-provided test video clips (10 to 20s) and static images.
  * Implement aspect-preserving letterbox scaling and normalized tensor transformations:
    $$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X}_{\text{letterbox}}}{255.0} \in [0.0, 1.0]^{3 \times 640 \times 640}$$
  * Apply zero-copy memory mapping directly into contiguous model input buffers via <code>/dev/shm</code>.

* <strong style="color: #0969da;">Phase 2: ExecuTorch Runtime Deployment and Threshold Calibration</strong>
  * Load the pre-compiled <code>watchet8_yolo_int8.pte</code> model into the static ExecuTorch memory arena.
  * Verify hardware NEON <code>SDOT</code>/<code>UDOT</code> kernel dispatch on the Raspberry Pi.
  * Calibrate confidence ($\tau_{\text{conf}}$) and Intersection-over-Union ($\tau_{\text{IoU}}$) thresholds to maximize pedestrian and vehicle detection precision under varying illumination.

* <strong style="color: #0969da;">Phase 3: Automated Validation and Metrics Logging</strong>
  * Execute sequential batch inference across all organizer-supplied evaluation inputs.
  * Compute localized counting precision against organizer ground truth $\mathcal{K}_k^*(t)$:
    $$\text{Accuracy} = \left(1 - \frac{|\mathcal{K}_k(t) - \mathcal{K}_k^*(t)|}{\mathcal{K}_k^*(t)}\right) \times 100\%$$
  * Log detected bounding coordinates, class predictions, object counts, per-frame inference latency ($\tau_{\text{infer}}$), and resident memory footprint ($\mathcal{M}_{\text{RSS}}$) into structured CSV/JSON telemetry.

* <strong style="color: #0969da;">Phase 4: Output Delivery and Demonstration</strong>
  * Present a clean terminal CLI demonstration displaying real-time census counts, formatted results tables, and annotated sample detection outputs.
  * Validate that execution operates 100% offline with zero network connectivity.

## <span style="color: #0969da; border-bottom: 2px solid #eaeef2; padding-bottom: 4px; display: block;">5. Conclusion</span>

<strong style="color: #8250df;">WatchET8</strong> provides a robust, research-grade systems approach for edge-based spatial object detection and counting on low-cost Arm hardware. By systematically pairing an anchor-free single-stage backbone (YOLO11-Nano) with ExecuTorch's INT8 quantization and XNNPACK/NEON acceleration, the proposed methodology delivers real-time, air-gapped performance within the strict thermal and memory budgets of the Raspberry Pi. This framework offers a scalable, privacy-preserving foundation for decentralized municipal infrastructure and smart-community monitoring.

</div>

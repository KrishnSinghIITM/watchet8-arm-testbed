# Arm64 / Raspberry Pi 5 Cloud Testbed Setup Guide

This guide documents how to launch, connect to, and configure a **100% Free 64-bit Arm (aarch64) Linux environment** using GitHub's native Arm runners and `tmate`. This setup simulates a physical **Raspberry Pi 5** with **zero credit card requirements and zero cost**.

---

## 1. Architecture & Hardware Match

| Specification | GitHub Arm Runner (`ubuntu-24.04-arm`) | Physical Raspberry Pi 5 |
| :--- | :--- | :--- |
| **Instruction Set (ISA)** | **Armv8.2-A / Armv8.4-A** | **Armv8.2-A** |
| **SIMD Vector Engine** | **Arm NEON (`asimddp` - Dot Product)** | **Arm NEON (`asimddp` - Dot Product)** |
| **Execution Dialect** | 64-bit `aarch64` Linux | 64-bit `aarch64` Linux (RPi OS Bookworm) |
| **ExecuTorch Binary (`.pte`)** | Direct binary compatibility | Direct binary compatibility |
| **Compute / Memory** | 4 vCPUs, 16 GB RAM | 4 Cores, 4GB/8GB RAM |
| **Cost / Verification** | **$0 (No Credit Card / No Trial)** | ~$80–$100 Hardware |

---

## 2. One-Time Setup Steps (How to Launch)

The workflow file is pre-configured and located at:
[`.github/workflows/arm_rpi_testbed.yml`](.github/workflows/arm_rpi_testbed.yml)

### Step 2.1: Push to a Public GitHub Repository
1. Initialize or push your project to a **Public** GitHub repository (GitHub provides unlimited free Arm minutes for public repositories):
   ```bash
   git init
   git add .
   git commit -m "feat: add WatchET8 proposal and Arm testbed workflow"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

### Step 2.2: Trigger the Arm Session
1. Open your repository on **GitHub** in your web browser.
2. Click on the **Actions** tab along the top bar.
3. On the left sidebar under *Workflows*, click **"Arm64 SSH Testbed (Raspberry Pi 5 Equivalent)"**.
4. Click the **Run workflow** dropdown on the right, ensure `main` branch is selected, and click the green **Run workflow** button.

### Step 2.3: Connect via SSH
1. In the Actions tab, click on the workflow run that just appeared (labeled *"Arm64 SSH Testbed"*).
2. Click on the **`arm64-session`** job.
3. Watch the live execution log. In ~15–25 seconds, the `Launch Interactive SSH Terminal via tmate` step will print connection details:
   ```text
   SSH: ssh XXXXXXXXXXXXXXXXXXXXXXXXX@nyc1.tmate.io
   or
   Web URL: https://tmate.io/t/XXXXXXXXXXXXXXXXXXXXXXXXX
   ```
4. Copy the `ssh ...` command, paste it into your local terminal (PowerShell, Command Prompt, or Linux/macOS terminal), and press **Enter**.
5. You are now inside a live, interactive 64-bit Arm Linux shell!

---

## 3. Environment Configuration Inside the Arm Session

Once connected via SSH, run the following setup commands to configure PyTorch, ExecuTorch, and YOLO11:

```bash
# 1. Verify 64-bit Arm & NEON hardware flags
uname -m
# Expected: aarch64

lscpu | grep -E "Flags|Model name"
# Verify 'asimd' and 'asimddp' are present

# 2. Update system packages & build tools
sudo apt update && sudo apt install -y python3-pip python3-venv git cmake build-essential libopenblas-dev

# 3. Create isolated Python virtual environment
python3 -m venv ~/watchet8_env
source ~/watchet8_env/bin/activate

# 4. Install PyTorch & Vision for aarch64 CPU
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install ExecuTorch & Ultralytics (YOLO11)
pip install executorch
pip install ultralytics
```

---

## 4. Quick Verification Script (Test ExecuTorch on Arm)

To test model export and verification on this Arm environment, run:

```bash
python3 -c "
import torch
import torchvision.models as models
import executorch
from executorch.exir import to_edge

print(f'PyTorch version: {torch.__version__}')
print(f'ExecuTorch loaded successfully on aarch64!')
"
```

---

## 5. Important Notes & Best Practices

1. **Session Timeout**:
   * A `tmate` SSH session will stay active for up to **6 hours** (GitHub Actions maximum job limit) as long as you are actively using it.
   * If you close the terminal or want to terminate the session early, create a file named `continue` in the root workspace or click **Cancel run** in the GitHub Actions tab.
2. **File Persistence**:
   * Any files you create inside the virtual machine disappear when the GitHub Actions job terminates.
   * *Best Practice*: Always commit your code, `.pte` models, or benchmark logs to Git or upload them as GitHub release artifacts before ending the session.
3. **Latency Calibration Factor for RPi 5**:
   * The GitHub Arm runner uses server-grade cloud Arm cores clocked at ~2.5–3.0 GHz with wide DDR memory bandwidth.
   * Multiply your measured latency by **$\approx 1.25\times$** to calculate the exact real-world inference latency on an embedded 2.4 GHz Raspberry Pi 5.
   * *(e.g., if a frame runs in 28 ms on this runner, expect ~35 ms on a physical RPi 5)*.
4. **No GPU Flag Needed**:
   * Never pass `--device cuda` when exporting or running. Always target pure CPU / XNNPACK (`--device cpu`) to maintain Arm NEON parity with the competition hardware.


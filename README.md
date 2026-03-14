# Alpamayo Edge Inference Optimization and Benchmarking
Performance analysis and optimization of NVIDIA's Alpamayo VLA model for edge deployment. Testing conducted on NVIDIA RTX 6000 Ada.

## Motivation
NVIDIA's Alpamayo (nvidia/Alpamayo-R1-10B) is a vision-language-action model for autonomous vehicle decision-making. This project benchmarks its inference performance on edge hardware and explores optimization strategies for real-time deployment.


## Repository Structure
```
alpamayo-inference-benchmark/
├── benchmarks/          # Benchmark scripts
├── configs/             # YAML configurations
├── results/             # Raw data and analysis
└── README.md
```

## Hardware

- **Workstation**: NVIDIA RTX 6000 Ada (48GB)
- **Edge target**: NVIDIA Drive Thor (planned)

## Setup

**Prerequisites:**
- NVIDIA GPU with CUDA support
- Python 3.10+
- Access to nvidia/Alpamayo-R1-10B model

1. **Install Alpamayo:**
```bash
   git clone https://github.com/NVlabs/alpamayo.git
   cd alpamayo
```
Follow the Install instructions from Alpamayo repository

**Install dependencies:**
```bash
cd ../alpamayo-inference-benchmark
pip install -r requirements.txt
```

**Download model** (happens automatically on first run):
```bash
python benchmarks/run_suite.py
```

## Usage

Run baseline latency benchmark:
```bash
python benchmarks/run_suite.py --config configs/baseline.yaml
```

## References

- [Alpamayo GitHub](https://github.com/NVlabs/alpamayo)
- [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)


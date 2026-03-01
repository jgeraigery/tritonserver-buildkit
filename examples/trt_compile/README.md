# TensorRT Compilation — ResNet-18 Image Classification

This example compiles a pretrained ResNet-18 ONNX model to a TensorRT engine during the build phase using `trt_compile`, then serves it on Triton Inference Server.

During build, `tsbk` will:

1. Download the ONNX model artifact from MLflow
2. Compile it to a `.plan` file using `trtexec` with fp16 precision (via Docker or a Kubernetes Job)
3. Set the backend to `tensorrt` in the generated `config.pbtxt`
4. Cache the compiled engine locally so subsequent builds skip compilation

## Prerequisites

- Install example requirements:

```bash
pip install -r requirements.txt
```

- **Docker with GPU access** (for local compilation), or
- **Kubernetes cluster with GPU nodes** + `TSBK_S3_PREFIX` env var set (for remote compilation)

## Setup

Export a pretrained ResNet-18 to ONNX and register it with MLflow:

```bash
python create-model.py
```

This exports `resnet18.onnx` with:
- Input: `image` — `[batch, 3, 224, 224]` float32 (ImageNet-normalized RGB)
- Output: `logits` — `[batch, 1000]` float32 (class scores)

## Build and Run (local GPU via Docker)

```bash
python server.py --test
```

This will:
- Build the model repo, compiling the ONNX model to TensorRT with fp16 precision
- Launch Triton server in a Docker container with GPU access
- Run the MLflow registered input example as a test case
- Stop the server

## Build and Run (remote GPU via Kubernetes)

If you don't have a local GPU but have access to a Kubernetes cluster with GPU nodes, pass `--gpu-name` to target a specific GPU type via Karpenter:

```bash
export TSBK_S3_PREFIX=s3://your-bucket/tsbk-cache
python server.py --test --gpu-name a10g
```

The `--gpu-name` value maps to a Karpenter node selector (`karpenter.k8s.aws/instance-gpu-name`) so the compilation job is scheduled on the correct hardware.

## Build Only

```bash
python server.py --build-only
```

After building, the model repo will look like:

```
model-repo/
└── resnet18-trt/
    └── resnet18/
        ├── config.pbtxt    # backend: "tensorrt", max_batch_size: 8
        └── 1/
            └── model.plan   # compiled TensorRT engine (fp16)
```

## SDK Usage

The key addition compared to the standard SDK example is the `trt_compile` dict on the model version:

```python
tsbk.TritonModel(
    max_batch_size=8,
    versions=[
        tsbk.TritonModelVersion(
            artifact_uri="models:/resnet18-imagenet/1",
            trt_compile={
                "enabled": True,
                "precision": "fp16",          # optional: fp16, int8, best
                "workspace_size": 4096,       # optional: max workspace in MB
                "gpu_name": "a10g",           # optional: Karpenter GPU node selector for K8s
                "trt_image": "nvcr.io/...",   # optional: override TRT container image
                "extra_args": "--verbose",    # optional: raw trtexec flags
            },
        )
    ],
)
```

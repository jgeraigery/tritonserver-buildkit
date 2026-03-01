import argparse

import tsbk


def model_repo(model_repo_path: str, artifact_uri: str, gpu_name: str | None = None) -> tsbk.TritonModelRepo:
    """Build a model repo that compiles a ResNet-18 ONNX model to TensorRT.

    The trt_compile config on the version tells tsbk to:
      1. Download the ONNX model artifact
      2. Compile it to a TensorRT .plan file using trtexec (via Docker or K8s)
      3. Replace the .onnx with the .plan and set backend to tensorrt

    The compiled engine is cached under TSBK_DIR/trt_engines/ — subsequent
    builds with the same ONNX model and compile params skip compilation.

    When gpu_name is set, Kubernetes compilation uses it as a Karpenter node
    selector (karpenter.k8s.aws/instance-gpu-name) to schedule on the right
    GPU hardware. Requires TSBK_S3_PREFIX to be set for artifact transfer.
    """
    trt_compile = {
        "enabled": True,
        "precision": "fp16",
    }
    if gpu_name:
        trt_compile["gpu_name"] = gpu_name

    return tsbk.TritonModelRepo(
        "resnet18-trt",
        path=model_repo_path,
        models={
            "resnet18": tsbk.TritonModel(
                max_batch_size=8,
                versions=[
                    tsbk.TritonModelVersion(
                        artifact_uri=artifact_uri,
                        trt_compile=trt_compile,
                    )
                ],
            )
        },
    )


def main(args):
    repo = model_repo(args.model_repo, args.model_artifact_uri, gpu_name=args.gpu_name)
    repo.build()

    if args.build_only:
        return

    repo.run(detach=args.test, gpus=True)

    if args.test:
        repo.test(url=repo.http_url)
        repo.stop()
        print("Tests passed!")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and serve a TensorRT-compiled ResNet-18 with tsbk")
    parser.add_argument(
        "--model_artifact_uri",
        type=str,
        default="models:/resnet18-imagenet/1",
        help="MLflow model URI for the ONNX ResNet-18",
    )
    parser.add_argument("--model-repo", type=str, default="./model-repo", help="Path to the model repository")
    parser.add_argument(
        "--build-only", action="store_true", help="Only build the model repository without starting the server"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--gpu-name",
        type=str,
        default=None,
        help="Target GPU for compilation, used as Karpenter node selector (e.g. a10g, t4)",
    )
    args = parser.parse_args()

    assert not (args.build_only and args.test), "Cannot use --build-only and --test together"

    main(args)

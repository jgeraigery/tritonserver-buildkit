import mlflow
import numpy as np
import onnx
import torch
import torchvision.models as models

# Load a pretrained ResNet-18 for image classification
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()

# Export to ONNX with a batch dimension and standard ImageNet input shape
# Input: [batch, 3, 224, 224] RGB image normalized to ImageNet stats
# Output: [batch, 1000] class logits
model_path = "resnet18.onnx"
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    resnet,
    dummy_input,
    model_path,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)

onnx_model = onnx.load(model_path)

# Log model to MLflow with a sample input (a random "image" tensor)
with mlflow.start_run() as run:
    mlflow.onnx.log_model(
        onnx_model,
        artifact_path="resnet18",
        registered_model_name="resnet18-imagenet",
        input_example={"image": np.random.randn(1, 3, 224, 224).astype(np.float32)},
    )
    print(f"Model registered: models:/resnet18-imagenet/1 (run_id={run.info.run_id})")

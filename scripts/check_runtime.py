"""Verify runtime imports and GPU kernels without downloading or loading model weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_runtime(*, imports_only: bool) -> dict:
    modules = (
        "torch",
        "torchvision",
        "bitsandbytes",
        "transformers",
        "open3d",
        "chromadb",
        "sam3.model_builder",
        "ltmem.perception",
        "ltmem.object_memory",
        "ltmem.model_runtime",
    )
    for name in modules:
        importlib.import_module(name)
    report = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "bitsandbytes",
                "transformers",
                "sam3",
                "open3d",
                "numpy",
                "chromadb",
            )
        },
        "imports": "passed",
        "gpu": "not_checked",
        "models": "not_loaded",
    }
    if imports_only:
        return report

    import torch
    import torchvision
    import bitsandbytes as bnb

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with NVIDIA GPU access")
    device = torch.device("cuda:0")
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close((x.to(device) @ x.to(device)).cpu(), x @ x)
    boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    if torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS returned an unexpected result")
    layer = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4")
    layer = layer.to(device)
    with torch.inference_mode():
        output = layer(torch.ones(1, 128, dtype=torch.float16, device=device))
    if output.shape != (1, 64) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("bitsandbytes NF4 inference produced invalid output")
    torch.cuda.synchronize()
    report.update(
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        gpu="passed: torch matmul, torchvision NMS, bitsandbytes NF4",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only", action="store_true", help="For image builds without a GPU"
    )
    args = parser.parse_args()
    print(json.dumps(check_runtime(imports_only=args.imports_only), indent=2))


if __name__ == "__main__":
    main()

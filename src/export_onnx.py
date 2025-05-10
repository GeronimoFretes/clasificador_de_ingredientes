import argparse
from pathlib import Path

import torch
import torchvision as tv
from torch import nn


def parse_args():
    p = argparse.ArgumentParser("Export trained EfficientNet to ONNX")
    p.add_argument("--weights", required=True,
                   help="Path to best_model.pth (state_dict)")
    p.add_argument("--out", default="best_model.onnx",
                   help="Destination ONNX file")
    p.add_argument("--img-size", type=int, default=224,
                   help="Expected square input resolution (default 224)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (≥11). 17 is fine for ORT ≥1.17")
    p.add_argument("--dynamic", action="store_true",
                   help="Export with dynamic batch axis")
    return p.parse_args()


def build_model(num_classes: int) -> nn.Module:
    """Re‑create the training architecture."""
    model = tv.models.efficientnet_b0(weights=None)        # blank backbone
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    num_classes)
    return model


def main():
    args = parse_args()

    # ---- load state_dict and infer number of classes -----------------------
    sd = torch.load(args.weights, map_location="cpu")
    num_classes = sd["classifier.1.weight"].shape[0]
    print(f"✔ Detected {num_classes} output classes from checkpoint.")

    # ---- rebuild model & load weights --------------------------------------
    model = build_model(num_classes)
    model.load_state_dict(sd, strict=True)
    model.eval()                      # important for export

    # ---- dummy input --------------------------------------------------------
    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # ---- dynamic axes settings ---------------------------------------------
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    # ---- make sure output folder exists ------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- export -------------------------------------------------------------
    print("🔄  Exporting to ONNX …")
    torch.onnx.export(
        model, dummy, out_path.as_posix(),
        export_params=True,         # store trained weights inside graph
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"✅  ONNX model written to: {out_path}")

    # ---- quick sanity check (optional) -------------------------------------
    try:
        import onnx, onnxruntime as ort
        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)
        sess = ort.InferenceSession(out_path.as_posix(),
                                    providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        print("ONNX forward pass OK — output shape:", ort_out.shape)
    except ImportError:
        print("Tip: `pip install onnx onnxruntime` for automated checking.")


if __name__ == "__main__":
    main()

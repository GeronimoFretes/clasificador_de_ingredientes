import argparse, time
from pathlib import Path

import torch, torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch import nn

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:                 
    classification_report = confusion_matrix = None

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained model")
    p.add_argument("--data", default="data/split", help="root that contains test/")
    p.add_argument("--weights", default="models/best_model.pth",
                   help="checkpoint to evaluate")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--confmat", action="store_true",
                   help="print confusion matrix / report (needs scikit‑learn)")
    p.add_argument("--heatmap_out", default="heatmap_confmat.png",
                   help="destination confusion matrix heatmap PNG file")
    return p.parse_args()

def confmat_heatmap(cm, out_path) :
    import pandas as pd
    import altair as alt
    
    df = pd.DataFrame([
        {"x": j, "y": i, "value": int(cm[i, j])}
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    ])

    threshold = df["value"].max() / 2

    base = alt.Chart(df).encode(
        x=alt.X('x:O', axis=None),
        y=alt.Y('y:O', axis=None)
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('value:Q',
                        scale=alt.Scale(range=['#f9efdb','#25543d']),
                        legend=None)
    )

    text = base.mark_text(fontSize=15, fontWeight="bolder",fontStyle="montserrat").encode(
        text=alt.condition(alt.datum.value >= 1, 'value:Q', alt.value('')),
        color=alt.condition(alt.datum.value > threshold,
                            alt.value('#f9efdb'),
                            alt.value('#25543d'))
    )

    chart = (heatmap + text).properties(
        width=300,
        height=300,
        background='#f9efdb'
    ).configure_view(stroke=None)
    
    chart.save(out_path, scale_factor=2, method='selenium')

@torch.inference_mode()
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- transforms ---------------------------------------------------
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    test_tfms = T.Compose([
        T.Resize(args.img_size + 32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # ---------- dataset / loader --------------------------------------------
    test_dir   = Path(args.data) / "test"
    ds_test    = tv.datasets.ImageFolder(test_dir, test_tfms)
    test_loader = DataLoader(ds_test, args.batch,
                             num_workers=args.workers, pin_memory=True)

    num_classes = len(ds_test.classes)
    print(f"Classes: {ds_test.classes}")

    # ---------- model --------------------------------------------------------
    model   = tv.models.efficientnet_b0(weights=None)        # fresh backbone
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    num_classes)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    # ---------- loop ---------------------------------------------------------
    correct = total = 0
    all_preds, all_labels = [], []
    t0 = time.time()

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds  = logits.argmax(1)

        correct += (preds == yb).sum().item()
        total   += yb.size(0)

        if args.confmat:
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    acc = correct / total
    dur = time.time() - t0
    print(f"\nTop‑1 accuracy on test: {acc:.2%}  ({correct}/{total}) "
          f"in {dur:.1f}s")

    # ---------- optional metrics --------------------------------------------
    if args.confmat and classification_report is not None: 
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        print("\nClassification report:")
        print(classification_report(
            y_true, y_pred, target_names=ds_test.classes, digits=3))

        cm = confusion_matrix(y_true, y_pred)
        
        print("Confusion matrix:")
        print(cm)
        
        out_path = Path(args.heatmap_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        confmat_heatmap(cm, out_path.as_posix())
        
    elif args.confmat:
        print("⚠️  scikit‑learn not installed; skipping confusion matrix.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()

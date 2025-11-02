from ultralytics import YOLO
from pathlib import Path
import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="plexor_dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "0", "1"])
    ap.add_argument("--project", default="runs_plexor")
    ap.add_argument("--name", default="y11s_person_colab")
    args = ap.parse_args()

    # sets devices to auto by default
    device = args.device if args.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
    Path(args.project).mkdir(parents=True, exist_ok=True)

    # loads model pretrained weights if not present
    model = YOLO("yolo11s")

    # train the model accordingly
    model.train(
        
        # data and epoch flags
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        seed=42,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        # momentum=0.937,
        weight_decay=0.01,
        warmup_epochs=3,

        # data augmentation flags
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        mosaic=0.5, mixup=0.1, close_mosaic=10,
        rect=True,
        cache=True,
        patience=20,

        # device and path flags
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # validate similarly
    model.val(
        data=args.data,
        split="val",
        batch=args.batch,
        plots=True,
        project=args.project,
        name=f"{args.name}_val",
        device=device,
        exist_ok=True,
    )

if __name__ == "__main__":
    main()

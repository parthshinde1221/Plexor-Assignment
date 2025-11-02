from ultralytics import YOLO
from pathlib import Path
import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs_plexor/y11s_person_colab/weights/best.pt")
    ap.add_argument("--source", required=True, help="Path to a video file or an image folder")
    ap.add_argument("--conf", type=float, default=0.3)
    # ap.add_argument("--iou", type=float, default=0.35)
    # ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "0", "1"])
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    # defaulit is again set to auto
    device = args.device if args.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # loads weights from the path specified 
    model = YOLO(args.weights)

    # predicts the model on either a video or folder of files
    model.predict(
        source=args.source,
        conf=args.conf,
        # iou=args.iou,
        # imgsz=args.imgsz,
        device=device,
        save=True,
        show=True,
        project=args.outdir,
        name="preds",
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_thickness=2,
    )

if __name__ == "__main__":
    main()

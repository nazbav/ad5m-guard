"""Обучение RF-DETR (Roboflow, лицензия Apache-2.0) на датасете из build_dataset.py.

Запускать в отдельном окружении (у RF-DETR свои версии numpy/scipy):
    .venv-rfdetr\\Scripts\\python tools\\train_rfdetr.py defects --size small

Под 4 ГБ видеопамяти: батч 2, накопление градиента ×8 (эффективный батч 16),
gradient checkpointing. Лучшие веса копируются в models/<имя>_rfdetr_<size>.pth.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="датасет в data/ (defects, scene)")
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weights", type=Path, help="дообучить с этих весов (.pth) вместо стартовых COCO")
    args = ap.parse_args(argv)

    import rfdetr
    dataset = ROOT / "data" / args.name
    if not (dataset / "data.yaml").exists():
        raise SystemExit(f"Нет {dataset / 'data.yaml'} — сначала build_dataset.py / update_data.py")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = ROOT / "runs" / f"{args.name}_rfdetr_{args.size}" / stamp
    kwargs = {"gradient_checkpointing": True}
    if args.weights:
        kwargs["pretrain_weights"] = str(args.weights)
    model = getattr(rfdetr, SIZES[args.size])(**kwargs)
    model.train(dataset_dir=str(dataset), dataset_file="yolo", epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.accum, lr=args.lr, num_workers=args.workers, output_dir=str(out),
                early_stopping=True, early_stopping_patience=args.patience)

    candidates = [out / n for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")]
    best = next((c for c in candidates if c.exists()), None)
    if best is None:
        raise SystemExit(f"Не нашёл лучших весов в {out}: {[p.name for p in out.glob('*.pth')]}")
    dst = ROOT / "models" / f"{args.name}_rfdetr_{args.size}.pth"
    archive = ROOT / "models" / f"{args.name}_rfdetr_{args.size}" / f"{stamp}.pth"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, archive)
    shutil.copy2(best, dst)
    print(f"Веса: {archive}\nТекущая: {dst}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

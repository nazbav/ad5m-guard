"""Обучает модель на датасете из build_dataset.py и сразу проверяет её на test.

    python train.py defects                      # data/defects → models/defects.pt
    python train.py scene --model yolo26n.pt     # data/scene   → models/scene.pt

Лучшие веса копируются в models/<имя>.pt (последняя модель) и в
models/<имя>/<дата>.pt (архив). Рядом кладётся отчёт о проверке.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

from ddet.config import ROOT


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="имя модели = папка датасета в data/ (defects, scene)")
    ap.add_argument("--model", default="yolo26s.pt", help="с каких весов начинать (yolo26n/s/m.pt или свой .pt)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40, help="остановиться, если столько эпох нет улучшения")
    ap.add_argument("--imgsz", type=int, default=640)
    # AutoBatch на 4 ГБ ноутбучной A1000 выбирает 3 при реально занятом ~1 ГБ — слишком мало.
    ap.add_argument("--batch", type=float, default=8, help="размер батча или доля видеопамяти (0..1)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    data = ROOT / "data" / args.name / "data.yaml"
    if not data.exists():
        raise SystemExit(f"Нет {data}. Сначала соберите датасет: python build_dataset.py ...")
    if args.device == "cpu":
        print("ВНИМАНИЕ: CUDA недоступна, обучение на CPU будет очень долгим.")

    # Стартовые веса и веса для проверки AMP Ultralytics скачивает в текущую папку.
    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    os.chdir(weights)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    batch = int(args.batch) if args.batch >= 1 else args.batch
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
        device=args.device,
        cos_lr=True,
        close_mosaic=15,
        seed=0,
        deterministic=True,
        project=str(ROOT / "runs" / args.name),
        name=stamp,
        plots=True,
    )

    best = Path(model.trainer.best)
    archive = ROOT / "models" / args.name / f"{stamp}.pt"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, archive)
    shutil.copy2(best, ROOT / "models" / f"{args.name}.pt")
    print(f"\nВеса: {archive}\nТекущая модель: {ROOT / 'models' / f'{args.name}.pt'}")

    import evaluate
    evaluate.main([str(archive), "--data", str(data)])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

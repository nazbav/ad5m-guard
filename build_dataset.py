"""Собирает чистый YOLO-датасет из выгрузок Roboflow.

Что делает по сравнению со старым train.py:
  * полигоны переводит в описанные прямоугольники (раньше из полигона брались
    первые 4 числа как xc yc w h — получались случайные рамки);
  * переводит классы по конфигу модели (config/defects.yaml, config/scene.yaml);
  * убирает точные дубли картинок;
  * делит на train/val/test целыми группами почти одинаковых картинок
    (отрезок одной печати, дубли одной фотографии) со стратификацией по классам;
  * даёт файлам короткие имена — длинные имена Roboflow не открывались в Windows;
  * пишет data.yaml с относительными путями, отчёт и manifest.csv.

Пример:
    python build_dataset.py data/raw/roboflow_v18_yolov11.zip                        # → data/defects
    python build_dataset.py data/raw/roboflow_v18_yolov11.zip --config config/scene.yaml  # → data/scene
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image

from ddet.config import CONFIG_DIR, DEFECTS_YAML, ROOT, load_classes, long_path
from ddet.frames import parse_timestamp, split_sessions
from ddet.holdout import Holdout
from ddet.imagehash import UnionFind, dhash, near_duplicate_pairs, pack
from ddet.labels import Box, merge_copies, parse_text, remap
from ddet.split import Group, stratified_group_split

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARKER = ".ddet-dataset"  # по нему понимаем, что папку создали мы и её можно пересобрать
SPLITS = ("train", "val", "test")


@dataclass
class Item:
    source: str               # откуда взята картинка (для manifest)
    data: bytes
    boxes: list[Box]
    taken: datetime | None    # время съёмки для кадров своей камеры
    sha1: str = ""
    name: str = ""
    group: str = ""
    split: str = ""
    polygons: int = 0
    session: str = ""         # начало печати ГГГГММДД_ччмм (для кадров своей камеры)

    @property
    def synthetic(self) -> bool:
        """Кадр с искусственным сбоем (tools/synth_failures.py) — имя начинается с synth_."""
        return PurePosixPath(self.source.split(" | ")[0].split(":", 1)[-1]).name.startswith("synth_")
    dropped: Counter = field(default_factory=Counter)

    @property
    def own(self) -> bool:
        return self.taken is not None


# ---------------------------------------------------------------- чтение
def _read_roboflow(path: Path):
    """Выгрузка Roboflow в формате YOLO: zip или распакованная папка.
    Отдаёт (имя в источнике, байты картинки, текст разметки, имена классов)."""
    if path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        files = {PurePosixPath(n): n for n in z.namelist() if not n.endswith("/")}
        read = lambda p: z.read(files[p])  # noqa: E731
    else:
        files = {PurePosixPath(p.relative_to(path).as_posix()): p for p in path.rglob("*") if p.is_file()}
        read = lambda p: Path(long_path(files[p])).read_bytes()  # noqa: E731

    yamls = [p for p in files if p.name == "data.yaml"]
    if not yamls:
        raise SystemExit(f"{path}: нет data.yaml — это не выгрузка YOLO")
    names = yaml.safe_load(read(sorted(yamls, key=lambda p: len(p.parts))[0]))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    for p in sorted(files):
        if p.suffix.lower() not in IMAGE_EXT or "images" not in p.parts:
            continue
        parts = list(p.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        label = PurePosixPath(*parts).with_suffix(".txt")
        text = read(label).decode("utf-8") if label in files else ""
        yield f"{path.name}:{p}", read(p), text, names


def load_items(sources: list[Path], cfg) -> tuple[list[Item], Counter, list[str]]:
    """Читает источники. Копии одной картинки сводятся в одну (разметка объединяется);
    картинки, где копии противоречат друг другу, выбрасываются и возвращаются списком."""
    stats: Counter = Counter()
    copies: dict[str, list[Item]] = defaultdict(list)
    for src in sources:
        for source, data, text, src_names in _read_roboflow(src):
            stats["картинок в источниках"] += 1
            polygons = sum(1 for l in text.splitlines() if len(l.split()) > 5)
            raw = parse_text(text)
            boxes = remap(raw, src_names, cfg.roboflow_map, cfg.names)
            dropped = Counter(src_names[b.cls] for b in raw
                              if src_names[b.cls] in cfg.roboflow_map and cfg.roboflow_map[src_names[b.cls]] is None)
            sha1 = hashlib.sha1(data).hexdigest()
            copies[sha1].append(Item(source, data, boxes, parse_timestamp(PurePosixPath(source).name), sha1,
                                     polygons=polygons, dropped=dropped))
            stats["полигонов переведено в рамки"] += polygons
            stats.update({f"выброшено объектов {k}": v for k, v in dropped.items()})

    items: list[Item] = []
    conflicts: list[str] = []
    for group in copies.values():
        first = group[0]
        if len(group) > 1:
            stats["картинок, загруженных в Roboflow повторно"] += len(group) - 1
            first.boxes, conflict = merge_copies([it.boxes for it in group])
            first.source = " | ".join(it.source for it in group)
            if conflict:
                conflicts.append(first.source)
                continue
        items.append(first)
    stats["выброшено картинок с противоречивой разметкой"] = len(conflicts)
    return items, stats, conflicts


# ---------------------------------------------------------------- группы
def assign_groups(items: list[Item], session_gap: timedelta, block: timedelta, dup_distance: int) -> int:
    """Проставляет item.group. Возвращает число пар почти-дублей."""
    uf = UnionFind(len(items))

    # Кадры своей камеры: одна печать режется на отрезки по `block` минут.
    own = [(it.taken, i) for i, it in enumerate(items) if it.own]
    for session in split_sessions(own, session_gap):
        start = session[0][0]
        for _, i in session:
            items[i].session = f"{start:%Y%m%d_%H%M}"
        by_block: dict[int, list[int]] = defaultdict(list)
        for taken, i in session:
            by_block[int((taken - start) / block)].append(i)
        for members in by_block.values():
            for i in members[1:]:
                uf.union(members[0], i)

    # Почти одинаковые картинки — в одну группу. Пары «свой кадр — свой кадр»
    # пропускаем: камера неподвижна, и по цепочке похожих кадров в одну группу
    # слиплись бы все печати целиком. Их уже развели по отрезкам выше.
    hashes = [dhash(Image.open(io.BytesIO(it.data))) for it in items]
    pairs = 0
    for i, j in near_duplicate_pairs(pack(hashes), dup_distance):
        if items[i].own and items[j].own:
            continue
        uf.union(i, j)
        pairs += 1

    for i, it in enumerate(items):
        root = items[uf.find(i)]
        it.group = f"own:{root.taken:%Y%m%d_%H%M%S}" if root.own else f"ext:{root.sha1[:12]}"
    return pairs


# ---------------------------------------------------------------- запись
def name_items(items: list[Item]) -> None:
    used: set[str] = set()
    for it in items:
        if it.synthetic:
            base = f"synth_{it.taken:%Y%m%d_%H%M%S}"
        else:
            base = f"own_{it.taken:%Y%m%d_%H%M%S}" if it.own else f"ext_{it.sha1[:12]}"
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        it.name = name



def assign_splits(items: list[Item], groups: dict[str, Group], holdout: Holdout, previous: dict[str, str],
                  ratios: dict[str, float], seed: int, names: list[str], stats: Counter) -> dict[str, str]:
    """Группа → выборка.

    С отложенными печатями: их кадры — test целиком; остальные печати своей камеры делятся
    только на train/val; интернет и K1 Max — как обычно, с закреплением прошлого деления.
    Без отложенных печатей — закрепление прошлого деления для всех.
    Класс, который встречается меньше чем в трёх группах, — в train: разложить его по трём
    выборкам нельзя, а без train модель не увидит класс вовсе.
    """
    own_groups = {it.group for it in items if it.own}
    held = {it.group for it in items if it.own and it.taken in holdout}
    fixed: dict[str, str] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        if it.name in previous and not (holdout and it.group in own_groups and previous[it.name] == "test"):
            votes[it.group][previous[it.name]] += 1
    for gid, v in votes.items():
        fixed[gid] = v.most_common(1)[0][0]
    # Искусственные кадры — только в train: по ним нельзя ни выбирать эпоху, ни проверять модель.
    synth_groups = {it.group for it in items if it.synthetic}
    if synth_groups & held:
        raise SystemExit("Синтетика собрана на кадрах отложенных печатей — так нельзя (config/holdout.yaml)")
    for gid in synth_groups:
        fixed[gid] = "train"

    groups_of: dict = defaultdict(set)
    for g in groups.values():
        for key in g.classes:
            if key[1] != "фон":
                groups_of[key].add(g.id)
    for key, gids in groups_of.items():
        free = gids - held
        if len(gids) < 3 and free:
            for gid in free:
                fixed[gid] = "train"
            stats[f"класс «{names[key[1]]}» ({key[0]}, групп: {len(gids)}) — в train"] = len(free)

    if not holdout:
        return stratified_group_split(list(groups.values()), ratios, seed, fixed=fixed)
    own_free = [groups[g] for g in own_groups - held]
    ext = [g for gid, g in groups.items() if gid not in own_groups]
    tv = {"train": ratios["train"], "val": ratios["val"]}
    split_of = stratified_group_split(own_free, tv, seed,
                                      fixed={g: s for g, s in fixed.items() if g in own_groups and s in tv})
    split_of |= stratified_group_split(ext, ratios, seed, fixed={g: s for g, s in fixed.items() if g not in own_groups})
    split_of |= {g: "test" for g in held}
    return split_of


def read_previous_splits(manifest: Path) -> dict[str, str]:
    """Имя картинки (без расширения) → выборка из manifest.csv прошлой сборки."""
    if not manifest.exists():
        return {}
    with open(manifest, encoding="utf-8") as f:
        return {r["file"].rsplit(".", 1)[0]: r["split"] for r in csv.DictReader(f)}


def prepare_out_dir(out: Path) -> None:
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} существует и создан не этим скриптом — не трогаю. Укажите другую папку.")
        shutil.rmtree(out)
    for s in SPLITS:
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    (out / MARKER).write_text("Папка собрана build_dataset.py и пересоздаётся при каждом запуске.\n", encoding="utf-8")


def write_dataset(items: list[Item], out: Path, names: list[str], sources: list[Path],
                  own_repeat: int = 1, val_own: bool = False) -> None:
    for it in items:
        img = Image.open(io.BytesIO(it.data))
        ext = ".jpg" if img.format == "JPEG" else ".png"
        if img.format in ("JPEG", "PNG"):
            (out / it.split / "images" / f"{it.name}{ext}").write_bytes(it.data)
        else:
            img.convert("RGB").save(out / it.split / "images" / f"{it.name}.png")
            ext = ".png"
        it.name += ext
        label = "".join(b.to_line() + "\n" for b in it.boxes)
        (out / it.split / "labels" / f"{Path(it.name).stem}.txt").write_text(label, encoding="utf-8")

    data = {
        # path не указываем: Ultralytics тогда считает корнем папку с data.yaml,
        # и датасет можно переносить куда угодно.
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }

    # Отдельные списки кадров своей камеры: метрики на них — главные.
    for s in ("val", "test"):
        own = [f"./{s}/images/{it.name}" for it in items if it.split == s and it.own]
        (out / f"{s}_own.txt").write_text("\n".join(own) + "\n", encoding="utf-8")

    if own_repeat > 1:
        # Кадров своей камеры мало на фоне интернетных фото — показываем их модели чаще.
        train = [f"./train/images/{it.name}" for it in items if it.split == "train"
                 for _ in range(own_repeat if it.own else 1)]
        (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        data["train"] = "train.txt"
    if val_own:
        # Лучшую эпоху выбирать по кадрам своей камеры, а не по интернетным фото.
        data["val"] = "val_own.txt"

    header = "# Собрано build_dataset.py из: " + ", ".join(s.name for s in sources) + "\n"
    (out / "data.yaml").write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "group", "domain", "objects", "source"])
        for it in sorted(items, key=lambda x: (x.split, x.name)):
            objs = " ".join(f"{names[c]}:{k}" for c, k in sorted(Counter(b.cls for b in it.boxes).items()))
            w.writerow([it.name, it.split, it.group, "own" if it.own else "ext", objs, it.source])


def report(items: list[Item], names: list[str], stats: Counter, pairs: int, n_groups: int,
           conflicts: list[str]) -> str:
    lines = ["# Отчёт о сборке датасета", ""]
    lines += [f"- {k}: {v}" for k, v in stats.items()]
    lines += [f"- пар почти-дублей: {pairs}", f"- групп: {n_groups}", ""]
    if conflicts:
        lines += ["## Переразметить в Roboflow", "",
                  "Одна и та же картинка загружена несколько раз, и одну область копии называют",
                  "разными классами. В датасет эти картинки не вошли.", ""]
        lines += [f"- {c}" for c in conflicts] + [""]

    def table(title: str, subset: list[Item]) -> None:
        nonlocal lines
        per = {s: Counter() for s in SPLITS}
        imgs = {s: Counter() for s in SPLITS}
        for it in subset:
            per[it.split].update(b.cls for b in it.boxes)
            imgs[it.split]["картинок"] += 1
            imgs[it.split]["без брака (фон)"] += not it.boxes
        lines += [f"## {title}", "", "| класс | " + " | ".join(SPLITS) + " |", "|---|" + "---:|" * len(SPLITS)]
        for row in ("картинок", "без брака (фон)"):
            lines.append(f"| {row} | " + " | ".join(str(imgs[s][row]) for s in SPLITS) + " |")
        for c, n in enumerate(names):
            lines.append(f"| {n} | " + " | ".join(str(per[s][c]) for s in SPLITS) + " |")
        lines.append("")

    table("Все картинки (объекты по классам)", items)
    table("Кадры своей камеры", [it for it in items if it.own])
    table("Картинки из интернета", [it for it in items if not it.own])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path, help="выгрузки Roboflow в формате YOLO (zip или папка)")
    ap.add_argument("--config", type=Path, default=DEFECTS_YAML, help="конфиг модели: классы и откуда их брать")
    ap.add_argument("--out", type=Path, help="куда собрать (по умолчанию data/<имя конфига>)")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--session-gap-min", type=float, default=20, help="разрыв между кадрами, после которого начинается новая печать")
    ap.add_argument("--block-min", type=float, default=15, help="длина отрезка печати, который не делится между выборками")
    ap.add_argument("--dup-distance", type=int, default=12, help="порог почти-дубля: расстояние Хэмминга из 256 бит")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=Path, default=CONFIG_DIR / "holdout.yaml",
                    help="отложенные печати своей камеры — целиком в test (config/holdout.yaml)")
    ap.add_argument("--no-pin", action="store_true",
                    help="поделить заново; по умолчанию картинки из прошлой сборки остаются в своих выборках")
    ap.add_argument("--own-repeat", type=int, default=1, help="повторить кадры своей камеры в train столько раз")
    ap.add_argument("--val-own", action="store_true", help="валидация (выбор лучшей эпохи) только по кадрам своей камеры")
    args = ap.parse_args(argv)

    # cmd.exe не раскрывает шаблоны вида data\raw\*.zip — раскрываем сами.
    args.sources = [Path(p) for s in args.sources for p in (sorted(glob.glob(str(s))) if "*" in str(s) else [s])]
    cfg = load_classes(args.config)
    args.out = args.out or ROOT / "data" / cfg.name
    items, stats, conflicts = load_items(args.sources, cfg)
    if cfg.only_own:
        stats["пропущено картинок не со своей камеры"] = sum(not it.own for it in items)
        items = [it for it in items if it.own]
    if not items:
        raise SystemExit("Картинок не найдено")
    pairs = assign_groups(items, timedelta(minutes=args.session_gap_min), timedelta(minutes=args.block_min), args.dup_distance)

    # Стратифицируем по паре (источник, класс): иначе кадры своей камеры —
    # а метрики на них главные — делятся как попало на фоне интернетных фото.
    groups: dict[str, Group] = {}
    for it in items:
        domain = "own" if it.own else "ext"
        g = groups.setdefault(it.group, Group(it.group, 0))
        g.n_images += 1
        g.classes.update((domain, b.cls) for b in it.boxes)
        if not it.boxes:
            g.classes[(domain, "фон")] += 1
    name_items(items)
    holdout = Holdout.load(args.holdout)
    previous = {} if args.no_pin else read_previous_splits(args.out / "manifest.csv")
    split_of = assign_splits(items, groups, holdout, previous, dict(zip(SPLITS, args.ratios)), args.seed,
                             cfg.names, stats)
    for it in items:
        it.split = split_of[it.group]
    if previous:
        moved = sum(1 for it in items if it.name in previous and previous[it.name] != it.split)
        stats["картинок из прошлой сборки"] = sum(it.name in previous for it in items)
        if moved:
            stats["из них сменили выборку"] = moved
    if holdout:
        stats["кадров своей камеры в отложенных печатях (test)"] = sum(it.own and it.taken in holdout for it in items)

    prepare_out_dir(args.out)
    write_dataset(items, args.out, cfg.names, args.sources, args.own_repeat, args.val_own)
    text = report(items, cfg.names, stats, pairs, len(groups), conflicts)
    (args.out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nГотово: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

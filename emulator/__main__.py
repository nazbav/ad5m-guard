"""Эмулятор парка AD5M с пультом управления.

    python -m emulator                                   # 2 стоковых + 1 Klipper, пульт http://127.0.0.1:8790/
    python -m emulator --flashforge 5 --moonraker 2 --speed 10 --frames build/demo

Принтеры: 127.0.0.2, 127.0.0.3 … на стандартных портах (8898/8899, 7125, камера 8080).
Серийники SNEMU0001…, код доступа 12345678. В AD5M Guard добавляются как настоящие: по IP.
Пульт: запустить печать (чистую или со срывом), пауза/отмена «с принтера», неисправности —
принтер пропал из сети, камера отвалилась, выключен режим «Только LAN», ошибка прошивки, задержки.
"""
from __future__ import annotations

import argparse
import random
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from emulator.fleet import load_frames, timeline
from emulator.servers import PrinterNode, _Http
from emulator.virtual import VirtualPrinter, synthetic_frames

PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8><title>Эмулятор AD5M</title>
<style>body{font:14px 'Segoe UI',sans-serif;margin:16px;background:#f4f5f7}table{border-collapse:collapse;background:#fff}
td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}button{margin:1px;padding:3px 8px;cursor:pointer}.on{background:#fbd5d3}</style></head>
<body><h2>Эмулятор AD5M</h2><p>Добавляйте принтеры в AD5M Guard по IP. Код доступа — <b>12345678</b>.</p>
<table id=t></table><script>
async function act(i,a,v){await fetch(`/api/${i}/${a}`,{method:'POST',body:JSON.stringify(v??null)});load()}
async function load(){const s=await (await fetch('/api/state')).json();
t.innerHTML='<tr><th>Принтер</th><th>IP</th><th>Протокол</th><th>SN</th><th>Статус</th><th>Прогресс</th><th>Печать</th><th>С принтера</th><th>Неисправности</th></tr>'+
s.map((p,i)=>`<tr><td>${p.name}</td><td>${p.host}</td><td>${p.kind}</td><td>${p.serial}</td><td>${p.status}</td><td>${Math.round(p.progress*100)}%</td>
<td><button onclick="act(${i},'start','clean')">чистая</button><button onclick="act(${i},'start','fail')">со срывом</button></td>
<td><button onclick="act(${i},'control','pause')">пауза</button><button onclick="act(${i},'control','resume')">продолжить</button><button onclick="act(${i},'control','cancel')">отмена</button></td>
<td>${['offline','camera_down','lan_off','fw_error','slow'].map(f=>`<button class="${p.faults[f]?'on':''}" onclick="act(${i},'fault','${f}')">${f}</button>`).join('')}</td></tr>`).join('')}
load();setInterval(load,2000)</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flashforge", type=int, default=2)
    ap.add_argument("--moonraker", type=int, default=1)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--frames", type=Path, default=Path("build/demo"), help="папка с clean*/ и spaghetti/ (make_demo.py)")
    ap.add_argument("--panel-port", type=int, default=8790)
    args = ap.parse_args(argv)

    # у каждого принтера своя запись чистой печати (clean, clean2, clean3 — по кругу); «со срывом» — его же
    # начало печати, затем настоящий сбой (ком нити)
    spag = load_frames(args.frames / "spaghetti")
    cleans = [c for c in (load_frames(args.frames / n) for n in ("clean", "clean2", "clean3")) if c]

    def clips_for(k: int) -> dict:
        clean = cleans[k % len(cleans)] if cleans else []
        return {"clean": timeline(clean) if clean else synthetic_frames(120, 10.0),
                "fail": timeline(clean[:30] + spag) if clean and spag else synthetic_frames(120, 10.0, bad_from=60)}
    clips = []
    nodes = []
    kinds = ["flashforge"] * args.flashforge + ["moonraker"] * args.moonraker
    rng = random.Random(5)                   # одинаковые данные при каждом запуске
    for k, kind in enumerate(kinds):
        # как настоящие: имя из коробки, серийник как на экране принтера — 5M: SNMQLD…, 5M Pro: SNMOME…
        pro = k % 2 == 1
        model = "Adventurer 5M Pro" if pro else "Adventurer 5M"
        serial = ("SNMOME96" if pro else "SNMQLD91") + f"{rng.randrange(10**5):05d}"
        mac = "88:A9:A7:" + ":".join(f"{rng.randrange(256):02X}" for _ in range(3))
        name = model if kind == "flashforge" else f"AD5M-ZMOD-{k + 1}"
        petg = pro
        p = VirtualPrinter(name, serial, "12345678", kind=kind, speed=args.speed, heat_s=20, model=model,
                           firmware=rng.choice(["3.1.3", "3.1.5", "2.7.9"]), mac=mac,
                           nozzle_c=240.0 if petg else 220.0, bed_c=85.0 if petg else 60.0)
        clips.append(clips_for(k))
        p.start(clips[k]["clean"], "Крышка 2_PETG_4h24m.gcode" if petg else f"Корпус_{k + 1}_PLA.gcode", 250)
        nodes.append(PrinterNode(p, f"127.0.0.{k + 2}").start())
        print(f"{p.name}: {kind:10} http://127.0.0.{k + 2}  SN {p.serial}")

    def view(n: PrinterNode) -> dict:
        p, f = n.printer, n.printer.faults
        s = p.snapshot_state()
        return {"name": p.name, "host": n.host, "kind": p.kind, "serial": p.serial, "status": s["status"],
                "progress": s["progress"], "faults": {"offline": f.offline, "camera_down": f.camera_down,
                                                      "lan_off": not f.lan_only, "fw_error": bool(f.firmware_error),
                                                      "slow": f.latency_s > 0}}

    class Panel(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/api/state":
                return self._send(json.dumps([view(n) for n in nodes]).encode(), "application/json")
            self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            _, _, i, action = urlparse(self.path).path.split("/")
            arg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            p = nodes[int(i)].printer
            if action == "start":
                p.start(clips[int(i)][arg], p.file, 250)          # тот же файл: со срывом или нет — видно только в пульте
            elif action == "control":
                p.control(arg)
            elif action == "fault":
                f = p.faults
                if arg == "offline":
                    f.offline = not f.offline
                elif arg == "camera_down":
                    f.camera_down = not f.camera_down
                elif arg == "lan_off":
                    f.lan_only = not f.lan_only
                elif arg == "fw_error":
                    f.firmware_error = "" if f.firmware_error else "E0017"
                elif arg == "slow":
                    f.latency_s = 0.0 if f.latency_s else 3.0
            self._send(b"{}", "application/json")

    panel = _Http(("127.0.0.1", args.panel_port), Panel)
    print(f"\nПульт эмулятора: http://127.0.0.1:{args.panel_port}/   Ctrl+C — стоп")
    try:
        panel.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            n.stop()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

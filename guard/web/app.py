"""REST API и панель. Только перевод HTTP ↔ вызовы FleetService, без логики.

Если задан токен, всё, что меняет состояние (управление, принтеры, настройки), требует его
в заголовке X-Token. Просмотр — без токена.
"""
from __future__ import annotations

import hmac
import logging
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request

from guard import __version__
from guard.domain.policy import Settings
from guard.ports import AuthError, PrinterError
from guard.services.fleet import FleetService

log = logging.getLogger("guard.web")
STATIC = Path(__file__).with_name("static")
CLASSES = ["spaghetti", "detached", "warping", "cracks", "stringing", "garbage",
           "plate_not_visible", "glass_plate", "pei_misplaced", "head_no_cover"]


def create_app(fleet: FleetService, token: str | None = None, discovery=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    if discovery is None:
        from guard.services.discovery import DiscoveryService
        discovery = DiscoveryService(lambda: {w.cfg.host for w in list(fleet.workers.values())})

    def guard_write():
        if token and not hmac.compare_digest(request.headers.get("X-Token", ""), token):
            abort(Response('{"error": "нужен токен панели"}', 403, mimetype="application/json"))

    @app.errorhandler(ValueError)
    def bad(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(KeyError)
    def missing(e):
        return jsonify(error=str(e).strip("'")), 404

    @app.errorhandler(AuthError)
    def auth(e):
        return jsonify(error=str(e), kind="auth"), 422

    @app.errorhandler(PrinterError)
    def printer(e):
        return jsonify(error=str(e), kind="printer"), 502

    @app.get("/")
    def index():
        # без кэша: после обновления программы браузер не должен показывать старую панель
        return Response((STATIC / "index.html").read_text(encoding="utf-8"), mimetype="text/html",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return jsonify(fleet.view() | {"version": __version__, "classes": CLASSES, "token_required": bool(token)})

    @app.get("/api/frame/<pid>.jpg")
    def frame(pid):
        jpg = fleet.frame(pid)
        if jpg is None:
            abort(404)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/events")
    def events():
        limit = min(int(request.args.get("limit", 100)), 1000)
        return jsonify([e.__dict__ for e in fleet.events.recent(limit, request.args.get("printer_id") or None)])

    @app.get("/api/events/image/<name>")
    def event_image(name):
        jpg = fleet.events.image(name)
        if jpg is None:
            abort(404)
        return Response(jpg, mimetype="image/jpeg")

    @app.get("/api/discovery")
    def discovery_status():
        return jsonify(discovery.status())

    @app.post("/api/discovery")
    def discovery_start():
        guard_write()
        return jsonify(discovery.start((request.get_json(silent=True) or {}).get("subnet") or None))

    @app.get("/api/training")
    def training_stats():
        days = int(request.args.get("days", 30))
        samples = fleet.events.training_samples(days) if hasattr(fleet.events, "training_samples") else []
        size = sum(f.stat().st_size for d in samples for f in d.iterdir())
        return jsonify(days=days, alerts=len(samples), mb=round(size / 2**20, 1))

    @app.get("/api/training/export")
    def training_export():
        """Архив исходных кадров тревог — пользователь отдаёт его разработчикам для дообучения."""
        import tempfile
        from flask import send_file
        days = min(max(int(request.args.get("days", 30)), 1), 3650)
        tmp = Path(tempfile.gettempdir()) / f"ad5m-guard-training-{days}d.zip"
        fleet.events.export_training(tmp, days, {"app": f"AD5M Guard {__version__}",
                                                  "calibration": fleet.settings.calibration,
                                                  "printers": [w.cfg.name for w in fleet.workers.values()]})
        name = f"AD5M-Guard-training-{time.strftime('%Y-%m-%d')}-{days}d.zip"
        return send_file(tmp, as_attachment=True, download_name=name, mimetype="application/zip")

    @app.post("/api/printers/test")
    def test_printer():
        guard_write()
        info, cfg = fleet.test_connection(request.get_json(force=True) or {})
        return jsonify(protocol=info.protocol.value, model=info.model, name=info.name, serial=cfg.serial or info.serial,
                       firmware=info.firmware, camera_url=info.camera_url)

    @app.post("/api/printers")
    def add_printer():
        guard_write()
        return jsonify(fleet.add_printer(request.get_json(force=True) or {}).public()), 201

    @app.put("/api/printers/<pid>")
    def update_printer(pid):
        guard_write()
        return jsonify(fleet.update_printer(pid, request.get_json(force=True) or {}).public())

    @app.delete("/api/printers/<pid>")
    def remove_printer(pid):
        guard_write()
        fleet.remove_printer(pid)
        return jsonify(ok=True)

    @app.post("/api/printers/<pid>/control/<action>")
    def control(pid, action):
        guard_write()
        fleet.control(pid, action)
        return jsonify(ok=True)

    @app.post("/api/printers/<pid>/snapshot")
    def training_snapshot(pid):
        guard_write()
        return jsonify(saved=fleet.training_snapshot(pid, (request.get_json(silent=True) or {}).get("note") or ""))

    @app.post("/api/printers/<pid>/light/<onoff>")
    def light(pid, onoff):
        guard_write()
        fleet.set_light(pid, onoff == "on")
        return jsonify(ok=True)

    @app.get("/api/settings")
    def get_settings():
        return jsonify(fleet.settings.to_dict(secrets=False))

    @app.get("/api/settings/defaults")
    def default_settings():
        """Рекомендуемые значения распознавания (калибровка этой версии) — без Telegram и прочего личного."""
        d = Settings.recommended().to_dict(secrets=False)
        return jsonify({k: d[k] for k in ("action", "defect_conf", "scene_conf", "window", "min_hits", "cooldown_s",
                                          "hand_hold_s", "spatial", "spatial_iou", "check_plate", "stop_on",
                                          "notify_on", "muted")})

    @app.put("/api/settings")
    def put_settings():
        guard_write()
        return jsonify(fleet.update_settings(request.get_json(force=True) or {}).to_dict(secrets=False))

    return app

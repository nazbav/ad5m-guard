"""Поиск принтеров в сети как фоновое задание: панель запускает и спрашивает состояние, не ждёт."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger("guard.discovery")


class DiscoveryService:
    def __init__(self, known_hosts: Callable[[], set[str]], scanner=None, subnets=None):
        from guard.adapters import discovery
        self.known_hosts = known_hosts
        self.scanner = scanner or discovery.scan
        self.subnets = subnets or discovery.default_subnets
        self.lock = threading.Lock()
        self._state = {"running": False, "done": 0, "total": 0, "subnets": [], "found": [], "error": None,
                       "finished_at": None}

    def start(self, subnet: str | None = None) -> dict:
        from guard.adapters.discovery import parse_subnet
        with self.lock:
            if self._state["running"]:
                return self.status()
            nets = [parse_subnet(subnet)] if subnet else self.subnets()      # ValueError → 400
            if not nets:
                raise ValueError("не нашёл подключённой локальной сети — укажите подсеть, например 192.168.1.0/24")
            hosts = [str(h) for n in nets for h in (n.hosts() if n.num_addresses > 2 else n)]
            self._state = {"running": True, "done": 0, "total": len(hosts), "subnets": [str(n) for n in nets],
                           "found": [], "error": None, "finished_at": None}
        threading.Thread(target=self._run, args=(hosts,), daemon=True, name="discovery").start()
        return self.status()

    def _run(self, hosts: list[str]) -> None:
        def progress(done: int, total: int) -> None:
            self._state["done"] = done
        try:
            found = self.scanner(hosts, progress=progress)
            log.info("поиск принтеров: %s — найдено %d", ", ".join(self._state["subnets"]), len(found))
            self._state.update(found=found)
        except Exception as e:                       # noqa: BLE001 — показать в панели, а не уронить поток
            log.exception("поиск принтеров не удался")
            self._state.update(error=str(e))
        finally:
            self._state.update(running=False, finished_at=time.strftime("%H:%M:%S"))

    def status(self) -> dict:
        """Состояние для панели; уже добавленные принтеры в «найденных» не показываются."""
        known = self.known_hosts()
        s = dict(self._state)
        s["found"] = [f for f in s["found"] if f["host"] not in known]
        s["already_added"] = len(self._state["found"]) - len(s["found"])
        return s

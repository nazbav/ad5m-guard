"""Уведомления в Telegram (бот + чат из настроек). Отправка в фоне — цикл опроса не ждёт сеть."""
from __future__ import annotations

import logging
import queue
import threading

import requests

from guard.domain.models import Event

log = logging.getLogger("guard.notify")
ICONS = {"alert": "⚠️", "stopped": "⛔", "error": "❗"}


class TelegramNotifier:
    def __init__(self, get_settings):
        self.get_settings = get_settings          # функция: актуальные настройки (токен меняют в панели)
        self.q: queue.Queue = queue.Queue(maxsize=200)
        threading.Thread(target=self._loop, daemon=True, name="telegram").start()

    def notify(self, event: Event, jpg: bytes | None) -> None:
        s = self.get_settings()
        if not (s.telegram_token and s.telegram_chat_id):
            return
        try:
            self.q.put_nowait((s.telegram_token, s.telegram_chat_id, event, jpg))
        except queue.Full:
            log.warning("очередь Telegram переполнена — уведомление пропущено")

    def _loop(self) -> None:
        while True:
            token, chat, e, jpg = self.q.get()
            text = f"{ICONS.get(e.kind, '')} {e.printer}: {e.detail}"
            base = f"https://api.telegram.org/bot{token}"
            try:
                if jpg:
                    requests.post(f"{base}/sendPhoto", data={"chat_id": chat, "caption": text},
                                  files={"photo": ("alert.jpg", jpg, "image/jpeg")}, timeout=15)
                else:
                    requests.post(f"{base}/sendMessage", data={"chat_id": chat, "text": text}, timeout=15)
            except requests.RequestException as ex:
                log.warning("Telegram недоступен: %s", type(ex).__name__)

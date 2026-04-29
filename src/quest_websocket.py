import json
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

import websocket

class QuestWebSocketClient:
    """Minimal websocket client for Quest controller haptics."""

    def __init__(self, url: str, ws_log: Optional[Callable[[str], None]] = None):
        self.url = url
        self.ws: Optional["websocket.WebSocketApp"] = None
        self._thread: Optional[threading.Thread] = None
        self.on_message_callback: Optional[Callable[[dict], None]] = None
        self.ws_log = ws_log
        self._connect()

    # connection helpers -------------------------------------------------
    def _create_ws(self) -> "websocket.WebSocketApp":
        return websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
        )

    def _connect(self) -> None:
        self.ws = self._create_ws()
        self._thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self._thread.start()

    # websocket callbacks -------------------------------------------------
    def _on_open(self, ws):
        if self.ws_log: self.ws_log("[DEBUG] WebSocket connection opened")
        self.send_json({"cmd": "getinfo"})

    def _on_message(self, ws, message: str):
        try:
            msg = json.loads(message)
            # if self.ws_log: self.ws_log(f"[DEBUG] Received message: {msg}")
        except Exception as e:  # pragma: no cover - runtime feedback
            if self.ws_log: self.ws_log(f"[WARN] invalid message {e}")
            return
        if self.on_message_callback:
            self.on_message_callback(msg)

    def _on_close(self, ws, code, reason):  # pragma: no cover - runtime feedback
        if self.ws_log: self.ws_log("[INFO] Quest websocket closed; reconnecting in 2s")
        time.sleep(2)
        self._connect()

    # public api ---------------------------------------------------------
    def register_on_message(self, cb: Callable[[dict], None]) -> None:
        self.on_message_callback = cb

    def send_json(self, data: dict) -> None:
        if self.ws_log: self.ws_log(f"[DEBUG] Sending JSON data: {data}")
        if self.ws:
            self.ws.send(json.dumps(data))

    def send_pcm_signal(self, pcm: np.ndarray) -> None:
        if self.ws is None or not self.ws.sock or not self.ws.sock.connected:
            return
        # if self.ws_log: self.ws_log(f"[DEBUG] Sending PCM signal of shape {pcm.shape}")
        buf = pcm.astype(np.float32).tobytes()
        self.ws.send(buf, opcode=websocket.ABNF.OPCODE_BINARY)

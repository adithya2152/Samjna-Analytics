from fastapi import WebSocket
from typing import Dict
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.loop = asyncio.get_event_loop()

    async def connect(self, websocket: WebSocket, uid: str):
        await websocket.accept()
        self.active_connections[uid] = websocket

    def disconnect(self, uid: str):
        self.active_connections.pop(uid, None)

    async def _send_log_async(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)

    def send_log_background(self, uid: str, message: str):
        if uid in self.active_connections:
            websocket = self.active_connections[uid]
            asyncio.run_coroutine_threadsafe(
                self._send_log_async(websocket, message),
                self.loop
            )

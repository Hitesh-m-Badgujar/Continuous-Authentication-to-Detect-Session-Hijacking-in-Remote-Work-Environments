from __future__ import annotations
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asyncio import Lock

# Reuse your scorer from ae_conditional
from Apps.behavior.ae_conditional import RuntimeScorer

_SCORER = None
_SCORER_LOCK = Lock()

async def _get_scorer():
    global _SCORER
    if _SCORER is None:
        async with _SCORER_LOCK:
            if _SCORER is None:
                # model_dir is auto-detected in RuntimeScorer if not passed; adjust if you need a custom path
                _SCORER = RuntimeScorer()
    return _SCORER

class KeystrokeConsumer(AsyncWebsocketConsumer):
    """
    Simple per-connection consumer: accepts JSON messages shaped like the HTTP endpoint:
      { "mode": "global" | "per-user", "features": [..18..], "user_id": "s017" }
    Replies with the same scoring JSON the HTTP view returns.
    Maintains a rolling trust history per connection to compute smoothed trust.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hist = []

    async def connect(self):
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        try:
            msg = json.loads(text_data or "{}")
            feats = msg.get("features", None)
            mode = msg.get("mode", "global")
            user_id = msg.get("user_id", None)

            if not isinstance(feats, list):
                await self.send_json({"ok": False, "error": "bad_features"})
                return

            scorer = await _get_scorer()
            res = scorer.score(features=feats, mode=mode, user_id=user_id, rolling_hist=self._hist)

            # tack on channel indicator
            res["backend"] = "ws"
            await self.send_json(res)

        except Exception as e:
            await self.send_json({"ok": False, "error": "ws_exception", "detail": str(e)})

    async def disconnect(self, close_code):
        # nothing to clean
        return

    async def send_json(self, data: dict):
        await self.send(text_data=json.dumps(data))

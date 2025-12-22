from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"^ws/behavior/stream$", consumers.KeystrokeConsumer.as_asgi()),
]

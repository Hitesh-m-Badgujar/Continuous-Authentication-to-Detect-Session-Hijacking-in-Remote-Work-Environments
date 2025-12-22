import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django_asgi_app = get_asgi_application()

# Channels is optional; if not present, we still serve HTTP.
try:
    from channels.routing import ProtocolTypeRouter, URLRouter
    import Apps.behavior.routing

    application = ProtocolTypeRouter({
        "http": django_asgi_app,
        "websocket": URLRouter(Apps.behavior.routing.websocket_urlpatterns),
    })
except Exception:
    # Fallback: HTTP only (timer-based polling will still work)
    application = django_asgi_app

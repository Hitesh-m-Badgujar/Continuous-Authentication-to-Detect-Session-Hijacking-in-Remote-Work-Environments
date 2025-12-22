# Apps/behavior/urls.py
from django.urls import path
from . import views

app_name = "behavior"

urlpatterns = [
    # Redirect /behavior/ → monitor page
    path("", views.index, name="index"),

    # Monitoring UI
    path("monitor_page/", views.monitor_page, name="monitor_page"),

    # Health endpoints
    path("kb_health", views.kb_health, name="kb_health"),
    path("mouse_health", views.mouse_health, name="mouse_health"),

    # Streaming endpoints
    path("stream_keystrokes", views.stream_keystrokes, name="stream_keystrokes"),
    path("stream_mouse", views.stream_mouse, name="stream_mouse"),

    # Fusion endpoint
    path("fuse_scores", views.fuse_scores, name="fuse_scores"),

    # Face endpoints (THIS is what you were missing)
    path("face_enroll", views.face_enroll, name="face_enroll"),
    path("face_score", views.face_score, name="face_score"),
]

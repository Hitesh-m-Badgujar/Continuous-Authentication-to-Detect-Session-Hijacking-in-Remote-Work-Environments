# H1/core/urls.py
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path("admin/", admin.site.urls),
    path("behavior/", include("Apps.behavior.urls")),
    path("", lambda request: redirect("/behavior/monitor_page/")),
]

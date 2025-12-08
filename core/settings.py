from pathlib import Path
import os
import importlib.util as _ilu

# =============================================================================
# Base paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# Security / env (safe defaults for local dev)
# =============================================================================
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-secret-key")
DEBUG = os.environ.get("DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000"
).split(",")

# =============================================================================
# Applications
# =============================================================================
INSTALLED_APPS = [
    # Django core
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # 3rd-party
    'rest_framework',

    # Project apps
    'Apps.behavior',
]

# Make Channels optional: if installed, enable WebSocket support; if not, keep HTTP-only.
_ASGI_HAS_CHANNELS = _ilu.find_spec("channels") is not None
if _ASGI_HAS_CHANNELS:
    INSTALLED_APPS.append('channels')

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

# =============================================================================
# Templates
# =============================================================================
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # Global templates dir: H1/templates/
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI is always present. ASGI is also set; Channels routing is added in core/asgi.py if available.
WSGI_APPLICATION = 'core.wsgi.application'
ASGI_APPLICATION = 'core.asgi.application'

# If Channels is actually installed, provide an in-memory layer for dev.
if _ASGI_HAS_CHANNELS:
    CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    }

# =============================================================================
# Database (SQLite for dev)
# =============================================================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# =============================================================================
# Internationalization
# =============================================================================
LANGUAGE_CODE = 'en-gb'
TIME_ZONE = 'Europe/London'
USE_I18N = True
USE_TZ = True

# =============================================================================
# Static & media
# =============================================================================
STATIC_URL = '/static/'

# Only add the project static dir if it exists, so Django won’t warn every boot.
_PROJECT_STATIC = BASE_DIR / 'static'
STATICFILES_DIRS = [_PROJECT_STATIC] if _PROJECT_STATIC.exists() else []

STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# =============================================================================
# Project paths for ML artifacts and datasets
# =============================================================================
# Your code uses Data/ and Models/ — keep that consistent.
DATA_DIR = BASE_DIR / 'Data'
MODELS_DIR = BASE_DIR / 'Models'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

for _p in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Ignore creation errors in restricted environments.
        pass

# Optional convenience path used elsewhere in your project:
DATA_CLEANED = DATA_DIR / "events_clean.parquet"

# =============================================================================
# Django REST Framework
# =============================================================================
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

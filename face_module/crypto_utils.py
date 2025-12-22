import os
from typing import Optional

from cryptography.fernet import Fernet


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "Models")
SECRETS_DIR = os.path.join(MODELS_DIR, "secrets")

os.makedirs(SECRETS_DIR, exist_ok=True)

FERNET_KEY_PATH = os.path.join(SECRETS_DIR, "fernet.key")


def get_fernet() -> Fernet:
    """
    Load or create a symmetric Fernet key and return a Fernet instance.

    Key is stored in Models/secrets/fernet.key.
    """
    if os.path.exists(FERNET_KEY_PATH):
        with open(FERNET_KEY_PATH, "rb") as f:
            key = f.read().strip()
    else:
        key = Fernet.generate_key()
        with open(FERNET_KEY_PATH, "wb") as f:
            f.write(key)
    return Fernet(key)


def encrypt_bytes(data: bytes, f: Optional[Fernet] = None) -> bytes:
    if f is None:
        f = get_fernet()
    return f.encrypt(data)


def decrypt_bytes(token: bytes, f: Optional[Fernet] = None) -> bytes:
    if f is None:
        f = get_fernet()
    return f.decrypt(token)

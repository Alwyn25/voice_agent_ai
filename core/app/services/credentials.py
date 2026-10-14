"""At-rest encryption for tool_connections.credentials_encrypted /
integrations.credentials_encrypted (Vocal Backend Tasksheet P1).

Symmetric (Fernet) rather than one-way hashing, unlike ApiKey.key_hash — a stored
credential has to be decryptable again to actually call the provider's API later.
"""
import json

from cryptography.fernet import Fernet

from app.config import get_settings


class CredentialsEncryptionError(RuntimeError):
    pass


def _fernet() -> Fernet:
    key = get_settings().credentials_encryption_key
    if not key:
        raise CredentialsEncryptionError(
            "CREDENTIALS_ENCRYPTION_KEY is not configured — generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and set it in .env before connecting any tool.")
    return Fernet(key.encode())


def encrypt_credentials(data: dict) -> str:
    return _fernet().encrypt(json.dumps(data).encode()).decode()


def decrypt_credentials(token: str) -> dict:
    return json.loads(_fernet().decrypt(token.encode()).decode())

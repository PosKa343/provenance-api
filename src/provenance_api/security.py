"""Password hashing, access tokens and API keys. No database or HTTP code here."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

JWT_ALGORITHM = "HS256"
API_KEY_PREFIX = "pvk_"
API_KEY_DISPLAY_CHARS = 12

_password_hasher = PasswordHasher()
# Verified against when the email is unknown, so a missing account takes as
# long to reject as a wrong password and timing does not reveal who has one.
_DUMMY_PASSWORD_HASH = _password_hasher.hash("not-a-real-password")


class InvalidTokenError(Exception):
    pass


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _password_hasher.verify(password_hash or _DUMMY_PASSWORD_HASH, password) and (
            password_hash is not None
        )
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(
    user_id: uuid.UUID, *, secret: str, lifetime: timedelta, now: datetime | None = None
) -> str:
    issued_at = now or datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": issued_at, "exp": issued_at + lifetime}
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, *, secret: str) -> uuid.UUID:
    """Return the user id in a valid token, or raise InvalidTokenError."""
    try:
        # Naming the one accepted algorithm stops a forged token from choosing its own.
        claims = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )
        return uuid.UUID(claims["sub"])
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise InvalidTokenError from exc


@dataclass(frozen=True)
class NewApiKey:
    plaintext: str
    prefix: str
    key_hash: str


def generate_api_key() -> NewApiKey:
    plaintext = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return NewApiKey(
        plaintext=plaintext,
        prefix=plaintext[:API_KEY_DISPLAY_CHARS],
        key_hash=hash_api_key(plaintext),
    )


def hash_api_key(plaintext: str) -> str:
    # A key is 256 random bits, so it cannot be guessed the way a password can.
    # That makes a fast hash safe here, and it has to be fast: it runs on every request.
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

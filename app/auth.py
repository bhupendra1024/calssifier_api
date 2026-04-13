from __future__ import annotations

import datetime
import jwt
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app import config

auth_scheme = APIKeyHeader(name="authorization")


def f_decrypt_authorize_token(authorization_token: str) -> dict:
    try:
        if not authorization_token:
            return {"status": "ERROR", "user_id": "", "timestamp": ""}

        payload = jwt.decode(
            authorization_token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
        user_id = payload.get("user_id")

        return {
            "status": "SUCCESS",
            "user_id": user_id,
            "timestamp": str(payload.get("iat")),
        }

    except jwt.ExpiredSignatureError:
        return {"status": "ERROR", "user_id": "", "timestamp": ""}

    except jwt.InvalidTokenError:
        return {"status": "ERROR", "user_id": "", "timestamp": ""}


def f_validate_authorize_token(authorization_token: str) -> tuple[str, str | None]:
    try:
        if not authorization_token:
            return "ERROR", None

        token_data = f_decrypt_authorize_token(authorization_token)
        if token_data.get("status") != "SUCCESS":
            return "ERROR", None

        token_timestamp = token_data.get("timestamp")
        if not token_timestamp:
            return "ERROR", None

        token_time = datetime.datetime.utcfromtimestamp(int(token_timestamp))
        current_time = datetime.datetime.utcnow()
        timeout_limit = current_time - datetime.timedelta(
            seconds=config.TOKEN_EXPIRY_LIMIT
        )
        print(f"Token time: {token_time}, Current time: {current_time}, Timeout limit: {timeout_limit}")
        if token_time >= timeout_limit:
            return "SUCCESS", token_data.get("user_id")

        return "ERROR", None

    except Exception:
        return "ERROR", None


def get_current_user(authorization: str = Security(auth_scheme)) -> str:
    status, user_id = f_validate_authorize_token(authorization)
    if status != "SUCCESS" or not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


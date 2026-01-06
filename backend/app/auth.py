import os
import json
import time
from dataclasses import dataclass
from typing import Optional

import requests
import jwt
from fastapi import Header, HTTPException, Request


AUTH_MODE = os.getenv("AUTH_MODE", "either").strip().lower()  # api_key | entra | either
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()

ENTRA_TENANT_ID = os.getenv("ENTRA_TENANT_ID", "").strip()
ENTRA_API_CLIENT_ID = os.getenv("ENTRA_API_CLIENT_ID", "").strip()
ENTRA_REQUIRED_SCOPE = os.getenv("ENTRA_REQUIRED_SCOPE", "access_as_user").strip()

JWKS_TTL_SECONDS = int(os.getenv("JWKS_TTL_SECONDS", str(60 * 60)))  # 1 hour
JWKS_TIMEOUT_SECONDS = float(os.getenv("JWKS_TIMEOUT_SECONDS", "10"))


@dataclass
class AuthContext:
    auth_type: str  # "entra" | "api_key" | "none"
    oid: Optional[str] = None
    tid: Optional[str] = None
    upn: Optional[str] = None


_JWKS_CACHE: Optional[dict] = None
_JWKS_FETCHED_AT: float = 0.0


def _jwks_url() -> str:
    if not ENTRA_TENANT_ID:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing ENTRA_TENANT_ID.")
    return f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/discovery/v2.0/keys"


def _expected_issuers() -> set[str]:
    # Support both common forms we see in practice
    return {
        f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/v2.0",
        f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/",
        f"https://sts.windows.net/{ENTRA_TENANT_ID}/",
    }


def _expected_audiences() -> list[str]:
    if not ENTRA_API_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing ENTRA_API_CLIENT_ID.")
    return [ENTRA_API_CLIENT_ID, f"api://{ENTRA_API_CLIENT_ID}"]


def _fetch_jwks() -> dict:
    """
    Fetch JWKS with caching.
    IMPORTANT: If network fetch fails but we already have a cache, return the stale cache.
    That prevents random Microsoft/DNS hiccups from killing your entire backend.
    """
    global _JWKS_CACHE, _JWKS_FETCHED_AT

    now = time.time()
    if _JWKS_CACHE and (now - _JWKS_FETCHED_AT) < JWKS_TTL_SECONDS:
        return _JWKS_CACHE

    try:
        r = requests.get(_jwks_url(), timeout=JWKS_TIMEOUT_SECONDS)
        r.raise_for_status()
        _JWKS_CACHE = r.json()
        _JWKS_FETCHED_AT = now
        return _JWKS_CACHE
    except Exception:
        if _JWKS_CACHE:
            # stale fallback
            return _JWKS_CACHE
        raise HTTPException(status_code=503, detail="Auth unavailable: failed to fetch JWKS keys.")


def _require_api_key(x_api_key: Optional[str]) -> AuthContext:
    if not GATEWAY_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing GATEWAY_API_KEY.")
    if not x_api_key or x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid x-api-key.")
    return AuthContext(auth_type="api_key")


def _validate_bearer(token: str) -> AuthContext:
    jwks = _fetch_jwks()

    try:
        hdr = jwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token header.")

    kid = hdr.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Unauthorized: token missing kid.")

    key_obj = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            try:
                key_obj = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
            except Exception:
                key_obj = None
            break

    if key_obj is None:
        raise HTTPException(status_code=401, detail="Unauthorized: signing key not found for kid.")

    try:
        payload = jwt.decode(
            token,
            key=key_obj,
            algorithms=["RS256"],
            audience=_expected_audiences(),
            options={
                "require": ["exp"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
            },
            leeway=60,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Unauthorized: token expired.")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid audience.")
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token.")

    iss = payload.get("iss")
    if not iss or iss not in _expected_issuers():
        raise HTTPException(status_code=401, detail="Unauthorized: invalid issuer.")

    scp = payload.get("scp", "")
    scopes = set(scp.split()) if isinstance(scp, str) else set()
    if ENTRA_REQUIRED_SCOPE and ENTRA_REQUIRED_SCOPE not in scopes:
        raise HTTPException(status_code=403, detail=f"Forbidden: missing scope '{ENTRA_REQUIRED_SCOPE}'.")

    upn = payload.get("preferred_username") or payload.get("upn") or payload.get("email")

    return AuthContext(
        auth_type="entra",
        oid=payload.get("oid"),
        tid=payload.get("tid"),
        upn=upn,
    )


def _auth_core(authorization: Optional[str], x_api_key: Optional[str]) -> AuthContext:
    mode = AUTH_MODE
    if mode not in {"api_key", "entra", "either"}:
        raise HTTPException(status_code=500, detail="Server misconfigured: AUTH_MODE must be api_key|entra|either.")

    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()

    if bearer and mode in {"entra", "either"}:
        return _validate_bearer(bearer)

    if mode == "entra":
        raise HTTPException(status_code=401, detail="Unauthorized: missing bearer token.")

    return _require_api_key(x_api_key)


def auth_dep(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> AuthContext:
    ctx = _auth_core(authorization, x_api_key)
    request.state.auth_type = ctx.auth_type
    return ctx


def auth_dep_metrics(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> AuthContext:
    """
    Metrics MUST remain accessible via x-api-key even if Entra JWKS fetch is flaky.
    """
    if x_api_key:
        try:
            ctx = _require_api_key(x_api_key)
            request.state.auth_type = ctx.auth_type
            return ctx
        except Exception:
            pass

    ctx = _auth_core(authorization, x_api_key)
    request.state.auth_type = ctx.auth_type
    return ctx

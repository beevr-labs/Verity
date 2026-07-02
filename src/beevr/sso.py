"""SSO — OIDC ID-token validation (doc 11 §7, FR-SE-03, SEC-3).

Production flow: browser -> customer IdP -> callback with an id_token (JWT).
This module does the server-side validation with REAL cryptography:
  * RS256 signature against the IdP's JWKS (fetched once at configure time and
    cached in-boundary; rotation = re-configure)
  * iss / aud / exp / nbf checks
  * claim mapping -> Session (sub, roles, matter_grants)

No end-user password store (SEC-3). The stub /auth/token endpoint remains for
demos; deployments configure OIDC and use /auth/oidc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jwt
from jwt import InvalidTokenError

from .store import Session


class SsoError(Exception):
    """Raised when an id_token fails validation — always a 401."""


@dataclass
class OidcConfig:
    issuer: str
    audience: str                       # our client_id at the IdP
    jwks: dict = field(default_factory=dict)   # {"keys": [...]} — in-boundary cache
    roles_claim: str = "roles"
    grants_claim: str = "matter_grants"

    def _key_for(self, token: str):
        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError as ex:
            raise SsoError(f"malformed token: {ex}")
        kid = header.get("kid")
        for k in self.jwks.get("keys", []):
            if k.get("kid") == kid:
                return jwt.PyJWK(k).key
        raise SsoError(f"no JWKS key matches kid={kid!r}")


def validate_id_token(cfg: OidcConfig, token: str) -> Session:
    """Verify signature + registered claims; map to a Session. Raises SsoError."""
    key = cfg._key_for(token)
    try:
        claims = jwt.decode(
            token, key=key, algorithms=["RS256"],        # pinned; no alg confusion
            issuer=cfg.issuer, audience=cfg.audience,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except InvalidTokenError as ex:
        raise SsoError(f"id_token rejected: {ex}")

    roles = claims.get(cfg.roles_claim) or ["user"]
    grants = claims.get(cfg.grants_claim) or []
    return Session(user_id=claims["sub"],
                   role=roles[0] if roles else "user",
                   matter_grants=frozenset(grants),
                   walled_groups=frozenset(claims.get("walled_groups", [])))

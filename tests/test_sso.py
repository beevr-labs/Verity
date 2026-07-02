"""OIDC SSO tests — FR-SE-03, TC-503-class. Real RS256 crypto: a self-generated
RSA keypair plays the IdP; tokens are actually signed and actually verified."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app
from beevr.sso import OidcConfig, SsoError, validate_id_token

ISS, AUD = "https://idp.bank-x.example", "beevr-client"


@pytest.fixture(scope="module")
def idp():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk["kid"] = "kid-1"
    evil = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def sign(claims: dict, *, kid="kid-1", signer=key) -> str:
        return jwt.encode(claims, signer, algorithm="RS256", headers={"kid": kid})

    return {"sign": sign, "jwks": {"keys": [jwk]}, "evil_key": evil}


def _claims(**over):
    base = {"iss": ISS, "aud": AUD, "sub": "alice@bank-x.example",
            "exp": int(time.time()) + 600, "roles": ["user"],
            "matter_grants": ["A", "B"]}
    base.update(over)
    return base


def _cfg(idp) -> OidcConfig:
    return OidcConfig(issuer=ISS, audience=AUD, jwks=idp["jwks"])


def test_valid_token_maps_claims_to_session(idp):
    s = validate_id_token(_cfg(idp), idp["sign"](_claims()))
    assert s.user_id == "alice@bank-x.example"
    assert s.matter_grants == frozenset({"A", "B"})


def test_wrong_key_rejected(idp):
    tok = idp["sign"](_claims(), signer=idp["evil_key"])   # forged signature
    with pytest.raises(SsoError, match="rejected"):
        validate_id_token(_cfg(idp), tok)


def test_expired_wrong_aud_wrong_iss_rejected(idp):
    for bad in (_claims(exp=int(time.time()) - 10),
                _claims(aud="someone-else"),
                _claims(iss="https://evil.example")):
        with pytest.raises(SsoError):
            validate_id_token(_cfg(idp), idp["sign"](bad))


def test_unknown_kid_rejected(idp):
    with pytest.raises(SsoError, match="kid"):
        validate_id_token(_cfg(idp), idp["sign"](_claims(), kid="kid-999"))


def test_oidc_endpoint_end_to_end(idp):
    state = AppState()
    state.oidc = _cfg(idp)
    client = TestClient(create_app(state))
    # good id_token -> session token that works against the API
    r = client.post("/auth/oidc", json={"id_token": idp["sign"](_claims())})
    assert r.status_code == 200
    hdr = {"Authorization": f"Bearer {r.json()['token']}"}
    client.post("/matters", json={"id": "A", "client": "X", "name": "MA"}, headers=hdr)
    assert client.get("/matters", headers=hdr).status_code == 200
    # forged id_token -> 401
    forged = idp["sign"](_claims(), signer=idp["evil_key"])
    assert client.post("/auth/oidc", json={"id_token": forged}).status_code == 401


def test_oidc_unconfigured_is_501():
    client = TestClient(create_app(AppState()))
    assert client.post("/auth/oidc", json={"id_token": "x"}).status_code == 501

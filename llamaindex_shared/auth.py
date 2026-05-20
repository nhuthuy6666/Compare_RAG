from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from http import cookies
from typing import Any


SESSION_COOKIE_NAME = "ntu_fusion_session"
SESSION_TTL_SECONDS = 60 * 60 * 12
CLUSTER_TOKEN_HEADER = "X-Cluster-Token"


## Lấy secret dùng để ký cookie session và token nội bộ của cụm.
def _get_secret() -> bytes:
    return os.getenv("NTU_AUTH_SECRET", "ntu-fusion-local-secret").encode("utf-8")


## Dựng danh sách tài khoản mặc định từ environment cho user/admin.
def _get_accounts() -> list[dict[str, str]]:
    return [
        {
            "username": os.getenv("NTU_ADMIN_USERNAME", "admin"),
            "password": os.getenv("NTU_ADMIN_PASSWORD", "admin123"),
            "role": "admin",
            "display_name": os.getenv("NTU_ADMIN_DISPLAY_NAME", "Administrator"),
        },
        {
            "username": os.getenv("NTU_USER_USERNAME", "user"),
            "password": os.getenv("NTU_USER_PASSWORD", "user123"),
            "role": "user",
            "display_name": os.getenv("NTU_USER_DISPLAY_NAME", "User"),
        },
    ]


## Xác thực username/password và trả về session payload nếu hợp lệ.
def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    normalized_username = str(username or "").strip()
    raw_password = str(password or "")
    for account in _get_accounts():
        if normalized_username == account["username"] and raw_password == account["password"]:
            return {
                "username": account["username"],
                "role": account["role"],
                "display_name": account["display_name"],
            }
    return None


## Trả về hint tài khoản mặc định để UI đăng nhập có thể hiển thị nhanh.
def get_default_accounts_hint() -> list[dict[str, str]]:
    return [
        {
            "username": account["username"],
            "role": account["role"],
            "display_name": account["display_name"],
        }
        for account in _get_accounts()
    ]


## Ký payload bytes bằng HMAC-SHA256 để chống giả mạo session.
def _sign(payload: bytes) -> str:
    digest = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


## Mã hóa payload session JSON thành token base64url gọn nhẹ.
def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


## Giải mã phần payload của token session và kiểm tra kiểu dữ liệu.
def _decode_payload(value: str) -> dict[str, Any] | None:
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


## Tạo cookie đăng nhập chứa payload đã ký cho người dùng đã xác thực.
def build_session_cookie(account: dict[str, str]) -> str:
    payload = {
        "username": account["username"],
        "role": account["role"],
        "display_name": account["display_name"],
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    encoded_payload = _encode_payload(payload)
    signature = _sign(encoded_payload.encode("utf-8"))
    cookie = cookies.SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = f"{encoded_payload}.{signature}"
    cookie[SESSION_COOKIE_NAME]["path"] = "/"
    cookie[SESSION_COOKIE_NAME]["httponly"] = True
    cookie[SESSION_COOKIE_NAME]["samesite"] = "Lax"
    cookie[SESSION_COOKIE_NAME]["max-age"] = str(SESSION_TTL_SECONDS)
    return cookie.output(header="").strip()


## Tạo cookie logout để xóa session hiện tại ở phía trình duyệt.
def build_logout_cookie() -> str:
    cookie = cookies.SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = ""
    cookie[SESSION_COOKIE_NAME]["path"] = "/"
    cookie[SESSION_COOKIE_NAME]["httponly"] = True
    cookie[SESSION_COOKIE_NAME]["samesite"] = "Lax"
    cookie[SESSION_COOKIE_NAME]["max-age"] = "0"
    cookie[SESSION_COOKIE_NAME]["expires"] = "Thu, 01 Jan 1970 00:00:00 GMT"
    return cookie.output(header="").strip()


## Đọc, xác minh chữ ký và phục hồi session từ header Cookie của request.
def read_session_from_cookie(cookie_header: str | None) -> dict[str, str] | None:
    if not cookie_header:
        return None
    jar = cookies.SimpleCookie()
    try:
        jar.load(cookie_header)
    except cookies.CookieError:
        return None
    morsel = jar.get(SESSION_COOKIE_NAME)
    if morsel is None:
        return None
    token = morsel.value
    if "." not in token:
        return None
    encoded_payload, signature = token.rsplit(".", 1)
    expected = _sign(encoded_payload.encode("utf-8"))
    if not hmac.compare_digest(signature, expected):
        return None
    payload = _decode_payload(encoded_payload)
    if payload is None:
        return None
    expires_at = int(payload.get("exp") or 0)
    if expires_at <= int(time.time()):
        return None
    username = str(payload.get("username") or "").strip()
    role = str(payload.get("role") or "").strip()
    display_name = str(payload.get("display_name") or username).strip()
    if not username or role not in {"user", "admin"}:
        return None
    return {"username": username, "role": role, "display_name": display_name}


## Kiểm tra session hiện tại có đủ quyền `user` hoặc `admin` hay không.
def has_role(session: dict[str, str] | None, required_role: str) -> bool:
    if session is None:
        return False
    if required_role == "user":
        return session.get("role") in {"user", "admin"}
    if required_role == "admin":
        return session.get("role") == "admin"
    return False


## Lấy shared token dùng cho các request nội bộ giữa các backend trong cụm.
def get_cluster_token() -> str:
    return os.getenv("NTU_CLUSTER_TOKEN", "ntu-fusion-cluster-token")


## Dựng header xác thực cho lời gọi chéo giữa các backend trong cluster.
def build_cluster_headers() -> dict[str, str]:
    return {CLUSTER_TOKEN_HEADER: get_cluster_token()}


## Xác định request hiện tại có phải đến từ backend nội bộ đã ký token hay không.
def is_internal_cluster_request(headers) -> bool:
    try:
        provided = str(headers.get(CLUSTER_TOKEN_HEADER) or "").strip()
    except Exception:
        return False
    return bool(provided) and hmac.compare_digest(provided, get_cluster_token())

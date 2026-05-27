from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import time
import unicodedata
from email.message import EmailMessage
from http import cookies
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import bcrypt
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
SESSION_COOKIE_NAME = "ntu_fusion_session"
SESSION_TTL_SECONDS = 60 * 60 * 12
EMAIL_VERIFICATION_TTL_SECONDS = 60 * 60 * 24
CLUSTER_TOKEN_HEADER = "X-Cluster-Token"
PASSWORD_MIN_LENGTH = 8
PASSWORD_BCRYPT_ROUNDS = 12
PASSWORD_HASH_ITERATIONS = 310_000
AUTH_STORE_PATH = Path(os.getenv("NTU_AUTH_STORE_PATH", str(PROJECT_ROOT / ".auth_users.json")))
EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
USERNAME_ALLOWED_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?$")
AUTH_STORE_LOCK = Lock()


def _get_secret() -> bytes:
    return os.getenv("NTU_AUTH_SECRET", "ntu-fusion-local-secret").encode("utf-8")


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


def _build_account_payload(account: dict[str, Any]) -> dict[str, str]:
    payload = {
        "username": str(account.get("username") or "").strip(),
        "role": str(account.get("role") or "user").strip(),
        "display_name": str(account.get("display_name") or account.get("username") or "").strip(),
        "account_type": str(account.get("account_type") or "system").strip() or "system",
    }
    email = str(account.get("email") or "").strip()
    if email:
        payload["email"] = email
    return payload


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _validate_email(email: str) -> str:
    normalized = _normalize_email(email)
    if not normalized or not EMAIL_PATTERN.match(normalized):
        raise ValueError("Email không hợp lệ.")
    return normalized


def _validate_password(password: str) -> str:
    raw_password = str(password or "")
    if len(raw_password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Mật khẩu phải có ít nhất {PASSWORD_MIN_LENGTH} ký tự.")
    return raw_password


def _normalize_display_name(display_name: str, fallback_email: str) -> str:
    value = str(display_name or "").strip()
    if value:
        return value[:80]
    return fallback_email.split("@", 1)[0][:80] or "User"


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _slugify_username(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower()
    lowered = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    lowered = lowered.strip("._-")
    lowered = re.sub(r"[-._]{2,}", "-", lowered)
    return lowered[:32].strip("._-")


def _username_candidates(display_name: str, email: str) -> list[str]:
    local_part = _normalize_email(email).split("@", 1)[0]
    candidates = [
        _slugify_username(display_name),
        _slugify_username(local_part),
        _slugify_username(f"user-{local_part}"),
    ]
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    if not unique_candidates:
        unique_candidates.append("user")
    return unique_candidates


def _hash_password_pbkdf2(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def _hash_password_bcrypt(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=PASSWORD_BCRYPT_ROUNDS),
    ).decode("utf-8")


def _verify_password_pbkdf2(password: str, salt_hex: str, digest_hex: str) -> bool:
    try:
        _, calculated = _hash_password_pbkdf2(password, salt_hex=salt_hex)
    except ValueError:
        return False
    return hmac.compare_digest(calculated, digest_hex)


def _verify_password(password: str, user: dict[str, Any]) -> bool:
    algo = str(user.get("password_algo") or "pbkdf2_sha256").strip().lower()
    digest = str(user.get("password_hash") or "")
    if algo == "bcrypt":
        try:
            return bcrypt.checkpw(password.encode("utf-8"), digest.encode("utf-8"))
        except ValueError:
            return False
    return _verify_password_pbkdf2(password, str(user.get("password_salt") or ""), digest)


def _upgrade_password_hash_if_needed(email: str, password: str, user: dict[str, Any]) -> None:
    algo = str(user.get("password_algo") or "pbkdf2_sha256").strip().lower()
    if algo == "bcrypt":
        return
    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        _ensure_registered_usernames(store)
        users = store.setdefault("users", {})
        current = users.get(email)
        if not isinstance(current, dict):
            return
        current["password_algo"] = "bcrypt"
        current["password_salt"] = ""
        current["password_hash"] = _hash_password_bcrypt(password)
        current["updated_at"] = int(time.time())
        users[email] = current
        _save_auth_store(store)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _empty_auth_store() -> dict[str, dict[str, dict[str, Any]]]:
    return {"users": {}}


def _load_auth_store() -> dict[str, dict[str, dict[str, Any]]]:
    if not AUTH_STORE_PATH.exists():
        return _empty_auth_store()
    try:
        payload = json.loads(AUTH_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_auth_store()
    users = payload.get("users")
    if not isinstance(users, dict):
        return _empty_auth_store()
    return {"users": users}


def _save_auth_store(payload: dict[str, Any]) -> None:
    AUTH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = AUTH_STORE_PATH.with_suffix(f"{AUTH_STORE_PATH.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, AUTH_STORE_PATH)


def _system_usernames() -> set[str]:
    return {_normalize_username(account.get("username") or "") for account in _get_accounts()}


def _username_exists(store: dict[str, Any], username: str, *, exclude_email: str = "") -> bool:
    normalized_username = _normalize_username(username)
    if not normalized_username:
        return False
    if normalized_username in _system_usernames():
        return True
    users = store.get("users") or {}
    for email, user in users.items():
        if exclude_email and email == exclude_email:
            continue
        if not isinstance(user, dict):
            continue
        if _normalize_username(str(user.get("username") or "")) == normalized_username:
            return True
    return False


def _build_unique_registered_username(
    store: dict[str, Any],
    *,
    display_name: str,
    email: str,
    exclude_email: str = "",
) -> str:
    for base_candidate in _username_candidates(display_name, email):
        if not _username_exists(store, base_candidate, exclude_email=exclude_email):
            return base_candidate
        suffix = 2
        while suffix <= 9999:
            shortened = base_candidate[: max(1, 32 - len(str(suffix)) - 1)].rstrip("._-") or "user"
            candidate = f"{shortened}-{suffix}"
            if USERNAME_ALLOWED_PATTERN.match(candidate) and not _username_exists(store, candidate, exclude_email=exclude_email):
                return candidate
            suffix += 1
    raise ValueError("Không tạo được username hợp lệ cho tài khoản.")


def _ensure_registered_usernames(store: dict[str, Any]) -> bool:
    users = store.get("users") or {}
    changed = False
    for email, user in users.items():
        if not isinstance(user, dict):
            continue
        current_username = _normalize_username(str(user.get("username") or ""))
        if current_username and USERNAME_ALLOWED_PATTERN.match(current_username):
            continue
        user["username"] = _build_unique_registered_username(
            store,
            display_name=str(user.get("display_name") or ""),
            email=str(user.get("email") or email),
            exclude_email=email,
        )
        changed = True
    return changed


def _find_registered_user(identifier: str) -> tuple[str, dict[str, Any]] | None:
    normalized_identifier = _normalize_username(identifier)
    if not normalized_identifier:
        return None
    normalized_email = _normalize_email(identifier)
    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        changed = _ensure_registered_usernames(store)
        users = store.get("users") or {}
        direct_user = users.get(normalized_email)
        if isinstance(direct_user, dict):
            if changed:
                _save_auth_store(store)
            return normalized_email, dict(direct_user)
        for email, user in users.items():
            if not isinstance(user, dict):
                continue
            if _normalize_username(str(user.get("username") or "")) == normalized_identifier:
                if changed:
                    _save_auth_store(store)
                return email, dict(user)
        if changed:
            _save_auth_store(store)
    return None


def _find_system_account(identifier: str) -> dict[str, str] | None:
    normalized = str(identifier or "").strip()
    if not normalized:
        return None
    for account in _get_accounts():
        if normalized == account["username"]:
            return account
    return None


def authenticate_user_with_status(username: str, password: str) -> dict[str, Any]:
    normalized_username = str(username or "").strip()
    raw_password = str(password or "")
    for account in _get_accounts():
        if normalized_username == account["username"] and raw_password == account["password"]:
            return {
                "account": _build_account_payload({**account, "account_type": "system"}),
                "error": None,
            }

    user_record = _find_registered_user(normalized_username)
    if user_record is None:
        return {"account": None, "error": "invalid_credentials"}
    matched_email, user = user_record
    if not _verify_password(raw_password, user):
        return {"account": None, "error": "invalid_credentials"}
    if not bool(user.get("verified")):
        return {
            "account": None,
            "error": "email_not_verified",
            "message": "Email chưa được xác thực. Vui lòng kiểm tra hộp thư rồi mở liên kết xác thực trước khi đăng nhập.",
        }
    _upgrade_password_hash_if_needed(matched_email, raw_password, user)
    return {
        "account": _build_account_payload(
            {
                "username": user.get("username") or matched_email,
                "email": user.get("email") or matched_email,
                "role": user.get("role") or "user",
                "display_name": user.get("display_name") or user.get("username") or matched_email,
                "account_type": "registered",
            }
        ),
        "error": None,
    }


def _sign(payload: bytes) -> str:
    digest = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_payload(value: str) -> dict[str, Any] | None:
    padding = "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def build_session_cookie(account: dict[str, str]) -> str:
    payload = {
        "username": account["username"],
        "role": account["role"],
        "display_name": account["display_name"],
        "account_type": str(account.get("account_type") or "system"),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    email = str(account.get("email") or "").strip()
    if email:
        payload["email"] = email
    encoded_payload = _encode_payload(payload)
    signature = _sign(encoded_payload.encode("utf-8"))
    cookie = cookies.SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = f"{encoded_payload}.{signature}"
    cookie[SESSION_COOKIE_NAME]["path"] = "/"
    cookie[SESSION_COOKIE_NAME]["httponly"] = True
    cookie[SESSION_COOKIE_NAME]["samesite"] = "Lax"
    cookie[SESSION_COOKIE_NAME]["max-age"] = str(SESSION_TTL_SECONDS)
    return cookie.output(header="").strip()


def build_logout_cookie() -> str:
    cookie = cookies.SimpleCookie()
    cookie[SESSION_COOKIE_NAME] = ""
    cookie[SESSION_COOKIE_NAME]["path"] = "/"
    cookie[SESSION_COOKIE_NAME]["httponly"] = True
    cookie[SESSION_COOKIE_NAME]["samesite"] = "Lax"
    cookie[SESSION_COOKIE_NAME]["max-age"] = "0"
    cookie[SESSION_COOKIE_NAME]["expires"] = "Thu, 01 Jan 1970 00:00:00 GMT"
    return cookie.output(header="").strip()


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
    email = str(payload.get("email") or "").strip()
    account_type = str(payload.get("account_type") or "system").strip() or "system"
    if not username or role not in {"user", "admin"}:
        return None
    session = {
        "username": username,
        "role": role,
        "display_name": display_name,
        "account_type": account_type,
    }
    if email:
        session["email"] = email
    return session


def supports_self_service_account(session: dict[str, str] | None) -> bool:
    if session is None:
        return False
    return str(session.get("account_type") or "").strip() == "registered" and bool(str(session.get("email") or "").strip())


def change_password_for_session(
    session: dict[str, str] | None,
    *,
    current_password: str,
    new_password: str,
) -> dict[str, str]:
    if session is None:
        raise ValueError("Bạn chưa đăng nhập.")
    if not supports_self_service_account(session):
        raise ValueError("Tài khoản hiện tại không hỗ trợ đổi mật khẩu tại giao diện này.")

    email = _normalize_email(str(session.get("email") or ""))
    validated_current_password = str(current_password or "")
    validated_new_password = _validate_password(new_password)

    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        _ensure_registered_usernames(store)
        users = store.setdefault("users", {})
        user = users.get(email)
        if not isinstance(user, dict):
            raise ValueError("Không tìm thấy tài khoản.")
        if not _verify_password(validated_current_password, user):
            raise ValueError("Mật khẩu hiện tại không đúng.")
        user["password_algo"] = "bcrypt"
        user["password_salt"] = ""
        user["password_hash"] = _hash_password_bcrypt(validated_new_password)
        user["updated_at"] = int(time.time())
        users[email] = user
        _save_auth_store(store)

    return {"message": "Đổi mật khẩu thành công."}


def delete_account_for_session(session: dict[str, str] | None, *, password: str) -> dict[str, str]:
    if session is None:
        raise ValueError("Bạn chưa đăng nhập.")
    if not supports_self_service_account(session):
        raise ValueError("Tài khoản hiện tại không hỗ trợ xóa tại giao diện này.")

    email = _normalize_email(str(session.get("email") or ""))
    raw_password = str(password or "")

    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        _ensure_registered_usernames(store)
        users = store.setdefault("users", {})
        user = users.get(email)
        if not isinstance(user, dict):
            raise ValueError("Không tìm thấy tài khoản.")
        if not _verify_password(raw_password, user):
            raise ValueError("Mật khẩu không đúng.")
        users.pop(email, None)
        _save_auth_store(store)

    return {"message": "Tài khoản đã được xóa."}


def _build_verification_link(verification_base_url: str, token: str) -> str:
    base_url = str(verification_base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Thiếu verification_base_url để gửi email xác thực.")
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}token={quote(token)}"


def _smtp_enabled() -> bool:
    return bool(os.getenv("NTU_SMTP_HOST")) and bool(os.getenv("NTU_EMAIL_FROM"))


def _send_verification_email(email: str, display_name: str, verification_link: str) -> dict[str, str]:
    if not _smtp_enabled():
        print(f"[auth] Verification link for {email}: {verification_link}")
        return {"delivery_mode": "console", "verification_link": verification_link}

    smtp_host = str(os.getenv("NTU_SMTP_HOST") or "").strip()
    smtp_port = int(os.getenv("NTU_SMTP_PORT", "587"))
    smtp_username = str(os.getenv("NTU_SMTP_USERNAME") or "").strip()
    smtp_password = str(os.getenv("NTU_SMTP_PASSWORD") or "")
    smtp_use_tls = str(os.getenv("NTU_SMTP_USE_TLS", "true")).strip().lower() not in {"0", "false", "no"}
    smtp_use_ssl = str(os.getenv("NTU_SMTP_USE_SSL", "false")).strip().lower() in {"1", "true", "yes"}
    sender = str(os.getenv("NTU_EMAIL_FROM") or "").strip()

    message = EmailMessage()
    message["Subject"] = "Xác thực địa chỉ email cho tài khoản NTU Bot"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "\n".join(
            [
                f"Kính chào {display_name},",
                "",
                "Hệ thống NTU Bot đã nhận được yêu cầu đăng ký tài khoản bằng địa chỉ email này.",
                "Vui lòng mở liên kết dưới đây để xác thực email và hoàn tất kích hoạt tài khoản:",
                verification_link,
                "",
                f"Liên kết xác thực có hiệu lực trong {EMAIL_VERIFICATION_TTL_SECONDS // 3600} giờ.",
                "Nếu bạn không thực hiện yêu cầu đăng ký, vui lòng bỏ qua email này.",
                "",
                "Trân trọng,",
                "NTU Bot",
            ]
        )
    )
    message.add_alternative(
        f"""\
<!DOCTYPE html>
<html lang="vi">
  <body style="margin:0;padding:24px;background:#f4f7fb;font-family:Arial,sans-serif;color:#16202a;">
    <div style="max-width:560px;margin:0 auto;background:#ffffff;border:1px solid #dbe4ef;border-radius:18px;overflow:hidden;">
      <div style="padding:20px 24px;background:linear-gradient(135deg,#0f1722,#1c2735);color:#e8edf5;">
        <div style="font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#7cc0ff;">NTU Bot</div>
        <h1 style="margin:10px 0 0;font-size:24px;line-height:1.3;">Xác thực địa chỉ email</h1>
      </div>
      <div style="padding:24px;">
        <p style="margin:0 0 14px;">Kính chào <strong>{_escape_html(display_name)}</strong>,</p>
        <p style="margin:0 0 14px;line-height:1.7;color:#445160;">
          Hệ thống NTU Bot đã nhận được yêu cầu đăng ký tài khoản bằng địa chỉ email này.
          Vui lòng xác thực email để hoàn tất kích hoạt tài khoản.
        </p>
        <p style="margin:0 0 20px;">
          <a href="{_escape_html(verification_link)}" style="display:inline-block;padding:12px 18px;border-radius:12px;background:#2f7fe8;color:#ffffff;text-decoration:none;font-weight:700;">
            Xác thực email
          </a>
        </p>
        <p style="margin:0 0 12px;line-height:1.7;color:#445160;">
          Nếu nút trên không hoạt động, bạn có thể mở liên kết sau trong trình duyệt:
        </p>
        <p style="margin:0 0 16px;word-break:break-all;">
          <a href="{_escape_html(verification_link)}" style="color:#2f7fe8;text-decoration:underline;">{_escape_html(verification_link)}</a>
        </p>
        <p style="margin:0 0 8px;line-height:1.7;color:#445160;">
          Liên kết xác thực có hiệu lực trong {EMAIL_VERIFICATION_TTL_SECONDS // 3600} giờ.
        </p>
        <p style="margin:0;line-height:1.7;color:#445160;">
          Nếu bạn không thực hiện yêu cầu đăng ký, vui lòng bỏ qua email này.
        </p>
      </div>
    </div>
  </body>
</html>
""",
        subtype="html",
    )

    try:
        if smtp_use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
                if smtp_username:
                    server.login(smtp_username, smtp_password)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
                server.ehlo()
                if smtp_use_tls:
                    server.starttls()
                    server.ehlo()
                if smtp_username:
                    server.login(smtp_username, smtp_password)
                server.send_message(message)
    except Exception as exc:
        raise RuntimeError(f"Không gửi được email xác thực. Chi tiết: {exc}") from exc

    return {"delivery_mode": "smtp"}


def register_email_user(email: str, password: str, display_name: str, verification_base_url: str) -> dict[str, Any]:
    normalized_email = _validate_email(email)
    normalized_password = _validate_password(password)
    normalized_display_name = _normalize_display_name(display_name, normalized_email)

    for account in _get_accounts():
        if normalized_email == str(account.get("username") or "").strip().lower():
            raise ValueError("Email này đang trùng với tài khoản hệ thống.")

    token = secrets.token_urlsafe(32)
    token_digest = _token_hash(token)
    now = int(time.time())
    expires_at = now + EMAIL_VERIFICATION_TTL_SECONDS
    password_hash = _hash_password_bcrypt(normalized_password)

    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        _ensure_registered_usernames(store)
        users = store.setdefault("users", {})
        existing = users.get(normalized_email)
        if isinstance(existing, dict) and bool(existing.get("verified")):
            raise ValueError("Email này đã được đăng ký.")
        registered_username = _build_unique_registered_username(
            store,
            display_name=normalized_display_name,
            email=normalized_email,
            exclude_email=normalized_email,
        )
        users[normalized_email] = {
            "email": normalized_email,
            "username": registered_username,
            "display_name": normalized_display_name,
            "role": "user",
            "password_algo": "bcrypt",
            "password_salt": "",
            "password_hash": password_hash,
            "verified": False,
            "verification_token_hash": token_digest,
            "verification_expires_at": expires_at,
            "created_at": int(existing.get("created_at") or now) if isinstance(existing, dict) else now,
            "updated_at": now,
        }
        _save_auth_store(store)

    verification_link = _build_verification_link(verification_base_url, token)
    delivery = _send_verification_email(normalized_email, normalized_display_name, verification_link)
    return {
        "email": normalized_email,
        "username": registered_username,
        "display_name": normalized_display_name,
        "message": "Đăng ký thành công. Vui lòng mở email để xác thực tài khoản trước khi đăng nhập.",
        **delivery,
    }


def verify_email_token(token: str) -> dict[str, str]:
    normalized_token = str(token or "").strip()
    if not normalized_token:
        raise ValueError("Thiếu token xác thực email.")
    token_digest = _token_hash(normalized_token)
    now = int(time.time())

    with AUTH_STORE_LOCK:
        store = _load_auth_store()
        users = store.setdefault("users", {})
        _ensure_registered_usernames(store)
        matched_email = ""
        matched_user: dict[str, Any] | None = None
        for email, user in users.items():
            if not isinstance(user, dict):
                continue
            if hmac.compare_digest(str(user.get("verification_token_hash") or ""), token_digest):
                matched_email = email
                matched_user = user
                break
        if matched_user is None:
            raise ValueError("Liên kết xác thực không hợp lệ hoặc đã được sử dụng.")
        expires_at = int(matched_user.get("verification_expires_at") or 0)
        if expires_at <= now:
            raise ValueError("Liên kết xác thực đã hết hạn. Vui lòng đăng ký lại để nhận email mới.")

        matched_user["verified"] = True
        matched_user["verification_token_hash"] = ""
        matched_user["verification_expires_at"] = 0
        matched_user["updated_at"] = now
        users[matched_email] = matched_user
        _save_auth_store(store)

    return {
        "email": str(matched_user.get("email") or matched_email),
        "username": str(matched_user.get("username") or ""),
        "display_name": str(matched_user.get("display_name") or matched_email),
        "message": "Xác thực email thành công. Bạn có thể quay lại ứng dụng và đăng nhập.",
    }


def render_email_verification_result(success: bool, message: str, login_href: str = "/") -> str:
    title = "Xác thực thành công" if success else "Không thể xác thực email"
    accent = "#3ecf7a" if success else "#e05252"
    badge = "Email verified" if success else "Verification failed"
    safe_title = _escape_html(title)
    safe_message = _escape_html(message)
    safe_href = _escape_html(login_href or "/")
    safe_badge = _escape_html(badge)
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #0a0c0f;
      --surface: #111318;
      --surface-2: #171b22;
      --border: rgba(255,255,255,0.08);
      --text: #e8edf5;
      --text-2: #98a4b3;
      --accent: {accent};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at top, rgba(78,155,255,0.14), transparent 34%),
        radial-gradient(circle at bottom left, rgba(255,255,255,0.05), transparent 28%),
        var(--bg);
      color: var(--text);
      font: 16px/1.6 'Segoe UI', sans-serif;
    }}
    .card {{
      width: min(520px, 100%);
      background:
        radial-gradient(circle at top right, rgba(78,155,255,0.12), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.018), transparent 100%),
        var(--surface);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 30px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
      padding: 6px 11px;
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--accent);
      border: 1px solid rgba(255,255,255,0.1);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .badge::before {{
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 12px currentColor;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: -0.03em;
    }}
    p {{
      margin: 0 0 22px;
      color: var(--text-2);
    }}
    a {{
      display: inline-block;
      padding: 11px 16px;
      border-radius: 14px;
      background: var(--accent);
      color: #fff;
      text-decoration: none;
      font-weight: 600;
    }}
    .panel {{
      padding: 16px 18px;
      margin-bottom: 22px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.06);
      background: var(--surface-2);
      color: var(--text-2);
    }}
  </style>
</head>
<body>
  <main class="card">
    <div class="badge">{safe_badge}</div>
    <h1>{safe_title}</h1>
    <div class="panel">{safe_message}</div>
    <a href="{safe_href}">Quay lại đăng nhập</a>
  </main>
</body>
</html>"""


def _escape_html(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def has_role(session: dict[str, str] | None, required_role: str) -> bool:
    if session is None:
        return False
    if required_role == "user":
        return session.get("role") in {"user", "admin"}
    if required_role == "admin":
        return session.get("role") == "admin"
    return False


def get_cluster_token() -> str:
    return os.getenv("NTU_CLUSTER_TOKEN", "ntu-fusion-cluster-token")


def build_cluster_headers() -> dict[str, str]:
    return {CLUSTER_TOKEN_HEADER: get_cluster_token()}


def is_internal_cluster_request(headers) -> bool:
    try:
        provided = str(headers.get(CLUSTER_TOKEN_HEADER) or "").strip()
    except Exception:
        return False
    return bool(provided) and hmac.compare_digest(provided, get_cluster_token())

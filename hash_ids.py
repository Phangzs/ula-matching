import os, hmac, hashlib, concurrent.futures, typing as _t

from ensure_pepper import ensure_pepper
PEPPER_HEX = ensure_pepper()
PEPPER = bytes.fromhex(PEPPER_HEX)


_HEX_PEPPER = os.getenv("STUDENT_PEPPER")
if not _HEX_PEPPER or len(_HEX_PEPPER) != 64:
    raise RuntimeError(
        "Missing or malformed STUDENT_PEPPER (need 64-char hex). "
        "Set it in your environment or .env file."
    )

PEPPER: bytes = bytes.fromhex(_HEX_PEPPER)

def hash_student_id(student_id: str, pepper: bytes = PEPPER) -> str:
    """
    Deterministically hash a student ID using HMAC-SHA-256 and a secret pepper.
    """
    # encode to utf-8
    if not isinstance(student_id, str):
        student_id = str(student_id)

    return hmac.new(pepper, student_id.encode("utf-8"), hashlib.sha256).hexdigest()

import os, secrets, stat, pathlib

_ENV_KEY = "STUDENT_PEPPER"
_EXPECTED_LEN = 64

def _is_valid(hexstr: str | None) -> bool:
    return bool(hexstr) and len(hexstr) == _EXPECTED_LEN and all(c in "0123456789abcdef" for c in hexstr)

def ensure_pepper(env_path: str | pathlib.Path = ".env") -> str:
    """
    Guarantee a 256-bit pepper in the current process AND in `.env`.
    Returns the pepper as a lowercase 64-hex-char string.
    """
    # 1. Does a good pepper already exist?
    current = os.getenv(_ENV_KEY)
    if _is_valid(current):
        return current.lower()

    # 2. Need a new one
    new_pepper = secrets.token_hex(32) # 32 bytes = 64 hex chars

    # 3. Add to .env (append or create)
    env_file = pathlib.Path(env_path)
    if not env_file.exists():
        # create with user-only rw-permissions (0600) on POSIX
        env_file.touch(mode=0o600, exist_ok=True)
    else:
        # ensure permissions aren’t too open; don’t fail on Windows
        try:
            if (env_file.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)) != 0:
                env_file.chmod(env_file.stat().st_mode & ~0o077)
        except OSError:
            pass  # platforms without chmod support

    # If the key already exists in the file, replace it. Otherwise append.
    lines = env_file.read_text().splitlines() if env_file.read_text() else []
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith(f"{_ENV_KEY}="):
            lines[idx] = f"{_ENV_KEY}={new_pepper}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{_ENV_KEY}={new_pepper}")

    env_file.write_text("\n".join(lines) + ("\n" if lines else ""))

    # 4. Expose to the running process
    os.environ[_ENV_KEY] = new_pepper
    return new_pepper

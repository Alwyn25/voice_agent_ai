"""Keep secrets in URLs out of the logs (docs/gtm-demo-links.md §14).

uvicorn logs every request line in full. Two of our URLs carry a secret in the path or the
query: a demo link's token (/public/demo/{token}/…), which opens that company's demo for
anyone holding it, and the voice WebSocket's ?token=… . This filter rewrites both before a
record is formatted. A demo token keeps its first 6 characters, the same prefix the admin
screen shows, so a log line can still be matched to its link.
"""
import logging
import re

_DEMO_TOKEN = re.compile(r"(/public/demo/)([A-Za-z0-9_-]{6})[A-Za-z0-9_-]{10,}")
_QUERY_TOKEN = re.compile(r"([?&](?:token|access_token)=)[^&\s\"']+")


def redact(text: str) -> str:
    return _QUERY_TOKEN.sub(r"\1<redacted>", _DEMO_TOKEN.sub(r"\1\2…", text))


class RedactSecretsFilter(logging.Filter):
    """Redacts the message and each string argument in place. The arguments keep their
    shape: uvicorn's access formatter unpacks them (client, method, path, …)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def install(*logger_names: str) -> None:
    f = RedactSecretsFilter()
    for name in logger_names:
        logging.getLogger(name).addFilter(f)

"""Small text utilities."""
import re

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(value):
    """Lowercase `value` and collapse non-alphanumerics into dashes."""
    value = _slug_re.sub("-", value.lower()).strip("-")
    return value or "item"


def truncate_words(text, max_words=25, suffix="…"):
    """Cut `text` to at most `max_words` words, appending `suffix` if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + suffix

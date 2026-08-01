"""User management service."""
from sample_repo.utils.text import slugify


class ValidationError(Exception):
    """Raised when user input fails validation."""


class UserService:
    """In-memory user store with light validation.

    Real persistence lives behind the same interface in production.
    """

    def __init__(self):
        self._users = {}
        self._next_id = 1

    def create_user(self, email, display_name):
        """Create a user after validating email and normalising the handle."""
        # -- validation --------------------------------------------------
        if not email or "@" not in email:
            raise ValidationError(f"invalid email: {email!r}")
        domain = email.rsplit("@", 1)[1]
        if "." not in domain:
            raise ValidationError(f"invalid email domain: {domain!r}")
        if not display_name or not display_name.strip():
            raise ValidationError("display name is required")
        # -- normalisation -----------------------------------------------
        handle = slugify(display_name)
        if any(u["handle"] == handle for u in self._users.values()):
            handle = f"{handle}-{self._next_id}"
        # -- persistence --------------------------------------------------
        user = {
            "id": self._next_id,
            "email": email.lower(),
            "display_name": display_name.strip(),
            "handle": handle,
        }
        self._users[self._next_id] = user
        self._next_id += 1
        return user

    def get_user(self, user_id):
        """Return the user dict for `user_id`, or None."""
        return self._users.get(user_id)

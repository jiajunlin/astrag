"""Payment service talking to the external payments API."""
from sample_repo.utils.http import HttpClient


class PaymentError(Exception):
    """Raised when a charge is rejected before it reaches the API."""


def luhn_checksum_ok(card_number):
    """Return True if `card_number` passes the Luhn check."""
    digits = [int(d) for d in str(card_number) if d.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def charge_card(client: HttpClient, card_number, amount_cents):
    """Charge a card through the payments API.

    NOTE: no retry handling yet — transient 5xx errors bubble up.
    """
    if amount_cents <= 0:
        raise PaymentError("amount must be positive")
    if not luhn_checksum_ok(card_number):
        raise PaymentError("invalid card number")
    return client.get(f"charges?card={card_number}&amount={amount_cents}")

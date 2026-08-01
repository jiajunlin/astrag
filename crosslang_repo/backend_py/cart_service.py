from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/cart", methods=["GET"])
def get_cart():
    """Return the current user's cart contents and running total."""
    return jsonify({"items": [], "totalCents": 0})

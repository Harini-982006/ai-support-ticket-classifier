"""
AI-Powered Customer Support Ticket Classifier
===============================================
Uses OpenAI API to automatically classify customer support messages
into categories and priority levels.

Categories: Billing, Technical Issue, Account, General Inquiry
Priority:   High, Medium, Low

Features:
- OpenAI GPT-4o-mini integration with prompt engineering
- Automatic fallback to keyword-based classifier when API is unavailable
- Robust error handling and JSON validation
- Structured logging for visibility
"""

import json
import os
import re
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load API key from .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"Billing", "Technical Issue", "Account", "General Inquiry"}
VALID_PRIORITIES = {"High", "Medium", "Low"}

# Track which classification mode is active
_using_fallback = False

# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------


def get_openai_client() -> OpenAI:
    """Initialise and return the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set. Will use fallback classifier.")
        return None
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Fallback Keyword-Based Classifier
# ---------------------------------------------------------------------------

# Keyword rules for category detection
CATEGORY_RULES = {
    "Billing": [
        "payment", "pay", "bill", "billing", "invoice", "charge", "charged",
        "deducted", "deduct", "refund", "subscription", "plan", "pricing",
        "price", "cost", "fee", "money", "transaction", "receipt", "overcharged",
        "credit card", "wallet", "renewal", "cancel subscription",
    ],
    "Technical Issue": [
        "crash", "crashes", "crashing", "bug", "error", "not working",
        "doesn't work", "broken", "stuck", "freeze", "freezing", "slow",
        "loading", "glitch", "fail", "failed", "failure", "down", "outage",
        "can't open", "cannot open", "won't load", "not loading", "blank screen",
        "unresponsive", "timeout", "lag", "lagging", "update", "install",
        "compatibility", "display", "not responding", "login issue",
    ],
    "Account": [
        "account", "password", "login", "log in", "sign in", "sign up",
        "register", "registration", "profile", "email", "username",
        "locked out", "locked", "reset password", "change email",
        "change password", "two factor", "2fa", "verification", "verify",
        "deactivate", "delete account", "settings", "preferences",
    ],
    "General Inquiry": [
        "how to", "what is", "where", "when", "information", "info",
        "help", "question", "inquiry", "hours", "contact", "support",
        "about", "policy", "policies", "feature", "feedback", "suggest",
        "recommendation", "guide", "tutorial", "learn",
    ],
}

# Keywords that indicate high priority
HIGH_PRIORITY_KEYWORDS = [
    "urgent", "immediately", "asap", "critical", "emergency", "blocked",
    "can't access", "cannot access", "locked out", "not working",
    "crash", "crashes", "deducted", "lost", "stolen", "unauthorized",
    "fraud", "hacked", "security", "data loss", "not activated",
    "service down", "outage", "unable to", "can't login", "failed",
]

# Keywords that indicate medium priority
MEDIUM_PRIORITY_KEYWORDS = [
    "slow", "issue", "problem", "error", "trouble", "broken",
    "doesn't work", "bug", "fix", "resolve", "delay", "waiting",
    "incorrect", "wrong", "missing", "change", "update", "reset",
    "modify", "help me",
]


def classify_with_keywords(message: str) -> dict:
    """
    Classify a message using keyword matching as a fallback
    when the OpenAI API is unavailable.

    Returns:
        dict with keys "category" and "priority".
    """
    msg_lower = message.lower()

    # --- Determine category ---
    category_scores = {}
    for category, keywords in CATEGORY_RULES.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        category_scores[category] = score

    # Pick the category with the highest match score
    best_category = max(category_scores, key=category_scores.get)

    # If no keywords matched at all, default to General Inquiry
    if category_scores[best_category] == 0:
        best_category = "General Inquiry"

    # --- Determine priority ---
    high_score = sum(1 for kw in HIGH_PRIORITY_KEYWORDS if kw in msg_lower)
    medium_score = sum(1 for kw in MEDIUM_PRIORITY_KEYWORDS if kw in msg_lower)

    if high_score > 0:
        priority = "High"
    elif medium_score > 0:
        priority = "Medium"
    else:
        priority = "Low"

    return {"category": best_category, "priority": priority}


# ---------------------------------------------------------------------------
# OpenAI Classification Logic
# ---------------------------------------------------------------------------


def classify_with_openai(client: OpenAI, message: str) -> dict:
    """
    Send a single support message to OpenAI and return its classification.

    Returns:
        dict with keys "category" and "priority".
    """
    prompt = (
        "You are a support ticket classifier.\n\n"
        "Classify the given message into:\n"
        "- Category: Billing, Technical Issue, Account, General Inquiry\n"
        "- Priority: High (urgent or blocking issues), "
        "Medium (moderate issues), Low (general or informational queries)\n\n"
        "Return ONLY valid JSON with no extra text:\n"
        '{\n  "category": "",\n  "priority": ""\n}\n\n'
        f'Message: "{message}"'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,  # deterministic output
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()

    # Safely parse the JSON returned by OpenAI
    result = _safe_parse_json(content)

    # Validate returned values
    if result["category"] not in VALID_CATEGORIES:
        logger.warning(
            "Unexpected category '%s' for message: %s", result["category"], message
        )
    if result["priority"] not in VALID_PRIORITIES:
        logger.warning(
            "Unexpected priority '%s' for message: %s", result["priority"], message
        )

    return result


def _safe_parse_json(text: str) -> dict:
    """
    Attempt to parse JSON from the model's response.

    Handles cases where the model wraps the JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last lines (the fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model response as JSON: {text}") from exc

    if "category" not in data or "priority" not in data:
        raise ValueError(f"Missing required keys in model response: {data}")

    return data


# ---------------------------------------------------------------------------
# Unified Classification (OpenAI → Fallback)
# ---------------------------------------------------------------------------


def classify_message(client: OpenAI, message: str) -> dict:
    """
    Classify a message using OpenAI API. If the API is unavailable
    (no key, quota exceeded, etc.), automatically fall back to the
    keyword-based classifier.

    Returns:
        dict with keys "category" and "priority".
    """
    global _using_fallback

    # If no client or already in fallback mode, use keywords
    if client is None or _using_fallback:
        return classify_with_keywords(message)

    try:
        return classify_with_openai(client, message)
    except (RateLimitError, APIError) as api_err:
        if not _using_fallback:
            logger.warning(
                "OpenAI API unavailable (%s). Switching to fallback classifier.",
                type(api_err).__name__,
            )
            _using_fallback = True
        return classify_with_keywords(message)
    except AuthenticationError:
        if not _using_fallback:
            logger.warning(
                "Invalid API key. Switching to fallback classifier."
            )
            _using_fallback = True
        return classify_with_keywords(message)


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------


def process_messages(messages: list[str]) -> list[dict]:
    """
    Classify a list of customer support messages.

    Returns:
        List of dicts, each containing "message", "category", and "priority".
    """
    client = get_openai_client()
    results: list[dict] = []

    for idx, msg in enumerate(messages, start=1):
        logger.info("Processing message %d/%d: %s", idx, len(messages), msg[:80])
        try:
            classification = classify_message(client, msg)
            results.append(
                {
                    "message": msg,
                    "category": classification["category"],
                    "priority": classification["priority"],
                }
            )
        except ValueError as parse_err:
            logger.error("Parsing error for message '%s': %s", msg[:60], parse_err)
            results.append(
                {"message": msg, "category": "Error", "priority": "Error"}
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error for message '%s': %s", msg[:60], exc)
            results.append(
                {"message": msg, "category": "Error", "priority": "Error"}
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample input messages
    sample_messages = [
        "My payment got deducted but service is not activated",
        "App crashes every time I login",
        "How to change my email address?",
        "I want to know your business hours",
        "I am unable to reset my password and I'm locked out of my account",
    ]

    logger.info("Starting classification of %d messages...", len(sample_messages))

    output = process_messages(sample_messages)

    # Print mode used
    mode = "Fallback (keyword-based)" if _using_fallback else "OpenAI GPT-4o-mini"

    # Pretty-print structured JSON output
    print("\n" + "=" * 60)
    print("  CLASSIFICATION RESULTS")
    print(f"  Mode: {mode}")
    print("=" * 60)
    print(json.dumps(output, indent=2))

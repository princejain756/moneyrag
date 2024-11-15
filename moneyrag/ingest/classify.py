"""Turn a bank narration into a typed transaction: rail, merchant, category.

This is deliberately rules, not a model. Narration grammar is stable, the same
string always means the same thing, and a deterministic classifier can be
diffed, tested and reasoned about. An LLM fallback would only be worth its cost
for narrations no rule matches, and that rate is measured rather than assumed.

Two things are kept apart on purpose:

* `channel` is the payment rail (upi, imps, neft, atm, ...). It answers "how did
  the money move".
* refunds, charges, interest and salary are *flags*, not rails. A salary credited
  by NEFT is NEFT, and it is also salary. Collapsing those into one field is how
  classifications start contradicting each other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VPA = re.compile(r"[\w.\-]{2,}@[a-zA-Z]{2,}")
RRN = re.compile(r"\b\d{10,18}\b")
IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
TOKENS = re.compile(r"[^A-Z0-9@._\- ]+")

# Rails only. First match wins, so the narrower tokens come first.
CHANNEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rtgs", ("RTGS",)),
    ("neft", ("NEFT",)),
    ("imps", ("IMPS",)),
    ("upi", ("UPI/", "UPI-", "UPI ", "/UPI")),
    ("atm", ("ATM", "ATW", "CASH WDL", "CASH WITHDRAWAL")),
    ("nach", ("NACH", "ECS", "ACH/", "MANDATE")),
    ("emi", ("EMI", "LOAN INSTAL")),
    ("cheque", ("CHQ", "CHEQUE", "CLG")),
    ("card", ("POS ", "POS/", "VISA", "MASTERCARD", "RUPAY")),
    ("cash", ("CASH DEP", "CASH DEPOSIT", "BY CASH")),
)

REVERSAL_WORDS = ("REFUND", "REVERSAL", "REVERSED", "CHARGEBACK")
CHARGE_WORDS = ("CHRG", "CHARGE", "AMC", "GST", "MIN BAL")
INTEREST_WORDS = ("INT.PD", "INTEREST", "INT CR")
SALARY_WORDS = ("SALARY", "PAYROLL")
# "EMI" must be a whole word: substring matching it would fire on PREMIUM.
EMI = re.compile(r"\bEMI\b")
# A load into a prepaid wallet is money parked, not a payment to a person.
WALLETS = ("PAYTM", "PHONEPE", "MOBIKWIK", "FREECHARGE", "AMAZON PAY BALANCE")

# Aggregator and plumbing words that carry no merchant signal.
NOISE = {
    "UPI", "DR", "CR", "IMPS", "NEFT", "RTGS", "POS", "ACH", "NACH", "PAYU", "RAZORPAY",
    "BILLDESK", "CCAVENUE", "PAYTM", "PG", "PMT", "TXN", "REF", "NO", "SELF", "TO", "FROM",
    "BY", "VIA", "PAYMENT", "PAYMENTS", "TRANSFER", "BANK", "LTD", "LIMITED", "PVT",
    "INDIA", "SERVICES", "TECHNOLOGIES", "TECHNOLOGY", "PRIVATE", "THE", "AND", "CLG",
}

# merchant -> strings seen in real narrations (aggregator and legal-entity names).
MERCHANTS: dict[str, tuple[str, ...]] = {
    "swiggy": ("SWIGGY", "BUNDL", "INSTAMART"),
    "zomato": ("ZOMATO",),
    "bigbasket": ("BIGBASKET", "INNOVATIVE RETAIL"),
    "blinkit": ("BLINKIT", "GROFERS"),
    "zepto": ("ZEPTO", "KIRANAKART"),
    "dmart": ("DMART", "AVENUE SUPERMARTS"),
    "reliance_fresh": ("RELIANCE FRESH", "RELIANCE SMART"),
    "amazon": ("AMAZON", "AMZN"),
    "flipkart": ("FLIPKART",),
    "myntra": ("MYNTRA",),
    "meesho": ("MEESHO",),
    "croma": ("CROMA",),
    "reliance_digital": ("RELIANCE DIGITAL",),
    "decathlon": ("DECATHLON",),
    "uber": ("UBER",),
    "ola": ("OLA ", "ANI TECHNOLOGIES"),
    "rapido": ("RAPIDO",),
    "irctc": ("IRCTC", "INDIAN RAILWAY"),
    "makemytrip": ("MAKEMYTRIP",),
    "redbus": ("REDBUS",),
    "indigo": ("INDIGO", "INTERGLOBE"),
    "jio": ("JIO", "RELIANCE JIO"),
    "airtel": ("AIRTEL", "BHARTI"),
    "vodafone": ("VODAFONE", "VODAFONE IDEA"),
    "act_fibernet": ("ACT FIBERNET", "ATRIA CONVERGENCE"),
    "bescom": ("BESCOM", "BANGALORE ELECTRICITY"),
    "tata_power": ("TATA POWER",),
    "mahanagar_gas": ("MAHANAGAR GAS",),
    "netflix": ("NETFLIX",),
    "spotify": ("SPOTIFY",),
    "hotstar": ("HOTSTAR", "DISNEY"),
    "bookmyshow": ("BOOKMYSHOW", "BIGTREE"),
    "pvr": ("PVR", "INOX"),
    "zerodha": ("ZERODHA",),
    "groww": ("GROWW", "NEXTBILLION"),
    "upstox": ("UPSTOX", "RKSV"),
    "kuvera": ("KUVERA",),
    "indmoney": ("INDMONEY", "FINZOOM"),
    "apollo_pharmacy": ("APOLLO PHARMACY", "APOLLO PHARMA"),
    "pharmeasy": ("PHARMEASY",),
    "tata_1mg": ("1MG",),
    "practo": ("PRACTO",),
    "lic": ("LIC ", "LIFE INSURANCE", "INSURANCE PREMIUM"),
}

MERCHANT_CATEGORY: dict[str, str] = {
    "swiggy": "food_delivery", "zomato": "food_delivery",
    "bigbasket": "groceries", "blinkit": "groceries", "zepto": "groceries",
    "dmart": "groceries", "reliance_fresh": "groceries",
    "amazon": "shopping", "flipkart": "shopping", "myntra": "shopping",
    "meesho": "shopping", "croma": "shopping", "reliance_digital": "shopping",
    "decathlon": "shopping",
    "uber": "travel", "ola": "travel", "rapido": "travel", "irctc": "travel",
    "makemytrip": "travel", "redbus": "travel", "indigo": "travel",
    "jio": "utilities", "airtel": "utilities", "vodafone": "utilities",
    "act_fibernet": "utilities", "bescom": "utilities", "tata_power": "utilities",
    "mahanagar_gas": "utilities",
    "netflix": "entertainment", "spotify": "entertainment", "hotstar": "entertainment",
    "bookmyshow": "entertainment", "pvr": "entertainment",
    "zerodha": "investments", "groww": "investments", "upstox": "investments",
    "kuvera": "investments", "indmoney": "investments",
    "apollo_pharmacy": "medical", "pharmeasy": "medical", "tata_1mg": "medical",
    "practo": "medical",
    "lic": "insurance",
}

RAIL_CATEGORY = {"atm": "cash", "cash": "cash", "emi": "loan_emi"}


@dataclass(frozen=True)
class Classification:
    channel: str
    merchant: str | None
    category: str
    vpa: str | None
    counterparty: str | None
    is_self_transfer: bool
    is_reversal: bool
    matched_rule: str


def clean(narration: str) -> str:
    return re.sub(r"\s+", " ", TOKENS.sub(" ", narration.upper())).strip()


def detect_channel(narration: str) -> str:
    text = " " + clean(narration) + " "
    for channel, needles in CHANNEL_RULES:
        if any(n in text for n in needles):
            return channel
    return "other"


def find_merchant(cleaned: str) -> tuple[str | None, str]:
    for merchant, aliases in MERCHANTS.items():
        for alias in aliases:
            if alias.strip() in cleaned:
                return merchant, "alias"
    return None, "none"


def counterparty_name(cleaned: str) -> str | None:
    without_ids = RRN.sub(" ", IFSC.sub(" ", VPA.sub(" ", cleaned)))
    tokens = [t for t in re.split(r"[^A-Z]+", without_ids) if len(t) >= 3 and t not in NOISE]
    return max(tokens, key=len) if tokens else None


def classify(narration: str, direction: str = "debit",
             owner_aliases: tuple[str, ...] = ()) -> Classification:
    cleaned = clean(narration)
    channel = detect_channel(narration)
    has = lambda words: any(w in cleaned for w in words)  # noqa: E731

    is_reversal = has(REVERSAL_WORDS)
    own = any(alias.upper() in cleaned for alias in owner_aliases if alias.strip())
    self_transfer = own or "SELF" in cleaned

    merchant, how = find_merchant(cleaned)
    counterparty = counterparty_name(cleaned)
    vpa_match = VPA.search(narration)

    if self_transfer:
        category = "transfer"
    elif is_reversal:
        category = "other"        # the linker rewrites this from the original
    elif has(CHARGE_WORDS):
        category = "fees"
    elif has(INTEREST_WORDS):
        category = "income"
    elif has(SALARY_WORDS):
        category = "salary"
    elif EMI.search(cleaned) or "LOAN INSTAL" in cleaned:
        # An EMI mandate arrives over NACH: the rail is nach, the money is a loan instalment.
        category = "loan_emi"
    elif merchant in MERCHANT_CATEGORY:
        category = MERCHANT_CATEGORY[merchant]
    elif channel in RAIL_CATEGORY:
        category = RAIL_CATEGORY[channel]
    elif has(WALLETS):
        category = "other"
    elif channel in {"upi", "imps"}:
        # No known merchant on a UPI leg is usually a person, not a shop.
        category = "person"
    else:
        category = "other"

    return Classification(
        channel=channel, merchant=merchant, category=category,
        vpa=vpa_match.group(0).lower() if vpa_match else None,
        counterparty=counterparty, is_self_transfer=self_transfer,
        is_reversal=is_reversal, matched_rule=how,
    )

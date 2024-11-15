"""Statement ingestion: parsing, classification, reconciliation, ledger integration."""
import sqlite3

import pytest

from moneyrag.ingest import (classify, infer_mapping, ingest, parse_amount_paise, read_rows,
                             reconcile, to_ledger)
from moneyrag.ingest.cli import main as cli_main
from moneyrag.ledger import query_transactions

STATEMENT_A = "data/statements/example_a.csv"
STATEMENT_B = "data/statements/example_b.csv"


def read(path):
    from pathlib import Path
    return Path(path).read_text()


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,paise", [
    ("1,20,500.50", 12050050),
    ("120500", 12050000),
    ("₹1,234.50", 123450),
    ("Rs 42.00", 4200),
    ("(1,000.00)", -100000),
    ("", None),
    ("-", None),
    (None, None),
])
def test_indian_amount_formats(text, paise):
    assert parse_amount_paise(text) == paise


def test_header_detection_handles_two_bank_layouts():
    a = infer_mapping(["Date", "Narration", "Chq/Ref No", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
    b = infer_mapping(["Date", "Particulars", "Chq/Ref No", "Debit", "Credit", "Closing Balance"])
    assert a and a.narration == "Narration" and a.debit == "Withdrawal Amt."
    assert b and b.narration == "Particulars" and b.credit == "Credit"
    assert infer_mapping(["Foo", "Bar"]) is None


def test_unparseable_rows_are_reported_not_dropped():
    text = ('Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance\n'
            'not-a-date,UPI/DR/x,100.00,,900.00\n'
            '01/04/2026,,,,900.00\n'
            '02/04/2026,UPI/DR/y,50.00,,850.00\n')
    result = read_rows(text)
    assert len(result.rows) == 1
    assert [reason for _, reason in result.skipped] == ["unparseable date", "no amount"]


# --- reconciliation --------------------------------------------------------

def test_reconciliation_passes_on_both_fixtures():
    for path in (STATEMENT_A, STATEMENT_B):
        assert reconcile(read_rows(read(path)).rows).ok


def test_ascending_and_descending_are_detected():
    assert reconcile(read_rows(read(STATEMENT_A)).rows).order == "ascending"
    assert reconcile(read_rows(read(STATEMENT_B)).rows).order == "descending"


def test_a_wrong_amount_breaks_the_balance_chain():
    """The one thing a bad parse cannot hide: the bank's own running balance."""
    text = read(STATEMENT_A).replace("1287.50", "1288.50")
    report = reconcile(read_rows(text).rows)
    assert not report.ok
    assert report.mismatches and report.mismatches[0]["off_by_paise"] == 100


def test_cli_strict_mode_exits_nonzero_on_a_broken_statement(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text(read(STATEMENT_A).replace("1287.50", "1288.50"))
    assert cli_main([str(broken), "--strict"]) == 1
    assert cli_main([STATEMENT_A, "--strict"]) == 0


# --- classification --------------------------------------------------------

def test_rail_and_semantics_are_separate_fields():
    c = classify("NEFT CR-ACME PAYROLL-SALARY APR 2026", "credit")
    assert c.channel == "neft" and c.category == "salary"


def test_emi_is_a_whole_word_not_a_substring_of_premium():
    # "PREMIUM" contains "EMI"; the first version of this rule misfiled insurance as a loan.
    assert classify("NACH DR-TATA A1 LIFE INSURANCE PREMIUM", "debit").category == "insurance"
    assert classify("NACH/DR/HDFC BANK LTD-ACH/EMI HOME LOAN", "debit").category == "loan_emi"


def test_wallet_load_is_not_a_person():
    assert classify("UPI/DR/4123/PAYTM/YESB/paytm@ybl", "debit").category == "other"
    assert classify("UPI/DR/4123/ZEESHAN KHAN/HDFC/z@okhdfcbank", "debit").category == "person"


def test_aggregator_name_maps_to_the_merchant_behind_it():
    assert classify("UPI/DR/4123/BUNDL TECHNOLOGIES/YESB/bundl@ybl", "debit").merchant == "swiggy"


def test_self_transfer_needs_an_owner_alias():
    narration = "IMPS/DR/412345678904/PRINCE JAIN/ACME"
    assert not classify(narration, "debit").is_self_transfer
    assert classify(narration, "debit", ("PRINCE JAIN",)).is_self_transfer


def test_refund_inherits_category_and_merchant_from_the_original():
    result = ingest(read(STATEMENT_A), owner_aliases=("PRINCE JAIN",))
    refund = next(t for t in result.txns if "SWIGGY REFUND" in t.narration)
    original = result.txns[refund.reversal_of]
    assert original.merchant == "swiggy"
    assert (refund.category, refund.merchant) == (original.category, original.merchant)


def test_equal_and_opposite_rows_across_accounts_are_paired():
    result = ingest(read(STATEMENT_A), owner_aliases=("PRINCE JAIN",))
    paired = [t for t in result.txns if t.is_self_transfer]
    assert len(paired) == 2 and {t.direction for t in paired} == {"debit", "credit"}


# --- integration with the existing query tool -------------------------------

def test_imported_rows_are_queryable_by_the_typed_tool(tmp_path):
    result = ingest(read(STATEMENT_A), owner_aliases=("PRINCE JAIN",))
    db = tmp_path / "imported.db"
    assert to_ledger(db, result.txns) == len(result.txns)
    conn = sqlite3.connect(db)
    # The tool sums debits by default: 349 + 1287.50 + 487.
    # The 349 refund is a credit, so it is not netted off here. Netting refunds
    # needs the reversal link, which the SQL tool does not know about — a real gap.
    assert query_transactions(conn, op="sum", category="food_delivery")["total"] == "Rs 2,123.50"
    assert query_transactions(conn, op="sum", category="food_delivery",
                              direction="credit")["total"] == "Rs 349.00"

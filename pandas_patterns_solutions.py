"""Pandas patterns weekend: drill solutions
Run top to bottom: later tasks use earlier ones."""

# ---- Task 1. Set up the practice data (run this first; it's the same data as in the notes).
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
clients = pd.DataFrame({
    "client_id": [f"C{i:04d}" for i in range(300)],
    "source": rng.choice(["Meta-IMR", "Google", "Referral", "Website"], 300),
    "broker": rng.choice(["T66541", "V28482", "M148910", None], 300),
    "status": rng.choice(["Active", "Inactive", "Sleeping", "Dormant"], 300),
    "created_at": pd.Timestamp("2026-08-01")
                  + pd.to_timedelta(rng.integers(0, 100, 300), unit="D"),
})
n = 2_000
kind = rng.choice(["BILL", "DEPOSIT", "WITHDRAW"], n, p=[0.6, 0.3, 0.1])
amount = rng.integers(10, 3_000, n).astype(float)
bill_loss = (kind == "BILL") & (rng.random(n) < 0.55)          # a BILL is a loss or a profit
debit_rows = (kind == "WITHDRAW") | bill_loss
credit_rows = (kind == "DEPOSIT") | ((kind == "BILL") & ~bill_loss)
ledger = pd.DataFrame({
    "ClientID": rng.choice(clients["client_id"], n),
    "CreatedAt": pd.Timestamp("2026-09-01")
                 + pd.to_timedelta(rng.integers(0, 80 * 1440, n), unit="min"),
    "LedgerType": kind,
    "Debit": np.where(debit_rows, amount, 0.0),
    "Credit": np.where(credit_rows, amount, 0.0),
})
print(clients.shape, ledger.shape)

# ---- Task 2. Pattern 2: rename the ledger columns to snake_case in one line.
led = ledger.rename(columns=lambda c: "".join("_" + ch.lower() if ch.isupper() else ch
                                             for ch in c).lstrip("_"))
print(list(led.columns))

# ---- Task 3. Pattern 3: make LedgerType a category and print how much memory that saves.
before = ledger["LedgerType"].memory_usage(deep=True)
after = ledger["LedgerType"].astype("category").memory_usage(deep=True)
print(f"{before:,} -> {after:,} bytes")

# ---- Task 4. Pattern 4: keep BILL rows in October with Debit above 1,000, using named conditions.
is_bill = ledger["LedgerType"].eq("BILL")
in_oct = ledger["CreatedAt"].between("2026-10-01", "2026-11-01", inclusive="left")
big_loss = ledger["Debit"].gt(1000)
print(len(ledger[is_bill & in_oct & big_loss]), "rows")

# ---- Task 5. Pattern 5: keep ledger rows for clients from Meta-IMR, using a set and isin.
meta_ids = set(clients.loc[clients["source"].eq("Meta-IMR"), "client_id"])
print(ledger["ClientID"].isin(meta_ids).sum(), "rows")

# ---- Task 6. Pattern 6: in one assign(), add net (Credit - Debit) and the month as a Period.
out = ledger.assign(net=lambda t: t["Credit"] - t["Debit"],
                    month=lambda t: t["CreatedAt"].dt.to_period("M"))
print(out[["net", "month"]].head(3))

# ---- Task 7. Pattern 7: label each row 'Loss', 'Profit' or 'Flat' with np.select (no apply).
label = np.select([ledger["Debit"].gt(ledger["Credit"]), ledger["Credit"].gt(ledger["Debit"])],
                  ["Loss", "Profit"], default="Flat")
print(pd.Series(label).value_counts().to_dict())

# ---- Task 8. Pattern 8: add each row's source with map, from the clients table.
source_of = clients.set_index("client_id")["source"]
print(ledger["ClientID"].map(source_of).value_counts().to_dict())

# ---- Task 9. Pattern 9: whole calendar days from each ledger row to 2026-11-20.
days = (pd.Timestamp("2026-11-20") - ledger["CreatedAt"].dt.normalize()).dt.days
print(days.describe()[["min", "max"]].to_dict())

# ---- Task 10. Pattern 10: clean messy IDs (' c0001', 'C0002 ') to upper case without spaces.
messy = pd.Series([" c0001", "C0002 ", "c0003", None])
print(messy.str.strip().str.upper().tolist())

# ---- Task 11. Pattern 11: per client, the number of BILL rows, total Debit and the last ledger date.
bills = ledger[ledger["LedgerType"].eq("BILL")]
per_client = bills.groupby("ClientID").agg(bills=("Debit", "size"),
                                          total_debit=("Debit", "sum"),
                                          last_at=("CreatedAt", "max"))
print(per_client.head(3))

# ---- Task 12. Pattern 12: add each row's share of its client's total Credit, with transform.
share = ledger["Credit"] / ledger.groupby("ClientID")["Credit"].transform("sum")
print(round(share.groupby(ledger["ClientID"]).sum().head(3), 6).tolist())

# ---- Task 13. Pattern 13: per broker, how many clients are Active and how many are Dormant.
flags = clients.assign(active=clients["status"].eq("Active"),
                       dormant=clients["status"].eq("Dormant"))
print(flags.groupby("broker", dropna=False)[["active", "dormant"]].sum())

# ---- Task 14. Pattern 14: a table of ledger rows per type (columns) and month (rows).
t = (ledger.assign(month=ledger["CreatedAt"].dt.to_period("M"))
       .groupby(["month", "LedgerType"]).size().unstack(fill_value=0))
print(t)

# ---- Task 15. Pattern 15: each client's first DEPOSIT row (amount and date).
first_dep = (ledger[ledger["LedgerType"].eq("DEPOSIT")]
             .sort_values("CreatedAt").drop_duplicates("ClientID"))
print(len(first_dep), "clients")
print(first_dep[["ClientID", "CreatedAt", "Credit"]].iloc[0].tolist())

# ---- Task 16. Pattern 16: a running total of Credit per client, and the gap in days since that client's previous row.
ordered = ledger.sort_values(["ClientID", "CreatedAt"])
ordered = ordered.assign(
    running_credit=ordered.groupby("ClientID")["Credit"].cumsum(),
    gap_days=ordered.groupby("ClientID")["CreatedAt"].diff().dt.days)
print(ordered[["ClientID", "Credit", "running_credit", "gap_days"]].head(4))

# ---- Task 17. Pattern 17: reconcile client IDs between clients and ledger with merge(indicator=True).
ids = clients[["client_id"]].merge(ledger[["ClientID"]].drop_duplicates(),
                                   left_on="client_id", right_on="ClientID",
                                   how="outer", indicator=True)
print(ids["_merge"].value_counts().to_dict())

# ---- Task 18. Pattern 18: deposits per source for September and October, stacked into one table.
dep = ledger[ledger["LedgerType"].eq("DEPOSIT")].assign(
    source=lambda t: t["ClientID"].map(clients.set_index("client_id")["source"]))
months = {m: dep[dep["CreatedAt"].dt.to_period("M") == m].groupby("source")["Credit"].sum()
          for m in ["2026-09", "2026-10"]}
print(pd.concat(months, names=["month"]).reset_index().head(4))

# ---- Task 19. Pattern 19: for each call, the first ledger row at or after it for that client.
calls = pd.DataFrame({"ClientID": ["C0001", "C0002", "C0001"],
                      "called_at": pd.to_datetime(["2026-09-20", "2026-10-05", "2026-10-25"])})
nxt = pd.merge_asof(calls.sort_values("called_at"), ledger.sort_values("CreatedAt"),
                    left_on="called_at", right_on="CreatedAt",
                    by="ClientID", direction="forward")
print(nxt[["ClientID", "called_at", "CreatedAt", "LedgerType"]])

# ---- Task 20. Pattern 20: turn a wide table (Debit and Credit columns) into long form, one row per amount.
long = ledger.head(3).melt(id_vars=["ClientID", "CreatedAt"], value_vars=["Debit", "Credit"],
                           var_name="side", value_name="amount")
print(long)

# ---- Task 21. Pattern 21: write a three-step pipeline with pipe(): BILL rows, add month, total per month.
def only_bills(t):
    return t[t["LedgerType"].eq("BILL")]

def add_month(t):
    return t.assign(month=t["CreatedAt"].dt.to_period("M"))

print(ledger.pipe(only_bills).pipe(add_month).groupby("month")["Debit"].sum())

# ---- Task 22. Pattern 23: replace an iterrows loop that totals Credit for DEPOSIT rows with one vectorised line.
loop_total = 0.0
for row in ledger.itertuples(index=False):         # if you must loop, itertuples, not iterrows
    if row.LedgerType == "DEPOSIT":
        loop_total += row.Credit
vector_total = ledger.loc[ledger["LedgerType"].eq("DEPOSIT"), "Credit"].sum()
print(loop_total == vector_total, vector_total)

# ---- Task 23. Pattern 24: set Debit to 0 for WITHDRAW rows on a copy of the ledger, the right way.
fixed = ledger.copy()
fixed.loc[fixed["LedgerType"].eq("WITHDRAW"), "Debit"] = 0.0
print(fixed.loc[fixed["LedgerType"].eq("WITHDRAW"), "Debit"].sum(), "| original untouched:",
      ledger.loc[ledger["LedgerType"].eq("WITHDRAW"), "Debit"].sum() > 0)

# ---- Task 24. Pattern 25: merge clients onto the ledger safely and assert nothing was duplicated.
n_before = len(ledger)
merged = ledger.merge(clients[["client_id", "broker"]],
                      left_on="ClientID", right_on="client_id",
                      how="left", validate="many_to_one")
assert len(merged) == n_before, "the merge duplicated rows"
print(n_before, "->", len(merged))

# ---- Task 25. Pattern 22: save two tables to one Excel file with a sheet each.
with pd.ExcelWriter("weekend_drill.xlsx") as writer:
    per_client.to_excel(writer, sheet_name="Per client")
    t.to_excel(writer, sheet_name="By month")
print(pd.ExcelFile("weekend_drill.xlsx").sheet_names)

# ---- Task 26. Challenge: per source, the number of clients, clients who deposited in October, October deposits, and all-time deposits. One row per source, biggest October deposits first.
deps = ledger[ledger["LedgerType"].eq("DEPOSIT")]
oct_mask = deps["CreatedAt"].between("2026-10-01", "2026-11-01", inclusive="left")
oct_by_client = deps[oct_mask].groupby("ClientID")["Credit"].sum()
all_by_client = deps.groupby("ClientID")["Credit"].sum()
per = clients.assign(oct_deposit=clients["client_id"].map(oct_by_client).fillna(0),
                     all_deposit=clients["client_id"].map(all_by_client).fillna(0))
summary = (per.groupby("source")
              .agg(clients=("client_id", "size"),
                   oct_depositors=("oct_deposit", lambda s: s.gt(0).sum()),
                   oct_deposit=("oct_deposit", "sum"),
                   all_deposit=("all_deposit", "sum"))
              .sort_values("oct_deposit", ascending=False))
assert summary["oct_deposit"].sum() == deps.loc[oct_mask, "Credit"].sum()
print(summary)

# ---- Task 27. Challenge: per broker, the client with the largest total Debit, and that total.
debit = ledger.groupby("ClientID", as_index=False)["Debit"].sum()
debit = debit.merge(clients[["client_id", "broker"]], left_on="ClientID", right_on="client_id",
                    validate="one_to_one")
idx = debit.groupby("broker", dropna=False)["Debit"].idxmax()
top = debit.loc[idx, ["broker", "ClientID", "Debit"]]
print(top.to_string(index=False))

"""Data preparation and analytics for the Smart Expense Analyzer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = ("date", "description", "montant", "categorie")
MIN_MONTHS_FOR_FORECAST = 6
MIN_TRANSACTIONS_FOR_ANOMALIES = 10


def load_transactions(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load a CSV and return validated, consistently typed transactions."""
    try:
        raw = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Impossible de lire le fichier CSV : {error}") from error
    return prepare_transactions(raw)


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize transaction fields and reject rows that cannot be analyzed."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in transactions.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires manquantes : " + ", ".join(missing_columns)
        )

    cleaned = transactions.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["description"] = cleaned["description"].fillna("").astype(str).str.strip()
    cleaned["categorie"] = cleaned["categorie"].fillna("").astype(str).str.strip()

    amounts = cleaned["montant"].astype(str).str.strip()
    amounts = amounts.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    amounts = amounts.str.replace(",", ".", regex=False)
    cleaned["montant"] = pd.to_numeric(amounts, errors="coerce")

    invalid_date = cleaned["date"].isna()
    invalid_amount = cleaned["montant"].isna() | (cleaned["montant"] <= 0)
    empty_description = cleaned["description"].eq("")
    empty_category = cleaned["categorie"].eq("")
    invalid_rows = invalid_date | invalid_amount | empty_description | empty_category
    if invalid_rows.any():
        row_numbers = (cleaned.index[invalid_rows] + 2).tolist()
        details: list[str] = []
        if invalid_date.any():
            details.append(f"{int(invalid_date.sum())} date(s) invalide(s)")
        if invalid_amount.any():
            details.append(f"{int(invalid_amount.sum())} montant(s) invalide(s) ou non positif(s)")
        if empty_description.any():
            details.append(f"{int(empty_description.sum())} description(s) vide(s)")
        if empty_category.any():
            details.append(f"{int(empty_category.sum())} catégorie(s) vide(s)")
        raise ValueError(
            f"Données invalides ({'; '.join(details)}). "
            f"Vérifie les lignes CSV : {', '.join(map(str, row_numbers[:10]))}."
        )

    return cleaned.sort_values("date").reset_index(drop=True)


def monthly_spending(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return total spending for each calendar month present in the data."""
    result = (
        transactions.assign(mois=transactions["date"].dt.to_period("M"))
        .groupby("mois", as_index=False)["montant"]
        .sum()
        .sort_values("mois")
        .reset_index(drop=True)
    )
    return result


def category_spending(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return spending totals and shares by category."""
    summary = (
        transactions.groupby("categorie", as_index=False)["montant"]
        .sum()
        .sort_values("montant", ascending=False)
        .reset_index(drop=True)
    )
    summary["part_pct"] = summary["montant"] / summary["montant"].sum() * 100
    return summary


def detect_anomalies(
    transactions: pd.DataFrame, contamination: float = 0.05
) -> pd.DataFrame:
    """Flag unusual transaction amounts with Isolation Forest."""
    if len(transactions) < MIN_TRANSACTIONS_FOR_ANOMALIES:
        raise ValueError(
            "Isolation Forest nécessite au moins "
            f"{MIN_TRANSACTIONS_FOR_ANOMALIES} transactions."
        )
    if not 0 < contamination <= 0.5:
        raise ValueError("La contamination doit être comprise entre 0 et 0,5.")
    if transactions["montant"].nunique() < 3:
        raise ValueError("Il faut au moins trois montants différents pour détecter des anomalies.")

    result = transactions.copy()
    features = np.log1p(result[["montant"]].to_numpy())
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    model.fit(features)
    result["score_anomalie"] = -model.score_samples(features)
    result["anomalie"] = model.predict(features) == -1
    return result.sort_values("score_anomalie", ascending=False).reset_index(drop=True)


def cluster_months(transactions: pd.DataFrame) -> pd.DataFrame:
    """Group months by category spending profile using K-Means."""
    monthly = (
        transactions.assign(mois=transactions["date"].dt.to_period("M"))
        .pivot_table(
            index="mois",
            columns="categorie",
            values="montant",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )
    if len(monthly) < 3:
        raise ValueError("K-Means nécessite au moins trois mois de transactions.")

    cluster_count = min(3, len(monthly))
    scaled = StandardScaler().fit_transform(monthly)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(scaled)
    return monthly.assign(groupe=labels).reset_index()


def forecast_next_month(transactions: pd.DataFrame) -> dict[str, float | pd.Period | None]:
    """Forecast next month's total using linear regression and a naive baseline."""
    monthly = monthly_spending(transactions)
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        raise ValueError(
            "La prévision nécessite au moins "
            f"{MIN_MONTHS_FOR_FORECAST} mois avec des transactions."
        )

    month_ordinals = np.array([period.ordinal for period in monthly["mois"]]).reshape(-1, 1)
    amounts = monthly["montant"].to_numpy(dtype=float)
    model = LinearRegression().fit(month_ordinals, amounts)
    next_month = monthly["mois"].max() + 1
    predicted_amount = max(0.0, float(model.predict([[next_month.ordinal]])[0]))

    backtest_errors: list[float] = []
    if len(monthly) >= MIN_MONTHS_FOR_FORECAST + 2:
        for index in range(MIN_MONTHS_FOR_FORECAST, len(monthly)):
            backtest_model = LinearRegression().fit(
                month_ordinals[:index], amounts[:index]
            )
            backtest_prediction = max(
                0.0, float(backtest_model.predict(month_ordinals[index : index + 1])[0])
            )
            backtest_errors.append(abs(backtest_prediction - amounts[index]))

    mae = float(np.mean(backtest_errors)) if backtest_errors else None
    return {
        "mois": next_month,
        "montant_prevu": predicted_amount,
        "reference_mois_precedent": float(amounts[-1]),
        "mae_retrospective": mae,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Analyse un CSV de dépenses.")
    parser.add_argument(
        "csv",
        nargs="?",
        default=Path(__file__).with_name("transactions_fictives.csv"),
        type=Path,
        help="Chemin vers le CSV (par défaut : transactions_fictives.csv).",
    )
    arguments = parser.parse_args()

    transactions = load_transactions(arguments.csv)
    print(f"{len(transactions)} transactions valides")
    print("\nDépenses mensuelles (DH)")
    print(monthly_spending(transactions).to_string(index=False))
    print("\nDépenses par catégorie (DH)")
    print(category_spending(transactions).to_string(index=False))


if __name__ == "__main__":
    _main()

import unittest

import pandas as pd

from analysis import (
    category_spending,
    cluster_months,
    detect_anomalies,
    forecast_next_month,
    load_transactions,
    monthly_spending,
    prepare_transactions,
)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transactions = load_transactions("transactions_fictives.csv")

    def test_sample_data_is_valid_and_has_twelve_months(self) -> None:
        self.assertEqual(len(self.transactions), 84)
        self.assertEqual(len(monthly_spending(self.transactions)), 12)
        self.assertTrue((self.transactions["montant"] > 0).all())

    def test_monthly_and_category_totals_match_transaction_total(self) -> None:
        total = self.transactions["montant"].sum()
        self.assertAlmostEqual(monthly_spending(self.transactions)["montant"].sum(), total)
        categories = category_spending(self.transactions)
        self.assertAlmostEqual(categories["montant"].sum(), total)
        self.assertAlmostEqual(categories["part_pct"].sum(), 100)

    def test_invalid_rows_are_reported_instead_of_silently_dropped(self) -> None:
        invalid = pd.DataFrame(
            {
                "date": ["not-a-date"],
                "description": ["Test"],
                "montant": [-10],
                "categorie": ["Loisirs"],
            }
        )
        with self.assertRaisesRegex(ValueError, "lignes CSV"):
            prepare_transactions(invalid)

    def test_models_return_expected_outputs(self) -> None:
        anomalies = detect_anomalies(self.transactions)
        self.assertIn("anomalie", anomalies.columns)
        self.assertTrue(anomalies["anomalie"].any())

        groups = cluster_months(self.transactions)
        self.assertEqual(len(groups), 12)
        self.assertGreaterEqual(groups["groupe"].nunique(), 2)

        forecast = forecast_next_month(self.transactions)
        self.assertEqual(str(forecast["mois"]), "2026-01")
        self.assertGreater(forecast["montant_prevu"], 0)
        self.assertIsNotNone(forecast["mae_retrospective"])

    def test_forecast_requires_sufficient_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "au moins 6 mois"):
            forecast_next_month(self.transactions.iloc[:20])


if __name__ == "__main__":
    unittest.main()

"""Streamlit dashboard for personal expense analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from analysis import (
    REQUIRED_COLUMNS,
    category_spending,
    cluster_months,
    detect_anomalies,
    forecast_next_month,
    load_transactions,
    monthly_spending,
)

st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #202a35;
        --muted: #6b7785;
        --line: #e6e9ed;
        --accent: #315f55;
        --canvas: #fbfbfa;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    .block-container {
        max-width: 1320px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    h1 { font-size: 2rem; font-weight: 600; }
    h2 { font-size: 1.2rem; font-weight: 600; }
    h3 { font-size: 1rem; font-weight: 600; }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        padding: 1rem 1.1rem;
        border-radius: 5px;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.8rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 600;
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        font-size: 0.9rem;
        padding: 0.65rem 0;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent);
        border-bottom-color: var(--accent);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 5px;
    }
    [data-testid="stAlert"] {
        border-radius: 5px;
    }
    [data-testid="stSidebar"] {
        background: #f5f6f4;
        border-right: 1px solid var(--line);
    }
    .stDownloadButton button {
        border-radius: 4px;
        border-color: var(--line);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Expense Analyzer")
st.caption("Analyse des dépenses et habitudes financières")

with st.sidebar:
    st.header("Données")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    st.caption("Colonnes requises : date, description, montant, categorie.")
    st.caption("Les montants doivent être positifs et exprimés dans la même devise.")

if uploaded_file is not None:
    try:
        transactions = load_transactions(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    source_label = uploaded_file.name
else:
    sample_path = Path(__file__).with_name("transactions_fictives.csv")
    if not sample_path.exists():
        st.error("Le CSV de démonstration est introuvable. Importe un fichier CSV pour commencer.")
        st.stop()
    try:
        transactions = load_transactions(sample_path)
    except ValueError as error:
        st.error(f"Le CSV de démonstration est invalide : {error}")
        st.stop()
    source_label = "transactions_fictives.csv (données de démonstration)"

duplicate_count = int(transactions.duplicated(subset=list(REQUIRED_COLUMNS)).sum())
monthly = monthly_spending(transactions)
categories = category_spending(transactions)
total_spending = float(transactions["montant"].sum())
average_month = total_spending / len(monthly)
main_category = str(categories.iloc[0]["categorie"])

st.info(f"Source : {source_label} · {len(transactions)} transactions analysées")
if duplicate_count:
    st.warning(
        f"{duplicate_count} transaction(s) identique(s) détectée(s). "
        "Elles sont conservées dans les calculs : vérifie qu'il ne s'agit pas de doublons."
    )

metric_total, metric_average, metric_category, metric_count = st.columns(4)
metric_total.metric("Dépenses totales", f"{total_spending:,.0f} DH".replace(",", " "))
metric_average.metric("Moyenne par mois observé", f"{average_month:,.0f} DH".replace(",", " "))
metric_category.metric("Catégorie principale", main_category)
metric_count.metric("Transactions", f"{len(transactions)}")

overview_tab, anomaly_tab, trends_tab, data_tab = st.tabs(
    ["Vue d'ensemble", "Anomalies", "Tendances & modèles", "Transactions"]
)

with overview_tab:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Dépenses mensuelles")
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(
            monthly["mois"].astype(str),
            monthly["montant"],
            marker="o",
            markersize=4,
            linewidth=2,
            color="#315f55",
        )
        axis.set_ylabel("Montant (DH)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(colors="#6b7785", labelsize=8, length=0)
        axis.grid(axis="y", color="#e6e9ed", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#e6e9ed")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with chart_right:
        st.subheader("Répartition par catégorie")
        figure, axis = plt.subplots(figsize=(7, 4))
        bars = axis.barh(
            categories["categorie"],
            categories["montant"],
            color="#66877f",
            height=0.58,
        )
        axis.invert_yaxis()
        axis.set_xlabel("Dépenses (DH)")
        axis.set_ylabel("")
        axis.set_xlim(0, float(categories["montant"].max()) * 1.4)
        axis.tick_params(axis="both", colors="#6b7785", labelsize=9, length=0)
        axis.grid(axis="x", color="#e6e9ed", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#e6e9ed")
        for bar, amount, share in zip(
            bars,
            categories["montant"],
            categories["part_pct"],
            strict=True,
        ):
            axis.annotate(
                f"{amount:,.0f} DH · {share:.0f}%".replace(",", " "),
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color="#4f5b66",
                fontsize=8,
            )
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    st.subheader("Dépenses par catégorie")
    display_categories = categories.rename(
        columns={"categorie": "Catégorie", "montant": "Dépenses (DH)", "part_pct": "Part (%)"}
    ).copy()
    display_categories["Part (%)"] = display_categories["Part (%)"].round(1)
    st.dataframe(display_categories, hide_index=True, use_container_width=True)

with anomaly_tab:
    st.subheader("Transactions inhabituelles")
    st.caption(
        "Isolation Forest repère des montants atypiques dans cet échantillon. "
        "Une alerte est à vérifier et ne signifie pas qu'une transaction est erronée."
    )
    try:
        anomaly_results = detect_anomalies(transactions)
    except ValueError as error:
        st.info(str(error))
    else:
        alerts = anomaly_results.loc[anomaly_results["anomalie"]].copy()
        if alerts.empty:
            st.success("Aucune anomalie n'a été signalée.")
        else:
            alerts = alerts.rename(
                columns={
                    "date": "Date",
                    "description": "Description",
                    "montant": "Montant (DH)",
                    "categorie": "Catégorie",
                    "score_anomalie": "Score d'atypie",
                }
            )
            st.dataframe(
                alerts[["Date", "Description", "Montant (DH)", "Catégorie", "Score d'atypie"]],
                hide_index=True,
                use_container_width=True,
            )

with trends_tab:
    st.subheader("Prévision du mois suivant")
    try:
        forecast = forecast_next_month(transactions)
    except ValueError as error:
        st.info(str(error))
    else:
        forecast_columns = st.columns(3)
        forecast_columns[0].metric(
            f"Prévision · {forecast['mois']}",
            f"{forecast['montant_prevu']:,.0f} DH".replace(",", " "),
        )
        forecast_columns[1].metric(
            "Référence · dernier mois",
            f"{forecast['reference_mois_precedent']:,.0f} DH".replace(",", " "),
        )
        mae = forecast["mae_retrospective"]
        if mae is None:
            forecast_columns[2].metric("Erreur rétrospective moyenne", "Non disponible")
            st.caption("Ajoute davantage de mois pour évaluer la qualité des prévisions.")
        else:
            forecast_columns[2].metric(
                "Erreur rétrospective moyenne",
                f"{mae:,.0f} DH".replace(",", " "),
            )
        st.caption(
            "Régression linéaire sur les totaux mensuels historiques. "
            "Cette estimation est indicative et ne tient pas compte des événements futurs."
        )

    st.subheader("Profils mensuels (K-Means)")
    st.caption("Les groupes sont exploratoires ; leurs numéros ne représentent pas un classement.")
    try:
        clusters = cluster_months(transactions)
    except ValueError as error:
        st.info(str(error))
    else:
        clusters = clusters.rename(columns={"mois": "Mois", "groupe": "Groupe"})
        clusters["Mois"] = clusters["Mois"].astype(str)
        st.dataframe(clusters, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("Transactions chargées")
    st.dataframe(transactions, hide_index=True, use_container_width=True)
    csv_data = transactions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les données validées",
        data=csv_data,
        file_name="transactions_validees.csv",
        mime="text/csv",
    )

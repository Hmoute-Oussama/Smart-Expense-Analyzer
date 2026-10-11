# Smart Expense Analyzer

Application de démonstration qui analyse des transactions fictives ou importées,
visualise les dépenses et présente des résultats exploratoires de machine learning.

## Fonctionnalités

- Import et validation d'un CSV de transactions.
- Synthèse des dépenses par mois et par catégorie.
- Détection de montants atypiques avec Isolation Forest.
- Regroupement de profils mensuels avec K-Means.
- Prévision indicative du mois suivant par régression linéaire, avec comparaison
  à une référence basée sur le dernier mois et une erreur rétrospective.
- Téléchargement des transactions validées.

## Format du CSV

Le fichier doit contenir les colonnes `date`, `description`, `montant` et
`categorie`. Les montants doivent être positifs, numériques et exprimés dans
une seule devise. Le séparateur attendu est la virgule ; le point ou la virgule
peuvent servir de séparateur décimal. Les dates doivent être comprises par
Pandas (par exemple `2025-01-31`).

Le projet est livré avec `transactions_fictives.csv`, un jeu de démonstration
qui ne contient aucune donnée bancaire réelle.

## Installation sous Windows

Dans PowerShell, depuis le dossier du projet :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Lance le dashboard :

```powershell
streamlit run app.py
```

Lance les tests :

```powershell
python -m unittest -v
```

Une analyse textuelle du CSV de démonstration est aussi disponible :

```powershell
python analysis.py
```

## Limites des modèles

Les anomalies sont des alertes statistiques à vérifier, pas des conclusions de
fraude. K-Means produit des groupes exploratoires, dont les numéros sont
arbitraires. La prévision linéaire est indicative : elle s'appuie uniquement sur
les montants mensuels historiques et ne connaît ni les revenus ni les dépenses
futures. Les données fictives servent à tester le pipeline, pas à tirer des
conclusions financières.

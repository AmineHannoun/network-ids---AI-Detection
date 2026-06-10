# Network IDS/IPS — Détection d'attaques réseau par IA

> **Quand le modèle détecte une attaque, il a raison à ~88%**  
> **Il détecte 93% de toutes les attaques présentes dans le trafic**

---

Détection d'intrusions réseau par apprentissage automatique sur le dataset NSL-KDD.

## Objectif

Identifier automatiquement des comportements réseau malveillants à partir de
features extraites de logs de trafic réseau, en simulant un système IDS/IPS réel.

## Dataset

NSL-KDD — version améliorée du KDD Cup 1999
- 125,973 connexions pour l'entraînement
- 22,544 connexions pour le test
- 41 features par connexion
- 5 classes : Normal, DoS, Probe, R2L, U2R

Téléchargement : https://www.unb.ca/cic/datasets/nsl.html  
Placer les fichiers dans `data/raw/`

## Installation

```bash
git clone https://github.com/TON_USERNAME/network-ids---AI-Detection.git
cd network-ids---AI-Detection
pip3 install -r requirements.txt
```

## Usage

```bash
# Random Forest
python3 src/models/random_forest.py

# XGBoost
python3 src/models/xgboost_model.py

# XGBoost + SMOTE (meilleur résultat)
python3 src/models/xgboost_smote.py
```

## Résultats

| Modèle | Accuracy | F1 Score | R2L Recall | U2R Recall |
|--------|----------|----------|------------|------------|
| Random Forest | 75.16% | 0.7027 | 0.00 | 0.04 |
| XGBoost | 76.66% | 0.7263 | 0.05 | 0.17 |
| XGBoost + SMOTE | **77.43%** | **0.7457** | **0.14** | **0.30** |

## Analyse

- **DoS** → très bien détecté (recall 0.97) grâce au grand nombre d'exemples
- **Probe** → bon recall (0.79 avec SMOTE)
- **R2L / U2R** → classes rares, améliorées significativement avec SMOTE
- **SMOTE** équilibre les classes en générant des exemples synthétiques

## Structure

```
network-ids---AI-Detection/
├── data/
│   └── raw/               # KDDTrain+.txt, KDDTest+.txt (non versionnés)
├── src/
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── xgboost_smote.py
│   └── utils/
│       └── constants.py
├── reports/figures/       # Matrices de confusion
└── requirements.txt
```

## Auteur

Amine — Cybersecurity & AI Engineer  
ENSIMAG — Grenoble INP

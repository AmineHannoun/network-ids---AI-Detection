import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.constants import COLUMNS, CATEGORICAL_FEATURES, ATTACK_MAP, CLASS_NAMES

print("=" * 55)
print("   XGBOOST + SMOTE — Détection d'attaques réseau")
print("=" * 55)

print("\n[1/5] Chargement des données...")
df_train = pd.read_csv('data/raw/KDDTrain+.txt', header=None, names=COLUMNS)
df_test  = pd.read_csv('data/raw/KDDTest+.txt',  header=None, names=COLUMNS)
print(f"      Train : {len(df_train):,} lignes | Test : {len(df_test):,} lignes")

print("\n[2/5] Preprocessing...")
for df in [df_train, df_test]:
    df['attack_cat'] = df['label'].map(ATTACK_MAP).fillna('Unknown')

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    le.fit(pd.concat([df_train[col], df_test[col]]))
    df_train[col] = le.transform(df_train[col])
    df_test[col]  = le.transform(df_test[col])

feature_cols = [c for c in COLUMNS if c not in ('label', 'difficulty')]
X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values

le_target = LabelEncoder()
le_target.fit(df_train['attack_cat'].tolist() + df_test['attack_cat'].tolist())
y_train = le_target.transform(df_train['attack_cat'])
y_test  = le_target.transform(df_test['attack_cat'])

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("      Features encodées et normalisées ✅")

print("\n[3/5] Application SMOTE...")
print(f"      Avant SMOTE : {dict(zip(*np.unique(y_train, return_counts=True)))}")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"      Après SMOTE : {dict(zip(*np.unique(y_train, return_counts=True)))}")
print("      Classes équilibrées ✅")

print("\n[4/5] Entraînement XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("      Modèle entraîné ✅")

print("\n[5/5] Évaluation...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print("\n" + "=" * 55)
print("   RÉSULTATS XGBOOST + SMOTE")
print("=" * 55)
print(f"\n   Accuracy  : {acc*100:.2f}%")
print(f"   F1 Score  : {f1:.4f}")
print("\n" + "-" * 55)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

os.makedirs('reports/figures', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matrice de confusion — XGBoost + SMOTE')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('reports/figures/confusion_matrix_xgb_smote.png', dpi=150)
print("   Matrice sauvegardée → reports/figures/confusion_matrix_xgb_smote.png")
print("=" * 55)

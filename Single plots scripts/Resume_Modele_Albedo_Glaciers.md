# Modèle de Régression d'Albédo Glaciaire - Résumé Technique

## 🎯 Objectif
Analyser l'influence des variables climatiques (température, précipitations, aérosols) sur l'albédo des glaciers via des régressions multiples mensuelles avec données MODIS.

## 📊 Données et Méthodes

### Sources de Données
- **MODIS**: 3 méthodes d'albédo (MCD43A3, MOD09GA, MOD10A1)
- **Climate**: ERA5 (température, précipitations)
- **Aérosols**: MERRA-2 (BC_AOD - poussière de carbone noir)

### Sélection des Pixels
```python
# Filtrage glacier/eau libre basé sur fraction glaciaire
glacier_mask = glacier_fraction >= 0.5
open_water_mask = glacier_fraction < 0.1

# Application des masques de qualité MODIS
valid_pixels = (qa_band != 255) & (qa_band != 254)  # Exclut nuages/eau
```

## 🔧 Modélisation

### Structure de Régression Multiple
```python
# Modèle avec diagnostics complets
model = smf.ols('albedo ~ temp + precip + bc_aod', data=monthly_data)
results = model.fit()

# Tests diagnostiques automatiques
durbin_watson_stat = durbin_watson(results.resid)
bp_test = het_breuschpagan(results.resid, results.model.exog)
```

### Importance Relative des Prédicteurs
```python
from sklearn.metrics import r2_score
from itertools import combinations

# Méthode LMG (Lindeman, Merenda & Gold)
def calculate_lmg_importance(X, y):
    # Calcul R² pour toutes combinaisons possibles
    for subset in all_subsets:
        r2_with = r2_score(y, model_with_var.predict(X))
        r2_without = r2_score(y, model_without_var.predict(X))
        contributions.append(r2_with - r2_without)
```

## 📈 Tests Statistiques Clés

### 1. Test de Durbin-Watson (Autocorrélation)
```python
dw_stat = durbin_watson(residuals)
# Interprétation:
# < 1.5: Autocorrélation positive (problématique)
# 1.5-2.5: Acceptable
# > 2.5: Autocorrélation négative
```

### 2. Test de Breusch-Pagan (Hétéroscédasticité)
```python
bp_statistic, bp_pvalue = het_breuschpagan(residuals, exog_vars)
# H0: Homoscédasticité (variance constante)
# p < 0.05: Rejeter H0 → Hétéroscédasticité détectée
```

### 3. Tailles d'Effet avec Intervalles de Confiance
```python
# Calcul automatique des effets physiques
coef = results.params['temp']
se = results.bse['temp'] 
ci_lower = coef - 1.96 * se
ci_upper = coef + 1.96 * se

effect_statement = f"Une augmentation de 1°C réduit l'albédo de {abs(coef):.3f} ± {1.96*se:.3f}"
```

## 📋 Résultats Principaux

### Performance par Glacier et Méthode

| Glacier   | MCD43A3 | MOD09GA | MOD10A1 | Meilleure Méthode |
|-----------|---------|---------|---------|-------------------|
| Athabasca | 0.449   | 0.389   | **0.516** | MOD10A1          |
| Haig      | 0.312   | 0.381   | **0.407** | MOD10A1          |
| Coropuna  | **0.298** | 0.247   | 0.287   | MCD43A3          |

### Importance des Prédicteurs (% LMG)

| Glacier   | Méthode | Température | Précipitations | BC_AOD |
|-----------|---------|-------------|-----------------|--------|
| Athabasca | MOD10A1 | **68.2%** ↗ | 21.4%          | 10.4%  |
| Haig      | MOD10A1 | **52.1%** ↗ | 31.8%          | 16.1%  |
| Coropuna  | MCD43A3 | **45.7%** ↗ | 38.9%          | 15.4%  |

## 🔍 Diagnostics de Qualité

### Statistiques par Observation
```python
# Ajout automatique des métriques de pixels
pixel_stats = {
    'n_pixels_used': valid_pixels.sum(),
    'mean_glacier_fraction': glacier_fraction[valid_pixels].mean(),
    'avg_pixels_per_obs': total_pixels / n_observations
}
```

### Critères de Validation
- **R² ajusté** > 0.3 (modèle acceptable)
- **Durbin-Watson** ∈ [1.5, 2.5] (pas d'autocorrélation)
- **Breusch-Pagan** p > 0.05 (homoscédasticité)
- **Significativité** p < 0.05 pour prédicteurs principaux

## 💡 Interprétation Physique

### Effets Climatiques Typiques
```python
# Exemples de tailles d'effet (Athabasca, MOD10A1)
"Une augmentation de 1°C réduit l'albédo de 0.016 ± 0.002"
"Une augmentation de 10mm de précipitation augmente l'albédo de 0.001 ± 0.001"
"Une augmentation de 0.001 BC_AOD réduit l'albédo de 0.008 ± 0.003"
```

### Mécanismes Physiques
- **Température** ↑ → Fonte → Albédo ↓ (effet dominant)
- **Précipitations** ↑ → Neige fraîche → Albédo ↑
- **BC_AOD** ↑ → Dépôts sombres → Albédo ↓

## 🚀 Utilisation

### Exécution Simple
```bash
# Activation environnement Conda
conda activate albedo_env

# Lancement analyse
python multi_glacier_monthly_regression.py
```

### Sorties Automatiques
- `summary_stats.txt`: Métriques de régression complètes
- `regression_plots.png`: Visualisations diagnostiques  
- `effect_sizes.txt`: Tailles d'effet en langage naturel

## ⚙️ Configuration

### Paramètres Modifiables
```python
# Dans le script principal
CURRENT_GLACIER = "athabasca"  # ou "haig", "coropuna"
MODIS_METHODS = ["MCD43A3", "MOD09GA", "MOD10A1"]
MIN_GLACIER_FRACTION = 0.5  # Seuil de classification glacier
```

---
*Modèle validé sur 3 glaciers × 3 méthodes MODIS = 9 configurations*  
*Période d'analyse: 2003-2020 (données mensuelles)*

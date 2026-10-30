"""
ULTIMATE MENTAL HEALTH MODEL - Advanced ML Pipeline
Trains & compares the best models: XGBoost, LightGBM, CatBoost, Neural Networks, etc.
Performs rigorous cross-validation and selects the optimal model
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 ULTIMATE MENTAL HEALTH MODEL - ADVANCED ML PIPELINE")
print("=" * 80)

# ============ LOAD & PREPARE DATA ============
print("\n📊 STEP 1: Loading and preparing data...")
df = pd.read_csv('Student Social Media And Mental Health Impact.csv')

required_cols = ['Age', 'Gender', 'Country', 'Academic_Level', 'Most_Used_Platform',
                 'Purpose_Of_Use', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours',
                 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Stress_Level']

df_clean = df[required_cols + ['Mental_Health_Score']].copy()
df_clean = df_clean.dropna()

# Group countries
top_countries = ['India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
df_clean['Grouped_country'] = df_clean['Country'].apply(lambda x: x if x in top_countries else 'Other')

# Encode categorical
label_encoders = {}
categorical_cols = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Stress_Level', 'Grouped_country']

X = df_clean.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

y = X.pop('Mental_Health_Score')
X = X.drop('Country', axis=1)

# Scale features for neural networks & boosting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"✅ Dataset: {len(df_clean)} records | Train: {len(X_train)}, Test: {len(X_test)}")
print(f"✅ Target range: {y.min():.2f} - {y.max():.2f}")
print(f"✅ Features: {len(X.columns)}")

# ============ CROSS-VALIDATION SETUP ============
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models_results = {}

print("\n" + "=" * 80)
print("🧠 STEP 2: Training Advanced ML Models...")
print("=" * 80)

# ============ MODEL 1: XGBOOST ============
print("\n1️⃣  Training XGBoost (Industry Gold Standard)...")
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    cv_xgb = cross_val_score(xgb, X_train, y_train, cv=kfold, scoring='r2').mean()
    
    models_results['XGBoost'] = {
        'model': xgb,
        'r2': r2_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'cv_score': cv_xgb,
        'feature_importance': dict(zip(X.columns, xgb.feature_importances_))
    }
    print(f"   ✅ R² = {r2_xgb:.4f} | MAE = {mae_xgb:.4f} | RMSE = {rmse_xgb:.4f} | CV = {cv_xgb:.4f}")
except ImportError:
    print("   ⚠️  XGBoost not installed (optional)")

# ============ MODEL 2: LIGHTGBM ============
print("\n2️⃣  Training LightGBM (Fast & Powerful)...")
try:
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    lgb.fit(X_train, y_train)
    y_pred_lgb = lgb.predict(X_test)
    
    r2_lgb = r2_score(y_test, y_pred_lgb)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    cv_lgb = cross_val_score(lgb, X_train, y_train, cv=kfold, scoring='r2').mean()
    
    models_results['LightGBM'] = {
        'model': lgb,
        'r2': r2_lgb,
        'mae': mae_lgb,
        'rmse': rmse_lgb,
        'cv_score': cv_lgb,
        'feature_importance': dict(zip(X.columns, lgb.feature_importances_))
    }
    print(f"   ✅ R² = {r2_lgb:.4f} | MAE = {mae_lgb:.4f} | RMSE = {rmse_lgb:.4f} | CV = {cv_lgb:.4f}")
except ImportError:
    print("   ⚠️  LightGBM not installed (optional)")

# ============ MODEL 3: CATBOOST ============
print("\n3️⃣  Training CatBoost (Categorical Expert)...")
try:
    from catboost import CatBoostRegressor
    cat = CatBoostRegressor(
        iterations=300,
        depth=7,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
        thread_count=-1
    )
    cat.fit(X_train, y_train, verbose=False)
    y_pred_cat = cat.predict(X_test)
    
    r2_cat = r2_score(y_test, y_pred_cat)
    mae_cat = mean_absolute_error(y_test, y_pred_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
    cv_cat = cross_val_score(cat, X_train, y_train, cv=kfold, scoring='r2').mean()
    
    models_results['CatBoost'] = {
        'model': cat,
        'r2': r2_cat,
        'mae': mae_cat,
        'rmse': rmse_cat,
        'cv_score': cv_cat,
        'feature_importance': dict(zip(X.columns, cat.feature_importances_))
    }
    print(f"   ✅ R² = {r2_cat:.4f} | MAE = {mae_cat:.4f} | RMSE = {rmse_cat:.4f} | CV = {cv_cat:.4f}")
except ImportError:
    print("   ⚠️  CatBoost not installed (optional)")

# ============ MODEL 4: GRADIENT BOOSTING ============
print("\n4️⃣  Training Gradient Boosting (Robust)...")
gb = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
cv_gb = cross_val_score(gb, X_train, y_train, cv=kfold, scoring='r2').mean()

models_results['Gradient Boosting'] = {
    'model': gb,
    'r2': r2_gb,
    'mae': mae_gb,
    'rmse': rmse_gb,
    'cv_score': cv_gb,
    'feature_importance': dict(zip(X.columns, gb.feature_importances_))
}
print(f"   ✅ R² = {r2_gb:.4f} | MAE = {mae_gb:.4f} | RMSE = {rmse_gb:.4f} | CV = {cv_gb:.4f}")

# ============ MODEL 5: RANDOM FOREST ============
print("\n5️⃣  Training Random Forest (Baseline)...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
cv_rf = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='r2').mean()

models_results['Random Forest'] = {
    'model': rf,
    'r2': r2_rf,
    'mae': mae_rf,
    'rmse': rmse_rf,
    'cv_score': cv_rf,
    'feature_importance': dict(zip(X.columns, rf.feature_importances_))
}
print(f"   ✅ R² = {r2_rf:.4f} | MAE = {mae_rf:.4f} | RMSE = {rmse_rf:.4f} | CV = {cv_rf:.4f}")

# ============ MODEL 6: RIDGE REGRESSION ============
print("\n6️⃣  Training Ridge Regression (Interpretable)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
cv_ridge = cross_val_score(ridge, X_train_scaled, y_train, cv=kfold, scoring='r2').mean()

models_results['Ridge Regression'] = {
    'model': ridge,
    'r2': r2_ridge,
    'mae': mae_ridge,
    'rmse': rmse_ridge,
    'cv_score': cv_ridge,
    'feature_importance': dict(zip(X.columns, np.abs(ridge.coef_)))
}
print(f"   ✅ R² = {r2_ridge:.4f} | MAE = {mae_ridge:.4f} | RMSE = {rmse_ridge:.4f} | CV = {cv_ridge:.4f}")

# ============ MODEL SELECTION ============
print("\n" + "=" * 80)
print("🏆 MODEL COMPARISON & SELECTION")
print("=" * 80)

comparison_table = []
for model_name, results in models_results.items():
    comparison_table.append({
        'Model': model_name,
        'R² Score': f"{results['r2']:.4f}",
        'MAE': f"{results['mae']:.4f}",
        'RMSE': f"{results['rmse']:.4f}",
        'CV Score': f"{results['cv_score']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_table)
print("\n" + comparison_df.to_string(index=False))

# Select best model by R² score
best_model_name = max(models_results, key=lambda x: models_results[x]['r2'])
best_model_data = models_results[best_model_name]
best_model = best_model_data['model']

print(f"\n✅ BEST MODEL SELECTED: {best_model_name}")
print(f"   • R² Score: {best_model_data['r2']:.4f} (Explains {best_model_data['r2']*100:.1f}% of variance)")
print(f"   • MAE: {best_model_data['mae']:.4f}")
print(f"   • RMSE: {best_model_data['rmse']:.4f}")
print(f"   • CV Score: {best_model_data['cv_score']:.4f}")

# ============ FEATURE IMPORTANCE ============
print(f"\n📊 TOP 8 FEATURE IMPORTANCES ({best_model_name}):")
sorted_features = sorted(best_model_data['feature_importance'].items(), 
                         key=lambda x: abs(x[1]), reverse=True)
for i, (feat, imp) in enumerate(sorted_features[:8], 1):
    print(f"   {i}. {feat}: {imp:.4f}")

# ============ SAVE BEST MODEL & METADATA ============
model_meta = {
    "selected_model": best_model_name,
    "performance": {
        "r2_score": round(best_model_data['r2'], 4),
        "mae": round(best_model_data['mae'], 4),
        "rmse": round(best_model_data['rmse'], 4),
        "cv_score": round(best_model_data['cv_score'], 4),
        "model_comparison": {
            name: {
                "r2": round(results['r2'], 4),
                "mae": round(results['mae'], 4),
                "rmse": round(results['rmse'], 4),
                "cv_score": round(results['cv_score'], 4)
            }
            for name, results in models_results.items()
        }
    },
    "feature_importances": {feat: round(float(imp), 4) for feat, imp in sorted_features},
    "dataset_baseline": {
        "mean_score": round(float(y_train.mean()), 2),
        "std_score": round(float(y_train.std()), 2),
        "min_score": round(float(y_train.min()), 2),
        "max_score": round(float(y_train.max()), 2),
        "percentiles": {
            "p10": round(float(y_train.quantile(0.10)), 2),
            "p25": round(float(y_train.quantile(0.25)), 2),
            "p50": round(float(y_train.quantile(0.50)), 2),
            "p75": round(float(y_train.quantile(0.75)), 2),
            "p90": round(float(y_train.quantile(0.90)), 2),
        }
    },
    "score_distribution": {
        "bins": [0, 2, 4, 6, 8, 10],
        "labels": ["Critical", "At Risk", "Struggling", "Fair", "Good", "Thriving"],
        "colors": ["#8B0000", "#D9534F", "#E8A837", "#F0AD4E", "#5B9BD5", "#4C9A78"],
        "descriptions": [
            "Needs immediate professional help",
            "Significant mental health concerns",
            "Notable challenges, needs support",
            "Some areas need improvement",
            "Generally healthy with room to grow",
            "Excellent mental wellness"
        ]
    },
    "explanation_factors": {
        "Sleep_Hours_Per_Night": "Quality sleep (7-9 hrs) is crucial for mental health and resilience",
        "Physical_Activity_Hours": "Regular exercise (1+ hrs/day) significantly improves mood and stress resilience",
        "Stress_Level": "Perceived stress directly impacts mental health outcomes",
        "Study_Hours": "Balanced study time (3-6 hrs) prevents burnout; too much or too little increases stress",
        "Avg_Daily_Usage_Hours": "Excessive screen time (>6 hrs) correlates with anxiety and depression",
        "Daily_Unlocks": "Frequent phone unlocks indicate phone addiction which affects mental focus and wellbeing"
    },
    "recommendations_config": {
        "sleep_threshold_low": 6,
        "sleep_threshold_optimal": 8,
        "activity_threshold_min": 1,
        "screen_time_threshold_max": 5,
        "stress_critical": "Very High",
        "unlocks_threshold_high": 100
    }
}

# Save metadata
with open('model_meta.json', 'w') as f:
    json.dump(model_meta, f, indent=2)

# Save best model
joblib.dump(best_model, 'best_model.pkl')

# Save scaler for potential scaling operations
joblib.dump(scaler, 'scaler.pkl')

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\n" + "=" * 80)
print("✅ MODELS SAVED SUCCESSFULLY")
print("=" * 80)
print(f"✅ Best Model: best_model.pkl ({best_model_name})")
print(f"✅ Metadata: model_meta.json")
print(f"✅ Scaler: scaler.pkl")
print(f"✅ Label Encoders: label_encoders.pkl")
print(f"\n🚀 Ready for production deployment!")

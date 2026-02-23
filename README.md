# 🎓 Admission Prediction - Regression Analysis

A **machine learning regression project** predicting university graduate program admission chances based on GRE scores, TOEFL scores, GPA, and other factors.

## 🎯 Overview

This project covers:
- ✅ Regression model building
- ✅ Feature analysis & correlation
- ✅ Linear & non-linear regression
- ✅ Model comparison
- ✅ Prediction confidence intervals
- ✅ Residual analysis

## 📊 Dataset Features

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AdmissionDataAnalysis:
    """Analyze admission dataset"""
    
    def __init__(self, filepath='admission_predict.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_data(self):
        """Exploratory analysis"""
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        print(f"\nBasic stats:\n{self.df.describe()}")
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        corr = self.df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        # Correlation with target
        target_corr = corr['Chance of Admit '].sort_values(ascending=False)
        print("\nCorrelation with Admission Chance:")
        print(target_corr)
        
        return corr
    
    def visualize_distributions(self):
        """Plot feature distributions"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.df.columns[1:]):  # Skip index
            axes[idx].hist(self.df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
```

## 🏗️ Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class AdmissionRegressionModels:
    """Multiple regression models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale')
        }
    
    def explain_models(self):
        """Explain each model"""
        explanations = {
            'Linear Regression': 'y = m1*x1 + m2*x2 + ... + b (assumes linear relationship)',
            'Ridge': 'Linear with L2 regularization (prevents overfitting)',
            'Lasso': 'Linear with L1 regularization (feature selection via coefficients=0)',
            'ElasticNet': 'Combination of Ridge and Lasso',
            'Random Forest': 'Ensemble of decision trees (handles non-linearity)',
            'Gradient Boosting': 'Sequential boosting (reduces bias/variance)',
            'SVR': 'Support Vector Regression (RBF kernel for non-linear patterns)'
        }
        
        for name, desc in explanations.items():
            print(f"\n{name}:\n  {desc}")
```

## 📈 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

class AdmissionFeatureSelection:
    """Select important features"""
    
    @staticmethod
    def correlation_based(df, target_col, n_features=None):
        """Select by correlation"""
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        if n_features:
            selected = correlations[1:n_features+1].index.tolist()
        else:
            selected = correlations[correlations > 0.3].index.tolist()
        
        return selected
    
    @staticmethod
    def statistical_test(X, y, n_features=5):
        """SelectKBest with f_regression"""
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print("Feature Importance Scores:")
        print(feature_scores)
        
        return selector
```

## 📊 Model Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class AdmissionEvaluator:
    """Evaluate regression models"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
    
    @staticmethod
    def compare_models(y_true, predictions_dict):
        """Compare all models"""
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = AdmissionEvaluator.calculate_metrics(y_true, y_pred)
        
        results_df = pd.DataFrame(results).T
        print("\nModel Comparison:")
        print(results_df)
        
        return results_df
    
    @staticmethod
    def plot_predictions(y_true, y_pred, model_name='Model'):
        """Plot actual vs predicted"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].grid()
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid()
        
        plt.tight_layout()
        plt.show()
```

## 🎯 Prediction Confidence Intervals

```python
from scipy import stats

class PredictionConfidence:
    """Calculate prediction intervals"""
    
    @staticmethod
    def confidence_interval(y_pred, residuals, confidence=0.95):
        """Calculate prediction interval"""
        std_error = np.std(residuals)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * std_error
        
        lower = y_pred - margin
        upper = y_pred + margin
        
        return lower, upper
    
    @staticmethod
    def predict_with_interval(model, X_new, residuals, confidence=0.95):
        """Predict with confidence interval"""
        y_pred = model.predict(X_new)
        lower, upper = PredictionConfidence.confidence_interval(y_pred, residuals, confidence)
        
        results = pd.DataFrame({
            'Prediction': y_pred,
            'Lower Bound': lower,
            'Upper Bound': upper,
            'Interval Width': upper - lower
        })
        
        return results
```

## 💡 Interview Talking Points

**Q: Why multiple models?**
```
Answer:
- Different models capture different patterns
- Ensemble combining strengths
- Some handle non-linearity better
- Model selection requires comparison
```

**Q: How interpret coefficients?**
```
Answer:
- Linear model: 1 unit increase → coefficient change in target
- Ridge/Lasso: Trade-off between bias and variance
- Tree models: Feature importance via splits
```

## 🌟 Portfolio Value

✅ Regression fundamentals
✅ Feature analysis & selection
✅ Multiple regression models
✅ Hyperparameter tuning
✅ Prediction intervals
✅ Model comparison
✅ Residual analysis

---

**Technologies**: Scikit-learn, Pandas, NumPy, Matplotlib, SciPy


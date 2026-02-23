# 📊 Admission Dataset Prediction - Classification

A **machine learning system** for predicting university admission outcomes using student metrics and ensemble models.

## 🎯 Overview

This project provides:
- ✅ Admission data analysis
- ✅ Student metric engineering
- ✅ Classification models
- ✅ Feature selection
- ✅ Probability calibration
- ✅ Early prediction
- ✅ University insights

## 📖 Data Analysis

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class AdmissionDataAnalyzer:
    """Process admission data"""
    
    def __init__(self):
        self.data = None
    
    def load_data(self, filepath):
        """Load admission dataset"""
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def analyze_acceptance_rate(self):
        """Calculate admission statistics"""
        df = self.data.copy()
        
        total_applicants = len(df)
        admitted = (df['admitted'] == 1).sum()
        acceptance_rate = (admitted / total_applicants) * 100
        
        stats = {
            'total_applicants': total_applicants,
            'admitted': admitted,
            'rejected': total_applicants - admitted,
            'acceptance_rate': acceptance_rate
        }
        
        print(f"Acceptance Rate: {acceptance_rate:.2f}%")
        
        return stats
    
    def analyze_metrics_by_group(self):
        """Compare admitted vs rejected"""
        df = self.data.copy()
        
        metrics_cols = ['gre_score', 'gmat_score', 'gpa', 'work_experience_years']
        
        admitted_stats = df[df['admitted'] == 1][metrics_cols].describe()
        rejected_stats = df[df['admitted'] == 0][metrics_cols].describe()
        
        comparison = pd.DataFrame({
            'Admitted_Mean': admitted_stats.loc['mean'],
            'Rejected_Mean': rejected_stats.loc['mean'],
            'Difference': admitted_stats.loc['mean'] - rejected_stats.loc['mean']
        })
        
        print(comparison)
        
        return comparison
    
    def feature_engineering(self):
        """Create derived features"""
        df = self.data.copy()
        
        # Composite score
        df['composite_score'] = (df['gre_score'] * 0.3 + 
                                 df['gmat_score'] * 0.2 + 
                                 df['gpa'] * 50)  # Normalize GPA
        
        # Test performance
        df['test_strong'] = ((df['gre_score'] > 320) | (df['gmat_score'] > 650)).astype(int)
        
        # Academic strength
        df['gpa_excellence'] = (df['gpa'] >= 3.7).astype(int)
        
        # Experience proxy
        df['experienced'] = (df['work_experience_years'] >= 3).astype(int)
        
        # University tier
        if 'university_rank' in df.columns:
            df['top_university'] = (df['university_rank'] <= 100).astype(int)
        
        # Recommendation quality (assume 1-4 scale)
        if 'recommendation_score' in df.columns:
            df['strong_recommendation'] = (df['recommendation_score'] >= 3).astype(int)
        
        # Statement strength (binary already)
        if 'statement_score' in df.columns:
            df['strong_statement'] = (df['statement_score'] >= 3).astype(int)
        
        # Overall profile strength
        strength_features = [
            'test_strong', 'gpa_excellence', 'experienced', 
            'strong_recommendation', 'strong_statement'
        ]
        
        if all(col in df.columns for col in strength_features):
            df['profile_strength'] = df[strength_features].sum(axis=1)
        
        return df
```

## 🤖 Classification Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class AdmissionClassifier:
    """Predict admission outcomes"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
    
    def logistic_regression(self, X_train, y_train):
        """Baseline model"""
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        self.models['lr'] = lr
        
        return lr
    
    def random_forest(self, X_train, y_train):
        """Random Forest classifier"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        self.models['rf'] = rf
        
        return rf
    
    def gradient_boosting(self, X_train, y_train):
        """Gradient Boosting"""
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        self.models['gb'] = gb
        
        return gb
    
    def svm_classifier(self, X_train, y_train):
        """SVM with probability"""
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm.fit(X_train, y_train)
        self.models['svm'] = svm
        
        return svm
    
    def ensemble_voting(self, X_train, y_train):
        """Ensemble voting"""
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', self.models['lr']),
                ('rf', self.models['rf']),
                ('gb', self.models['gb']),
                ('svm', self.models['svm'])
            ],
            voting='soft'
        )
        
        voting_clf.fit(X_train, y_train)
        self.models['voting'] = voting_clf
        
        return voting_clf
    
    def predict_admission(self, applicant_features):
        """Predict for single applicant"""
        if self.best_model is None:
            raise ValueError("Model not trained")
        
        prediction = self.best_model.predict([applicant_features])[0]
        probability = self.best_model.predict_proba([applicant_features])[0]
        
        return {
            'admitted': prediction,
            'admission_probability': probability[1],
            'rejection_probability': probability[0]
        }
    
    def batch_predict(self, X_test):
        """Predict for multiple applicants"""
        probabilities = self.best_model.predict_proba(X_test)
        
        results = pd.DataFrame({
            'admission_probability': probabilities[:, 1],
            'predicted_class': self.best_model.predict(X_test)
        })
        
        return results
```

## 📈 Model Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

class AdmissionEvaluator:
    """Evaluate admission model"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Comprehensive evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nROC-AUC Score: {roc_auc:.3f}")
        
        return {
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    @staticmethod
    def admission_threshold_analysis(y_true, y_pred_proba):
        """Analyze different admission thresholds"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
            admitted = y_pred.sum()
            actual_success_rate = y_true[y_pred == 1].mean()
            
            results.append({
                'threshold': threshold,
                'applicants_admitted': admitted,
                'success_rate': actual_success_rate
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Admission Prediction ROC Curve')
        plt.legend()
        plt.show()
```

## 🎓 Admission Insights

```python
class AdmissionInsights:
    """Generate admission insights"""
    
    @staticmethod
    def identify_factors(model, feature_names):
        """Find key admission factors"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("Top Admission Factors:")
            for i in range(min(5, len(feature_names))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
            return importances
    
    @staticmethod
    def admission_probability_stats(y_pred_proba):
        """Analyze probability distribution"""
        probabilities = y_pred_proba[:, 1]
        
        stats = {
            'mean_probability': probabilities.mean(),
            'median_probability': np.median(probabilities),
            'high_confidence': (probabilities > 0.8).sum(),
            'borderline': ((probabilities >= 0.4) & (probabilities <= 0.6)).sum(),
            'low_confidence': (probabilities < 0.2).sum()
        }
        
        return stats
    
    @staticmethod
    def early_prediction_potential(model, X_partial):
        """Predict with partial information"""
        # Predict using only critical features
        probabilities = model.predict_proba(X_partial)
        
        return probabilities[:, 1]
```

## 💡 Interview Talking Points

**Q: Class imbalance in admissions?**
```
Answer:
- Acceptance rare (5-30%)
- Use ROC-AUC not accuracy
- Weighted classification
- Threshold tuning for balance
- Cost-sensitive learning
```

**Q: Model interpretability critical?**
```
Answer:
- Fairness and transparency needed
- SHAP for feature importance
- Threshold analysis necessary
- Bias detection (gender, race)
- Legal/ethical compliance
```

## 🌟 Portfolio Value

✅ Classification modeling
✅ Feature engineering
✅ Ensemble methods
✅ ROC-AUC evaluation
✅ Threshold analysis
✅ Educational domain
✅ Interpretability focus

---

**Technologies**: Scikit-learn, Pandas, XGBoost


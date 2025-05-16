import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import joblib
import textstat
import re

os.makedirs("plt", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plt/combined-features", exist_ok=True)
os.makedirs("plt/misclassified_examples", exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    if " -- " in text:
        return text.split(" -- ", 1)[1]
    return text

def load_dataset(true_path, fake_path, n_samples=None):
    with tqdm(total=4, desc="Loading dataset") as pbar:
        # Load data with NA removal
        true = pd.read_csv(true_path).dropna(subset=['text' if 'text' in pd.read_csv(true_path, nrows=1).columns else 'article'])
        fake = pd.read_csv(fake_path).dropna(subset=['text' if 'text' in pd.read_csv(fake_path, nrows=1).columns else 'article'])
        pbar.update(2)
        
        if n_samples is not None:
            true = true.sample(n=min(n_samples, len(true)), random_state=42)
            fake = fake.sample(n=min(n_samples, len(fake)), random_state=42)
            
        true['label'] = 1
        fake['label'] = 0
        pbar.update(1)
        
        text_col = 'text' if 'text' in true.columns else 'article'
        true['text'] = true[text_col].astype(str).apply(clean_text)
        fake['text'] = fake[text_col].astype(str).apply(clean_text)
        
        true = true[true['text'].str.strip().astype(bool)]
        fake = fake[fake['text'].str.strip().astype(bool)]
        pbar.update(1)
        
        df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)
    return df['text'], df['label']

class SBERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', progress_bar=True):
        self.model_name = model_name
        self.progress_bar = progress_bar
        self.model = SentenceTransformer(model_name)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(
            list(X),
            show_progress_bar=self.progress_bar,
            convert_to_numpy=True
        )
    
    def get_params(self, deep=True):
        return {"model_name": self.model_name, "progress_bar": self.progress_bar}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class StatisticalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = [
            'char_count', 'word_count', 'avg_word_length',
            'sentence_count', 'flesch_reading_ease',
            'smog_index', 'lexicon_count', 'syllable_count',
            'punctuation_count', 'uppercase_count'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = char_count / word_count if word_count > 0 else 0
            sentence_count = len(re.split(r'[.!?]+', text))
            
            flesch = textstat.flesch_reading_ease(text)
            smog = textstat.smog_index(text)
            lexicon = textstat.lexicon_count(text)
            syllable = textstat.syllable_count(text)
            punctuation = sum(1 for char in text if char in '.,;:!?')
            uppercase = sum(1 for char in text if char.isupper())
            
            features.append([
                char_count, word_count, avg_word_length,
                sentence_count, flesch, smog,
                lexicon, syllable, punctuation, uppercase
            ])
            
        return np.array(features)
    
    def get_feature_names(self):
        return self.feature_names

def create_combined_pipeline():
    """Create pipeline without LSA"""
    semantic_pipeline = Pipeline([
        ('sbert', SBERTTransformer()),
        ('scaler', RobustScaler())  # Removed LSA step
    ])
    
    statistical_pipeline = Pipeline([
        ('stats', StatisticalFeatures()),
        ('scaler', RobustScaler())
    ])
    
    combined_features = FeatureUnion([
        ('semantic', semantic_pipeline),
        ('statistical', statistical_pipeline)
    ])
    
    full_pipeline = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', verbose=1))
    ])
    
    return full_pipeline

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['Fake', 'True'],
                yticklabels=['Fake', 'True'])
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plt/combined-features/confusion_matrix_{dataset_name}.png")
    plt.close()

def plot_vector_distributions(model, X, y, dataset_name):
    try:
        features = model.named_steps['features'].transform(X)
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(features)
        
        plot_df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'label': y.map({0: 'Fake', 1: 'True'})
        })
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=plot_df,
            x='x',
            y='y',
            hue='label',
            palette={'Fake': 'red', 'True': 'blue'},
            alpha=0.6
        )
        plt.title(f"Vector Distribution - {dataset_name}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title='Class')
        plt.savefig(f"plt/combined-features/vector_distribution_{dataset_name}.png")
        plt.close()
        
    except Exception as e:
        print(f"Could not plot vector distribution for {dataset_name}: {str(e)}")

def plot_feature_importance(model, feature_names, dataset_name):
    if hasattr(model.named_steps['clf'], 'coef_'):
        plt.figure(figsize=(14, 6))
        coef = model.named_steps['clf'].coef_[0]
        
        sbert_coef = coef[:384]  # SBERT MiniLM-L6 has 384 dimensions
        
        stats_coef = coef[384:394]  # 10 statistical features
        
        plt.subplot(1, 2, 1)
        n_bars = min(50, len(sbert_coef))
        x_pos = np.arange(n_bars)
        plt.bar(x_pos, sbert_coef[:n_bars])
        plt.title(f"Top SBERT Feature Importance\n{dataset_name}")
        plt.xlabel("SBERT Dimension")
        plt.ylabel("Coefficient Value")
        plt.xticks(x_pos, x_pos, rotation=45)
        
        plt.subplot(1, 2, 2)
        stats_feature_names = model.named_steps['features'].transformer_list[1][1].named_steps['stats'].get_feature_names()
        x_pos = np.arange(len(stats_coef))
        plt.bar(x_pos, stats_coef)
        plt.xticks(x_pos, stats_feature_names, rotation=45)
        plt.title(f"Statistical Feature Importance\n{dataset_name}")
        plt.xlabel("Feature")
        plt.ylabel("Coefficient Value")
        
        plt.tight_layout()
        plt.savefig(f"plt/combined-features/feature_importance_{dataset_name}.png")
        plt.close()

def main():
    print("Loading datasets...")
    datasets = {
        "train": {"paths": ("true.csv", "fake.csv"), "n_samples": 12000},
        "test1": {"paths": ("true1.csv", "fake1.csv"), "n_samples": 4000},
        "test2": {"paths": ("true2.csv", "fake2.csv"), "n_samples": 4000}
    }
    
    loaded_data = {}
    for name, config in tqdm(datasets.items(), desc="Loading datasets"):
        loaded_data[name] = load_dataset(*config["paths"], n_samples=config["n_samples"])
        
    X_train, X_val, y_train, y_val = train_test_split(
        loaded_data["train"][0], loaded_data["train"][1], 
        test_size=0.2, 
        random_state=69
    )
    
    print("Training model:")
    pipeline = create_combined_pipeline()
    
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l2']
    }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=1,
        verbose=10
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    
    joblib.dump(best_model, "models/combined_sbert_stats_model.pkl")
    
    feature_names = (
        [f"sbert_{i}" for i in range(384)] +  # SBERT embeddings
        best_model.named_steps['features']
          .transformer_list[1][1]  # Statistical features
          .named_steps['stats'].get_feature_names()
    )
    
    results = []
    for name, (X, y) in loaded_data.items():
        if name == "train":
            y_pred = best_model.predict(X_train)
            y_proba = best_model.predict_proba(X_train)
            results.append({
                'Dataset': 'Training',
                'Accuracy': accuracy_score(y_train, y_pred),
                'F1 Score': f1_score(y_train, y_pred),
                'Log Loss': log_loss(y_train, y_proba)
            })
            plot_confusion_matrix(y_train, y_pred, "Training")
            
            val_pred = best_model.predict(X_val)
            val_proba = best_model.predict_proba(X_val)
            results.append({
                'Dataset': 'Validation',
                'Accuracy': accuracy_score(y_val, val_pred),
                'F1 Score': f1_score(y_val, val_pred),
                'Log Loss': log_loss(y_val, val_proba)
            })
            plot_confusion_matrix(y_val, val_pred, "Validation")
        else:
            y_pred = best_model.predict(X)
            y_proba = best_model.predict_proba(X)
            results.append({
                'Dataset': name,
                'Accuracy': accuracy_score(y, y_pred),
                'F1 Score': f1_score(y, y_pred),
                'Log Loss': log_loss(y, y_proba)
            })
            plot_confusion_matrix(y, y_pred, name)
    
    pd.DataFrame(results).to_csv("plt/combined_model_results.csv", index=False)
    print("\nResults saved.")
    
    print("\nGenerating visualizations:")
    plot_feature_importance(best_model, feature_names, "Final")
    plot_vector_distributions(best_model, X_train, y_train, "Training")
    

if __name__ == "__main__":
    main()

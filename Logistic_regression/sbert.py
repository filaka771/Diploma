import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import joblib

os.makedirs("plt/sbert", exist_ok=True)
os.makedirs("plt/sbert/vector_distributions", exist_ok=True)
os.makedirs("plt/sbert/confusion_matrices", exist_ok=True)
os.makedirs("plt/sbert/feature_importance", exist_ok=True)
os.makedirs("plt/sbert/misclassified_examples", exist_ok=True)
os.makedirs("models", exist_ok=True)

def clean_text(text):
    if " -- " in text:
        return text.split(" -- ", 1)[1]
    return text

def load_dataset(true_path, fake_path, n_samples=None):
    with tqdm(total=4, desc="Loading dataset") as pbar:
        true = pd.read_csv(true_path).dropna()
        fake = pd.read_csv(fake_path).dropna()
        pbar.update(2)
        
        if n_samples is not None:
            true = true.sample(n=min(n_samples, len(true)), random_state=42)
            fake = fake.sample(n=min(n_samples, len(fake)), random_state=42)
            
        true['label'] = 1
        fake['label'] = 0
        pbar.update(1)
        
        true['text'] = true['text'].astype(str).apply(clean_text) if 'text' in true.columns else true['article'].astype(str).apply(clean_text)
        fake['text'] = fake['text'].astype(str) if 'text' in fake.columns else fake['article'].astype(str)
        
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

def plot_vector_distribution(vectors, labels, dataset_name, stage="raw"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    vectors_2d = tsne.fit_transform(vectors)
    
    plot_df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'label': labels.map({0: 'Fake', 1: 'True'})
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        palette={'Fake': 'red', 'True': 'blue'},
        data=plot_df,
        alpha=0.6,
        s=50
    )
    
    plt.title(f"SBERT Vector Distribution - {dataset_name} ({stage})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Class')
    plt.grid(True, alpha=0.3)
    
    filename = f"plt/sbert/vector_distributions/{dataset_name}_{stage}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved vector distribution plot: {filename}")

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=['Fake', 'True'], 
                yticklabels=['Fake', 'True'])
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plt/sbert/confusion_matrices/{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved confusion matrix: plt/sbert/confusion_matrices/{dataset_name}.png")

def plot_feature_importance(model, dataset_name):
    if hasattr(model.named_steps['clf'], 'coef_'):
        coef = model.named_steps['clf'].coef_[0]
        features = range(len(coef))
        
        plt.figure(figsize=(12, 6))
        plt.bar(features, coef)
        plt.title(f"Feature Importance - {dataset_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.grid(True, alpha=0.3)
        
        filename = f"plt/sbert/feature_importance/{dataset_name}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved feature importance plot: {filename}")

def save_misclassified_examples(X, y_true, y_pred, y_proba, dataset_name):
    results_df = pd.DataFrame({
        'text': X,
        'true_label': y_true,
        'pred_label': y_pred,
        'prob_true': y_proba[:, 1],
        'correct': y_true == y_pred
    })
    
    results_df['true_label'] = results_df['true_label'].map({0: 'Fake', 1: 'True'})
    results_df['pred_label'] = results_df['pred_label'].map({0: 'Fake', 1: 'True'})
    
    error_samples = results_df[~results_df['correct']].sample(min(2, sum(~results_df['correct'])))
    correct_samples = results_df[results_df['correct']].sample(min(2, sum(results_df['correct'])))
    
    samples_df = pd.concat([correct_samples, error_samples])
    samples_df.to_csv(f"plt/sbert/misclassified_examples/{dataset_name}.csv", index=False)
    print(f"Saved misclassified examples: plt/sbert/misclassified_examples/{dataset_name}.csv")

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('sbert', SBERTTransformer()),
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', verbose=1))
    ])
    
    param_grid = {
        'sbert__model_name': ['all-MiniLM-L6-v2'],
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
    
    print("Grid search with SBERT embegings:")
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, dataset_name):
    print(f"\nEvaluating on {dataset_name}...")
    
    sbert_vectors = model.named_steps['sbert'].transform(X)
    scaled_vectors = model.named_steps['scaler'].transform(sbert_vectors)
    
    plot_vector_distribution(sbert_vectors, y, dataset_name, "raw")
    plot_vector_distribution(scaled_vectors, y, dataset_name, "scaled")
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    loss = log_loss(y, y_proba)
    
    plot_confusion_matrix(y, y_pred, dataset_name)
    
    plot_feature_importance(model, dataset_name)
    
    save_misclassified_examples(X, y, y_pred, y_proba, dataset_name)
    
    return acc, f1, loss

def main():
    print("Loading datasets...")
    datasets = {
        "train": {"paths": ("true.csv", "fake.csv"), "n_samples": 12000},
        "test1": {"paths": ("true1.csv", "fake1.csv"), "n_samples": 4000},
        "test2": {"paths": ("true2.csv", "fake2.csv"), "n_samples": 4000}
    }
    
    loaded_data = {}
    for name, config in tqdm(datasets.items(), desc="Loading all datasets"):
        loaded_data[name] = load_dataset(*config["paths"], n_samples=config["n_samples"])
        
    X_train, X_val, y_train, y_val = train_test_split(
        loaded_data["train"][0], loaded_data["train"][1], 
        test_size=0.2, 
        random_state=69
    )
    
    best_model = train_model(X_train, y_train)
    joblib.dump(best_model, "models/sbert_model.pkl")
    
    results = []
    for name, (X, y) in loaded_data.items():
        if name == "train":
            acc, f1, loss = evaluate_model(best_model, X_train, y_train, "training")
            results.append({'Dataset': 'Training', 'Accuracy': acc, 'F1 Score': f1, 'Log Loss': loss})
            
            val_acc, val_f1, val_loss = evaluate_model(best_model, X_val, y_val, "validation")
            results.append({'Dataset': 'Validation', 'Accuracy': val_acc, 'F1 Score': val_f1, 'Log Loss': val_loss})
        else:
            acc, f1, loss = evaluate_model(best_model, X, y, name)
            results.append({'Dataset': name, 'Accuracy': acc, 'F1 Score': f1, 'Log Loss': loss})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("plt/sbert/results.csv", index=False)
    print("\n=== Final Results ===")
    print(results_df)

if __name__ == "__main__":
    main()

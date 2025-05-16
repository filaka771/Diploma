import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import joblib

os.makedirs("plt/tfidf", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plt/tfidf/vector_distributions", exist_ok=True)
os.makedirs("plt/tfidf/confusion_matrices", exist_ok=True)
os.makedirs("plt/tfidf/feature_importance", exist_ok=True)

def clean_text(text):
    if " -- " in text:
        return text.split(" -- ", 1)[1]
    return text

def load_dataset(true_path, fake_path, n_samples=None):
    with tqdm(total=4, desc="Loading dataset") as pbar:
        true = pd.read_csv(true_path)
        fake = pd.read_csv(fake_path)
        pbar.update(2)
        
        if n_samples is not None:
            true = true.sample(n=min(n_samples, len(true)), random_state=42)
            fake = fake.sample(n=min(n_samples, len(fake)), random_state=42)
            
        true['label'] = 1
        fake['label'] = 0
        pbar.update(1)
        
        true['text'] = true['text'].astype(str).apply(clean_text) if 'text' in true.columns else true['article'].astype(str).apply(clean_text)
        fake['text'] = fake['text'].astype(str) if 'text' in fake.columns else fake['article'].astype(str)
        pbar.update(1)
        
        df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)
    return df['text'], df['label']

def plot_vector_distribution(X, y, dataset_name, stage="raw"):
    """Visualize vector distribution using t-SNE"""
    plt.figure(figsize=(10, 8))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X_scaled)
    
    tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
    tsne_df['label'] = y.map({0: 'Fake', 1: 'True'})
    
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        palette={'Fake': 'red', 'True': 'blue'},
        data=tsne_df,
        alpha=0.6,
        s=50
    )
    
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.gca().set_aspect('equal')  
    
    plt.title(f't-SNE Distribution ({dataset_name}, {stage} data)')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend(title='Class')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"plt/tfidf/vector_distributions/tsne_{dataset_name}_{stage}.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, dataset_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=['Fake', 'True'], 
                yticklabels=['Fake', 'True'])
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plt/tfidf/confusion_matrices/cm_{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, dataset_name, top_n=20):
    if hasattr(model, 'coef_'):
        coef = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        coef = model.feature_importances_
    else:
        print(f"No feature importance available for {dataset_name}")
        return
    
    if len(coef) != len(feature_names):
        print(f"Warning: Feature dimension mismatch ({len(coef)} != {len(feature_names)}). Plotting component weights.")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(coef[:top_n])), coef[:top_n])
        plt.title(f"Top {top_n} Component Weights ({dataset_name})")
        plt.xlabel("Component Index")
        plt.ylabel("Weight")
        plt.savefig(f"plt/tfidf/feature_importance/component_weights_{dataset_name}.png", bbox_inches='tight', dpi=300)
        plt.close()
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': coef
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f"Top {top_n} Features Importance ({dataset_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"plt/tfidf/feature_importance/feature_importance_{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def train_lsa_model(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lsa', TruncatedSVD()),
        ('clf', LogisticRegression(max_iter=1000, verbose=1))
    ])
    
    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [2, 5],
        'lsa__n_components': [100, 200],
        'clf__C': [0.1, 1, 10],
    }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='f1',
        n_jobs=-1,
        verbose=10
    )
    
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, dataset_name):
    print(f"\nEvaluating on {dataset_name}...")
    with tqdm(total=6, desc="Evaluation steps") as pbar:
        if hasattr(model, 'named_steps'):
            if 'tfidf' in model.named_steps:
                tfidf = model.named_steps['tfidf']
                X_vectors = tfidf.transform(X)
                feature_names = tfidf.get_feature_names_out()
                if 'lsa' in model.named_steps:
                    X_vectors = model.named_steps['lsa'].transform(X_vectors)
        else:
            X_vectors = X
            feature_names = None
            
        pbar.update(1)
        
        plot_vector_distribution(X_vectors, y, dataset_name, "transformed")
        pbar.update(1)
        
        y_pred = model.predict(X)
        pbar.update(1)
        
        y_proba = model.predict_proba(X)
        pbar.update(1)
        
        plot_confusion_matrix(y, y_pred, dataset_name)
        pbar.update(1)
        
        if feature_names is not None and hasattr(model.named_steps['clf'], 'coef_'):
            plot_feature_importance(model.named_steps['clf'], feature_names, dataset_name)
            pbar.update(1)
            
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        loss = log_loss(y, y_proba)
        
    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {loss:.4f}")
    
    return acc, f1, loss

def main():
    print("Loading datasets...")
    datasets = {
        "train": {"paths": ("../true.csv", "../fake.csv"), "n_samples": 12000},
        "test1": {"paths": ("../true1.csv", "../fake1.csv"), "n_samples": 4000},
        "test2": {"paths": ("../true2.csv", "../fake2.csv"), "n_samples": 4000}
    }
    
    loaded_data = {}
    for name, config in tqdm(datasets.items(), desc="Loading all datasets"):
        loaded_data[name] = load_dataset(*config["paths"], n_samples=config["n_samples"])
        
    print("\nVisualizing raw data distributions...")
    temp_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('lsa', TruncatedSVD(n_components=100))
    ])
    
    for name, (X, y) in loaded_data.items():
        X_vectors = temp_pipe.fit_transform(X)
        plot_vector_distribution(X_vectors, y, name, "raw")
        
    X_train, X_val, y_train, y_val = train_test_split(
        loaded_data["train"][0], loaded_data["train"][1], 
        test_size=0.2, 
        random_state=42
    )
    
    best_model = train_lsa_model(X_train, y_train)
    
    joblib.dump(best_model, "models/tfidf_lsa_model.pkl")
    
    results = []
    for name, (X, y) in tqdm(loaded_data.items(), desc="Evaluating datasets"):
        if name == "train":
            acc, f1, loss = evaluate_model(best_model, X_train, y_train, "Training")
            results.append({'Dataset': 'Training', 'Accuracy': acc, 'F1 Score': f1, 'Log Loss': loss})
            
            val_acc, val_f1, val_loss = evaluate_model(best_model, X_val, y_val, "Validation")
            results.append({'Dataset': 'Validation', 'Accuracy': val_acc, 'F1 Score': val_f1, 'Log Loss': val_loss})
        else:
            acc, f1, loss = evaluate_model(best_model, X, y, name)
            results.append({'Dataset': name, 'Accuracy': acc, 'F1 Score': f1, 'Log Loss': loss})
            
    pd.DataFrame(results).to_csv("plt/tfidf/results.csv", index=False)
    print("\nTraining complete! Results saved to plt/tfidf/results.csv")

if __name__ == "__main__":
    main()

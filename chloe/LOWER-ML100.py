import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

def run_batch_experiments(input_dir, output_dir, filtration_suffix="_Filt_vect.npy", n_iterations=100):
    # 1. Create the output folder if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Gather all files from the INPUT directory
    files = sorted(list(input_dir.glob(f'*{filtration_suffix}')))
    
    if not files:
        print(f"⚠️ No files found in {input_dir} with suffix: {filtration_suffix}")
        return

    X, y = [], []
    for f in files:
        X.append(np.load(f))
        y.append(1 if "microgravity" in f.name.lower() else 0)
    X, y = np.array(X), np.array(y)
    
    filt_name = filtration_suffix.replace(".npy", "").replace("_", "")
    
    # 3. 100x Iteration Logic
    results = defaultdict(list)
    classifiers = {
        "SVM": SVC(kernel="rbf", gamma=2),
        "NN": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
    }
    
    print(f"🚀 Starting {n_iterations} iterations of classification...")
    for seed in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
        for name, clf in classifiers.items():
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results[name].append({"Accuracy": accuracy_score(y_test, pred), "F1": f1_score(y_test, pred)})

    # 4. Save detailed 100-run history AND summary stats to the OUTPUT directory
    for name, data in results.items():
        df = pd.DataFrame(data)
        
        # Save the full 100-iteration record
        df.to_csv(output_dir / f"{filt_name}_{name}_full_record.csv", index=False)
        
        # Calculate and save summary stats
        summary = pd.DataFrame({
            "Mean": df.mean(),
            "Std": df.std()
        })
        summary.to_csv(output_dir / f"{filt_name}_{name}_summary_stats.csv")
        
    print(f"✅ Analysis complete. Full records and summary stats saved to: {output_dir}")

if __name__ == '__main__':
    # Set your source data folder
    SOURCE_DIR = Path(r"C:\Users\chloe\OneDrive - Simpson College\IMAGES2.0\All Images\preprocessed_images")
    
    # Set your separate results folder
    RESULTS_DIR = SOURCE_DIR / "analysis_results"
    
    run_batch_experiments(SOURCE_DIR, RESULTS_DIR, filtration_suffix="_Filt_vect.npy", n_iterations=100)
import os
import numpy as np
import subprocess
from itertools import product
from scipy.stats import skew

# Parametri da testare
margin_taus = [0.05, 0.1, 0.15, 0.2]
pos_betas = [0.005, 0.01, 0.015, 0.02]
num_anomalies_list = [5, 10]
data_strategies = ["0", "0,1"]  # puoi estendere a "0,2" e "0,1,2"

# Directory dove salvare i punteggi
score_dir = "auto_scores"
os.makedirs(score_dir, exist_ok=True)

results = []

def run_and_extract_score(margin_tau, pos_beta, num_anomalies, data_strategy):
    exp_name = f"auto_margin{margin_tau}_beta{pos_beta}_na{num_anomalies}_ds{data_strategy.replace(',', '_')}"
    cmd = f"""
    python main.py \
        --flow_arch conditional_flow_model \
        --gpu 0 \
        --class_name bottle \
        --with_fas \
        --data_strategy {data_strategy} \
        --num_anomalies {num_anomalies} \
        --not_in_test \
        --exp_name {exp_name} \
        --focal_weighting \
        --pos_beta {pos_beta} \
        --margin_tau {margin_tau} \
        --meta_epochs 5 --sub_epochs 2
    """
    print("Eseguo:", cmd)
    subprocess.run(cmd, shell=True)

    # Carica i punteggi da validate()
    try:
        score_file = os.path.join("output", exp_name, "img_scores.npy")
        scores = np.load(score_file)
        sk = skew(scores)
        print(f"Skewness = {sk:.4f}")
        return sk
    except Exception as e:
        print("Errore nel caricamento punteggi:", e)
        return -999

# Loop sugli iperparametri
for margin_tau, pos_beta, num_anomalies, data_strategy in product(margin_taus, pos_betas, num_anomalies_list, data_strategies):
    sk_score = run_and_extract_score(margin_tau, pos_beta, num_anomalies, data_strategy)
    results.append({
        "margin_tau": margin_tau,
        "pos_beta": pos_beta,
        "num_anomalies": num_anomalies,
        "data_strategy": data_strategy,
        "skewness": sk_score
    })

# Ordina e salva
results = sorted(results, key=lambda x: x["skewness"], reverse=True)

import json
with open("auto_uad_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n📈 Top 3 configurazioni migliori (skewness più alta):")
for r in results[:3]:
    print(r)

# 🔍 Detecting Illicit Bitcoin Transactions with Graph Neural Networks

End-to-end pipeline that trains Graph Neural Networks (GNNs) on the **Elliptic Bitcoin dataset** to flag illicit cryptocurrency transactions (money laundering / fraud) — and explains *why* each transaction was flagged.

Unlike tabular models that treat every transaction as independent, GNNs use **message passing** over the payment graph, so a transaction's risk score is informed by the company it keeps. The project benchmarks three GNN architectures against an XGBoost baseline, handles extreme class imbalance with Focal Loss, and uses GNNExplainer to surface the neighborhood and features behind each prediction.

---

## 📊 The Problem

Fraud detection on Bitcoin is hard because the signal is rare and the data is relational:

| Property | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges (payment flows) | 234,355 |
| Time steps | 49 (~2 weeks each) |
| Features per node | 166 (94 local + 72 aggregated) |
| Labeled illicit | ~2% |
| Labeled licit | ~21% |
| Unlabeled | ~77% |

With only ~2% illicit nodes, naive accuracy is meaningless — a model that labels everything "licit" scores 98% while catching zero fraudsters. The project is evaluated on **PR-AUC**, **Macro F1**, and **Illicit-class F1/Recall** instead.

---

## 🧱 Pipeline

The full workflow lives in [`GNN_Elliptic_Fraud_Detection_Final.ipynb`](GNN_Elliptic_Fraud_Detection_Final.ipynb):

1. **Environment setup** — PyTorch + PyTorch Geometric
2. **Data loading & EDA** — class distribution, feature correlation, graph degree distribution
3. **Preprocessing** — label remapping (`illicit→1`, `licit→0`, `unknown→−1`), feature normalization, train/val/test masks
4. **Graph construction** — a `torch_geometric.data.Data` object (`x`, `edge_index`, `y`, masks)
5. **Baseline** — XGBoost on the 166 hand-crafted features (no graph structure)
6. **GNN models** — GCN, GraphSAGE, GAT
7. **Training** — Focal Loss, Adam, early stopping on validation PR-AUC
8. **Evaluation** — imbalance-aware metrics across all models
9. **Explainability** — GNNExplainer for per-prediction subgraph + feature attributions
10. **Results & conclusions** — including a feature ablation study

### Models

| Model | Idea |
|---|---|
| **XGBoost** | Gradient-boosted trees on tabular features — baseline with no graph awareness |
| **GCN** | Symmetric-normalized neighborhood aggregation (Kipf & Welling, 2017) |
| **GraphSAGE** | Inductive sampling-and-aggregating — scores *new* transactions without retraining (Hamilton et al., 2017) |
| **GAT** | Attention-weighted aggregation — weights double as explainable evidence (Veličković et al., 2018) |

All GNNs follow `Input → Conv1 (ReLU + Dropout) → Conv2 → Linear` and are trained with **Focal Loss** (`α=0.75, γ=2`) to focus learning on the rare illicit class.

---

## 📈 Results

Evaluated on the held-out test set (time steps 43–49):

| Model | PR-AUC | Macro F1 | Illicit F1 |
|---|---|---|---|
| XGBoost (baseline) | 0.0427 | 0.5044 | 0.0292 |
| GCN | 0.0385 | 0.4833 | 0.0132 |
| GraphSAGE | 0.0420 | 0.4960 | 0.0287 |
| GAT | 0.0398 | 0.4345 | **0.0934** |

**Takeaways**
- **No model wins outright.** XGBoost has the best PR-AUC (0.0427) and Macro F1 (0.5044), so on overall ranking the GNNs only match — not beat — the tabular baseline.
- **GAT is the most useful model for fraud.** It triples XGBoost's illicit-class F1 (0.0934 vs 0.0292) — the metric that actually matters for catching fraudsters — by aggregating neighborhood context tabular models can't reach, and its attention weights double as explainable evidence on which neighbors drove a flag.
- **GraphSAGE** is roughly on par with XGBoost while being **inductive**, so it can score new transactions without retraining.
- An **ablation study** shows GNNs can recover much of the signal from the 72 hand-engineered aggregated features by learning graph structure directly through message passing.
- **Focal Loss** is essential — plain cross-entropy collapses to predicting "licit" for everything.

> Note: absolute scores are low because of the dataset's severe imbalance and concept drift across time steps; the value here is the *relative* comparison and the explainability pipeline. See the [comprehensive report](Comprehensive%20Report-%20GNN.pdf) for full discussion.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install torch torch_geometric pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
> PyTorch Geometric needs to match your PyTorch/CUDA version — see the
> [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
> The notebook also contains an install cell you can run once.

### 2. Get the data
Download the three CSV files (`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`) from the
[Elliptic dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place them in the project root, next to the notebook.

### 3. Run
```bash
jupyter notebook GNN_Elliptic_Fraud_Detection_Final.ipynb
```
Run the cells top to bottom — all plots and metrics are generated inline.

---

## 📁 Repository

```
GNN_Elliptic_Fraud_Detection_Final.ipynb   # full pipeline (this is the project)
Comprehensive Report- GNN.pdf              # detailed write-up
GNN final project.pdf                      # project report
```

---

## ⚠️ Limitations

- **Transductive** setting for most GNNs — embeddings must be recomputed when new nodes arrive (GraphSAGE is the inductive exception).
- **77% unlabeled** nodes add noise; semi-supervised label propagation could help.
- **Concept drift** across 49 time steps — models trained on early steps degrade on later ones.
- GNNExplainer outputs are heuristic approximations, not formal causal explanations.

---

## 📚 References

- Weber et al. (2019) — *Anti-Money Laundering in Bitcoin* (Elliptic dataset)
- Kipf & Welling (2017) — *Semi-Supervised Classification with Graph Convolutional Networks*
- Hamilton et al. (2017) — *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Veličković et al. (2018) — *Graph Attention Networks*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- Ying et al. (2019) — *GNNExplainer: Generating Explanations for Graph Neural Networks*

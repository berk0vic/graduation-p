# CSE492 — Proje Yol Haritası
**Berk Tahir Kılıç | Hybrid GAT+VAE for Financial Fraud Detection**

---

## Notebook Yapısı (Sadeleştirilmiş)

| # | Notebook | İçerik | Durum |
|---|----------|--------|-------|
| 01 | `01_eda_ieee_cis.ipynb` | IEEE-CIS EDA + graph oluşturma | ✅ Tamamlandı |
| 02 | `02_eda_paysim.ipynb` | PaySim EDA | ✅ Tamamlandı |
| 03 | `03_model_training_ieee_cis.ipynb` | GCN baseline + Hybrid GAT+VAE eğitimi (IEEE-CIS) | ⏳ Çalıştırılacak |
| 04 | `04_elliptic_experiment.ipynb` | Elliptic Bitcoin deneyi (farklı domain) | ⏳ Çalıştırılacak |
| 05 | `05_explainability_scalability.ipynb` | SHAP + PaySim scalability testi | ⏳ Çalıştırılacak |

---

## Dataset Kullanımı

| Dataset | Kullanım | Ne gösterir |
|---------|----------|-------------|
| IEEE-CIS | Eğitim + Test (ana deney) | F1, Recall, AUC-ROC |
| Elliptic | Eğitim + Test (2. deney) | Farklı domain'de genelleme |
| PaySim | Scalability testi | 6M+ işlemde graph oluşturma + inference süresi |

---

## Çalıştırma Sırası

```
[✅] NB 01 — IEEE-CIS EDA (tamamlandı, graph hazır)
[✅] NB 02 — PaySim EDA (tamamlandı)
[ ] NB 03 — GCN + Hybrid eğitimi (IEEE-CIS)
[ ] NB 04 — Elliptic deneyi
[ ] NB 05 — SHAP explainability + PaySim scalability
```

### NB 03 — Önkoşul: Yok (graph zaten NB 01'de oluşturuldu)
### NB 04 — Önkoşul: Yok
### NB 05 — Önkoşul: NB 03 tamamlanmış olmalı (model checkpoint gerekli)

---

## Çıktılar

```
results/
├── tables/
│   ├── ieee_cis_results.csv    ← NB 03
│   └── elliptic_results.csv    ← NB 04
├── figures/
│   ├── training_loss.png       ← NB 03
│   ├── model_comparison.png    ← NB 03
│   ├── shap_feature_importance.png ← NB 05
│   └── scalability.png         ← NB 05
├── models/
│   ├── hybrid_gatvae_ieee_cis.pt  ← NB 03
│   └── hybrid_gatvae_elliptic.pt  ← NB 04
└── scalability_results.json    ← NB 05
```

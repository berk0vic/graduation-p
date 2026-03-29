# Model Nasıl Çalışır + Tüm Kodların Açıklaması
## PyTorch Öğrenme Rehberi — CSE492 Fraud Detection Projesi

Bu rehber projenin model mimarisini, tüm kodların ne yaptığını ve PyTorch temellerini açıklar.

---

# BÖLÜM 1: PyTorch Temelleri

## 1.1 Tensor Nedir?

```python
import torch

# Tensor = Numpy benzeri, ama GPU'da çalışabilen çok boyutlu dizi
x = torch.tensor([1.0, 2.0, 3.0])           # 1D vektör
y = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2D matris

# shape: boyutlar
print(x.shape)   # torch.Size([3])
print(y.shape)   # torch.Size([2, 2])
```

- **Tensor**: Sayıların çok boyutlu dizisi (vektör, matris, 3D küp...)
- **dtype**: `float32`, `long` (int64) — model hesaplamaları için genelde float32
- **device**: `cpu` veya `cuda`/`mps` — hesaplama nerede yapılıyor

## 1.2 nn.Module — Model Sınıfının Anası

PyTorch'ta her model `nn.Module` sınıfından türetilir:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # ZORUNLU — parent sınıfı initialize et
        self.fc = nn.Linear(10, 5)  # 10 giriş → 5 çıkış

    def forward(self, x):
        return self.fc(x)  # x'i bu katmandan geçir
```

- **`__init__`**: Katmanları tanımla (Linear, Conv2d, vs.) — `nn.Linear`, `nn.Conv2d` otomatik parametre tutar
- **`forward`**: Veriyi nasıl işleyeceğini yaz — `model(x)` çağrıldığında `forward` çalışır
- **`super().__init__()`**: Mutlaka yaz; yoksa model düzgün çalışmaz

## 1.3 Temel Katmanlar

| Katman | Ne yapar |
|--------|----------|
| `nn.Linear(in, out)` | `y = x @ W + b` — tam bağlı katman |
| `nn.Dropout(p)` | Eğitimde rastgele p oranında nöron kapat (overfitting azaltır) |
| `F.elu(x)` | Aktivasyon: ELU(x) = x if x>0 else α*(e^x - 1) |
| `F.relu(x)` | ReLU(x) = max(0, x) |

## 1.4 Eğitim Döngüsü

```python
model.train()
optimizer.zero_grad()      # Önceki gradient'leri sıfırla
outputs = model(inputs)    # İleri geçiş (forward pass)
loss = criterion(outputs, targets)
loss.backward()            # Geri yayılım (gradient hesapla)
optimizer.step()           # Parametreleri güncelle
```

- **`zero_grad()`**: Her batch'te gradient'ler birikir; sıfırlamazsan eskisiyle toplanır
- **`backward()`**: loss'a göre tüm parametrelerin gradient'ini hesaplar
- **`step()`**: optimizer, gradient'lere göre parametreleri günceller

## 1.5 torch.no_grad()

```python
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data)  # Gradient HESAPLANMAZ — bellekte tasarruf
```

- Test/eval sırasında gradient gerekmez → `no_grad` ile hesaplamayı kapatırsın
- `model.eval()`: Dropout/BatchNorm davranışını "inference" moduna alır

---

# BÖLÜM 2: Model Mimarisi — Nasıl Çalışır?

## 2.1 Genel Akış

```
[Ham Veri: Tablo]
      ↓
[Grafik Oluştur] — Hesap, İşlem, Merchant node'ları + edge'ler
      ↓
[GAT Encoder] — Grafik üzerinden mesaj geçişi → her node için embedding
      ↓
[Transaction Embedding h_t] — Sadece işlem node'larının vektörleri
      ↓
    ┌─────────────────────────────────────┐
    │ [VAE]                    [Classifier] │
    │ Encoder → z → Decoder    h_t → MLP   │
    │ x_recon (yeniden oluşturma)          │
    └─────────────────────────────────────┘
      ↓                    ↓
  recon_err            logit (sınıflandırma skoru)
      ↓                    ↓
      └────────┬───────────┘
               ↓
    [h_t || recon_err] → final classifier → fraud_score
```

## 2.2 Neden GAT?

- **Grafik**: Hesap–işlem–merchant ilişkilerini temsil eder
- **GAT (Graph Attention)**: Her node, komşularına "dikkat" verir — önemli komşular daha çok ağırlık alır
- **Çıktı**: Her transaction node için bir vektör (embedding) — komşu bilgisiyle zenginleştirilmiş

## 2.3 Neden VAE?

- **Anomali**: VAE normal pattern'leri öğrenir; fraud gibi nadir pattern'leri iyi "yeniden oluşturamaz"
- **Reconstruction error**: Yüksek hata → muhtemel anomali (fraud)
- **Hybrid**: Supervised (GAT + classifier) + Unsupervised (VAE) birlikte daha güçlü

## 2.4 Neden Focal Loss?

- Veri dengesiz: %96.5 normal, %3.5 fraud
- Normal BCE kolay negatiflere (normal işlemler) fazla odaklanır
- Focal Loss kolay örneklere düşük ağırlık verir → model "zor" fraud örneklerine odaklanır

---

# BÖLÜM 3: Dosya Dosya Kod Açıklamaları

## 3.1 `src/data/ieee_cis_loader.py`

```python
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ieee_cis"
```
- `Path(__file__)`: Bu dosyanın konumu
- `.parent.parent.parent`: 3 üst klasöre çık → proje kökü
- `/"data"/"raw"/"ieee_cis"`: Veri klasörüne git

```python
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW_DIR / "train_transaction.csv")
    train_id = pd.read_csv(RAW_DIR / "train_identity.csv")
    train = train.merge(train_id, on="TransactionID", how="left")
```
- `merge(..., on="TransactionID", how="left")`: İki tabloyu TransactionID üzerinden birleştir
- `how="left"`: train'deki her satır kalır; identity'de yoksa NaN gelir

---

## 3.2 `src/graph/builder.py`

### HeteroData Nedir?

PyTorch Geometric'te **heterogeneous graph** = farklı tiplerde node ve edge:
- Node tipleri: `account`, `transaction`, `merchant`
- Edge tipleri: `(account, initiates, transaction)`, `(transaction, paid_to, merchant)`

### IEEE-CIS Grafik Oluşturma

```python
accounts = pd.unique(df["card1"].dropna()).tolist()
acct_map = {a: i for i, a in enumerate(accounts)}
```
- `card1`: Kart/banka hesabı proxy'si
- Her benzersiz card1 bir account node
- `acct_map`: account id → sıra numarası (0, 1, 2...)

```python
df["merchant_key"] = df["addr1"].astype(str) + "_" + df["ProductCD"].astype(str)
merchants = pd.unique(df["merchant_key"].dropna()).tolist()
```
- Merchant = adres + ürün tipi kombinasyonu

```python
data["account"].x = torch.zeros(len(accounts), 4)   # placeholder
data["merchant"].x = torch.zeros(len(merchants), 2) # placeholder
data["transaction"].x = _ieee_txn_features(df)
data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
```
- `data["account"].x`: Her account node için 4 boyutlu feature (şimdilik sıfır)
- `data["transaction"].x`: Her işlem için 5 feature (TransactionAmt, card1, dist1, C1, C2)
- `data["transaction"].y`: Etiket (0=normal, 1=fraud)

```python
data["account", "initiates", "transaction"].edge_index = torch.tensor([src_acct, dst_txn_acct], ...)
```
- `edge_index`: [2, num_edges] tensor
- Satır 0: kaynak node index'leri (hangi account)
- Satır 1: hedef node index'leri (hangi transaction)
- Örnek: `[0, 1, 2], [5, 10, 15]` → account 0→txn 5, account 1→txn 10, ...

---

## 3.3 `src/models/vae.py` — Satır Satır

```python
class TransactionVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()

        # Encoder: x → h → (mu, logvar)
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)   # 32 → 64
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)    # 64 → 16
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
```
- **Encoder**: Girdiyi önce 64 boyuta, sonra `mu` ve `logvar`'a çevirir
- `mu`: latent dağılımın ortalaması
- `logvar`: latent dağılımın log(varyans)'ı — varyans hep pozitif olmalı, log alarak sınır kaldırılır

```python
        # Decoder: z → h → x_recon
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)   # 16 → 64
        self.dec_out = nn.Linear(hidden_dim, input_dim)     # 64 → 32
```
- **Decoder**: Latent vektör z'den girdiyi yeniden oluşturur

```python
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.elu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)
```
- `F.elu`: Aktivasyon fonksiyonu
- Çıkış: `(mu, logvar)` — Gaussian dağılım parametreleri

```python
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)   # std = sqrt(var)
            eps = torch.randn_like(std)     # N(0,1) örnek
            return mu + eps * std           # z = mu + eps*std (reparameterization trick)
        return mu  # inference'da deterministik
```
- **Reparameterization trick**: Gradyanı z üzerinden backprop ile geçirmek için
- Eğitimde: `z ~ N(mu, sigma²)` örnekle
- Inference'da: sadece mu kullan (deterministik)

```python
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.dec_fc1(z))
        return self.dec_out(h)
```

```python
    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_recon, x, reduction="none").mean(dim=1)
```
- `reduction="none"`: Her örnek için ayrı MSE
- `mean(dim=1)`: Her satır (örnek) için ortalama → (N,) boyutlu vektör
- Yüksek hata = anomali = muhtemel fraud

---

## 3.4 `src/models/gat_encoder.py` — Satır Satır

### GAT (Graph Attention Network) Mantığı

Her node, komşularından bilgi toplar. **Attention**: Hangi komşunun ne kadar önemli olduğunu öğrenir.

```python
self.input_proj = nn.ModuleDict({
    nt: nn.Linear(in_channels[nt], hidden_channels)
    for nt in node_types if nt in in_channels
})
```
- `ModuleDict`: Her node tipi için ayrı Linear katman
- `account`: 4 → 64, `merchant`: 2 → 64, `transaction`: 5 → 64 (boyutlar örnek)
- Farklı tiplerin feature boyutları farklı olabilir; hepsini aynı hidden boyuta getirir

```python
conv = HeteroConv({
    et: GATConv(in_c, out_c, heads=heads, concat=concat, ...)
    for et in edge_types
}, aggr="sum")
```
- `HeteroConv`: Her edge tipi için ayrı GATConv
- `GATConv`: Attention ile mesaj geçişi
- `heads`: Çoklu attention head (4 head = 4 farklı "bakış açısı")
- `concat=True`: Head'leri yan yana birleştir → boyut 4 katına çıkar
- `aggr="sum"`: Farklı edge tiplerinden gelen mesajları topla

```python
def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
    h = {nt: self.act(self.input_proj[nt](x)) for nt, x in x_dict.items() ...}

    for conv in self.convs:
        h = conv(h, edge_index_dict)
        h = {nt: self.act(self.dropout(t)) for nt, t in h.items()}

    return h
```
- `x_dict`: `{"account": tensor, "transaction": tensor, "merchant": tensor}`
- `edge_index_dict`: `{(src, edge_type, dst): edge_index}` 
- Her conv: mesaj geçişi (message passing)
- Çıkış: Her node tipi için güncel embedding

---

## 3.5 `src/models/hybrid_model.py` — Satır Satır

```python
self.gat = HeteroGATEncoder(...)
self.vae = TransactionVAE(input_dim=gat_out, ...)  # gat_out = 32
self.classifier = nn.Sequential(
    nn.Linear(gat_out + 1, 32),   # h_t (32) + recon_err (1) = 33 giriş
    nn.ELU(),
    nn.Dropout(dropout),
    nn.Linear(32, 1),
)
```
- GAT çıkışı 32 boyut
- VAE `gat_out` boyutundan alıyor (32)
- Classifier: `[h_t, recon_err]` → tek skor (logit)

```python
def forward(self, x_dict, edge_index_dict):
    h_dict = self.gat(x_dict, edge_index_dict)
    h_t = h_dict["transaction"]  # Sadece transaction embedding'leri

    x_recon, mu, logvar = self.vae(h_t)
    recon_err = TransactionVAE.reconstruction_error(h_t, x_recon)

    combined = torch.cat([h_t, recon_err.unsqueeze(1)], dim=1)
    logit = self.classifier(combined).squeeze(1)

    return {
        "logit": logit,
        "fraud_score": torch.sigmoid(logit),
        ...
    }
```
- `h_t`: (N, 32)
- `recon_err`: (N,) → `unsqueeze(1)` → (N, 1)
- `torch.cat(..., dim=1)`: (N, 33)
- `sigmoid(logit)`: 0–1 arası olasılık

---

## 3.6 `src/training/losses.py` — Satır Satır

### Focal Loss

```python
bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
pt = torch.exp(-bce)
focal = alpha * (1 - pt) ** gamma * bce
```
- `logits`: Model çıkışı (sigmoid öncesi)
- `bce`: Her örnek için binary cross entropy
- `pt`: Doğru sınıfın tahmin olasılığı (yüksek = kolay örnek)
- `(1 - pt)^gamma`: Kolay örneklerin ağırlığını azaltır (gamma=2)
- Kolay negatif (normal işlem, doğru tahmin) → düşük loss

### KL Divergence (VAE için)

```python
def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
```
- VAE'nin latent dağılımını N(0,1)'e yaklaştırmak için
- Formül: KL(N(mu, sigma²) || N(0,1)) = -0.5 * sum(1 + log(sigma²) - mu² - sigma²)

### Toplam Loss

```python
loss = l_cls + lambda1 * l_recon + lambda2 * l_kl
```
- Sınıflandırma + VAE reconstruction + VAE KL

---

## 3.7 `src/training/trainer.py` — Satır Satır

```python
if device == "auto":
    if torch.backends.mps.is_available():
        device = "mps"   # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
```
- Otomatik cihaz seçimi

```python
self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
```
- **Adam**: Adaptif learning rate
- **ReduceLROnPlateau**: Val loss düşmezse 5 epoch sonra lr'yi yarıya indir

```python
def train_epoch(self, data):
    self.model.train()
    data = data.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.model(data.x_dict, data.edge_index_dict)
    ...
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
```
- `clip_grad_norm_`: Gradient patlamasını önlemek için normu 1 ile sınırla

---

# BÖLÜM 4: PyTorch Öğrenme Kaynakları

1. **Resmi tutorial**: https://pytorch.org/tutorials/
2. **nn.Module**: "PyTorch nn.Module tutorial"
3. **Gradient / backward**: "PyTorch autograd explained"
4. **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

# Özet Tablo: Hangi Dosya Ne Yapar?

| Dosya | Ana Görevi |
|-------|-------------|
| `ieee_cis_loader.py` | CSV'leri yükle, merge et |
| `builder.py` | Tabloyu HeteroData grafiğine çevir |
| `vae.py` | Transaction embedding → mu, logvar, x_recon, recon_err |
| `gat_encoder.py` | Grafik → her node için embedding |
| `hybrid_model.py` | GAT + VAE + classifier birleştir |
| `losses.py` | Focal + reconstruction + KL |
| `trainer.py` | Epoch döngüsü, optimizer, backward |
| `baselines.py` | XGBoost, RF, GCN, GAT-only, VAE-only karşılaştırma |
| `metrics.py` | F1, recall, precision, AUC hesapla |
| `shap_explainer.py` | "Hangi feature fraud'a katkı yaptı?" |
| `gnn_explainer.py` | "Hangi komşu node/edge önemli?" |

---

# BÖLÜM 5: Pratik — PyTorch Denemeleri

Aşağıdaki kodu bir notebook'ta çalıştırarak PyTorch'u deneyebilirsin:

```python
# 1. Basit Linear model
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)  # 5 giriş → 1 çıkış

    def forward(self, x):
        return self.fc(x).squeeze(-1)

model = SimpleModel()
x = torch.randn(10, 5)  # 10 örnek, 5 feature
out = model(x)
print(out.shape)  # (10,)

# 2. Loss + backward
targets = torch.randint(0, 2, (10,)).float()
loss = nn.functional.binary_cross_entropy_with_logits(out, targets)
loss.backward()
print(model.fc.weight.grad)  # Gradient'ler hesaplandı
```

---

# BÖLÜM 6: Baselines.py Kısa Açıklama

- **XGBoost / Random Forest**: Tabular veri için — grafik yok, sadece feature vektörü
- **GCNBaseline**: Homogeneous graph — tek node tipi, GAT yerine basit GCN
- **GATOnlyBaseline**: HeteroGAT var ama VAE yok — sadece GAT + classifier
- **VAEOnlyBaseline**: VAE var ama grafik yok — ham feature'lara VAE, anomali = reconstruction error

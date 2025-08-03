下面是 **CLE-ViT（Contrastive Learning Encoded ViT）** 在 **timm** 骨幹上的**可落地實作規格**與**組件設計**（含關鍵超參數、資料流程、損失定義、訓練與推論流程，以及 PyTorch/timm 範例樣板）。設計完全對齊論文：雙視角（標準增強視角作為 *anchor*；「遮擋＋拼圖打亂」的 *positive*）、以 **triplet** 做**實例級**對比學習、同時對 *anchor* 與 *positive* 走分類路徑，推論時關閉自監督模組。

---

## 1) 模型骨幹（timm）與輸入規格

* **骨幹**：論文主實驗採 **Swin-B**，以 ImageNet-21K 預訓練初始化。對應 timm 可用
  `swin_base_patch4_window7_224` 或高解析版本（若要 448×448 輸入，Swin 支援彈性尺寸）。
* **輸入尺寸與增強**：訓練時將影像 **resize 至 600×600**，再 **random crop 至 448×448**；測試時 **center crop 448×448**。標準增強含水平翻轉、顏色抖動、隨機旋轉；最佳化採 **AdamW**，論文中 **batch=12、lr=1e-3**。
* **推論**：自監督（對比）分支在**推論階段關閉**，僅保留分類路徑。

> **timm 取用備註**：若沿用 Swin-B 並想與 448×448 對齊，可直接餵 448 輸入；若使用 ViT 系列（如 `vit_base_patch16_224`），一樣支援 448 輸入，但需留意 patch 與位置編碼插值。

---

## 2) 兩視角資料生成（核心自監督設計）

### 2.1 Anchor（標準視角）

* 對原圖施作**標準增強**（flip/rotate/color jitter 等）得到 **Ia**。

### 2.2 Positive（強增強視角）：**遮擋（mask）＋拼圖打亂（shuffle）**

1. 以與 anchor 相同之標準增強作為 base，再**隨機遮擋一塊矩形區域**（Cutout 風格）：

   $$
   I_p^*(i,j)=\begin{cases}
   I_a(i,j), & (i,j)\in H\\
   0, & (i,j)\notin H
   \end{cases}
   $$

   其中 $H$ 為要遮擋的矩形集合；遮擋比例 $\alpha=\tfrac{k\cdot t}{H\cdot W}$。論文設定 $\alpha\in[0.15,0.45]$。 &#x20;
2. 將 $I_p^*$ **等分成 $s\times s$ 區塊（$s=4$）並**隨機打亂區塊位置\*\*，產生 $I_p$。這一步避免模型僅靠遮擋區重建而忽略類別判斷，提升學習難度與泛化。&#x20;

### 2.3 負樣本（Negative）

* 採**實例級對比**設定：對每個 anchor（批內），**隨機抽一張其他樣本**的 anchor 當負樣本 $I_n$。

---

## 3) 特徵抽取與分類頭

* **Patch/Global 特徵**：骨幹產生 patch 特徵 $V\in\mathbb{R}^{N_p\times D}$，**全域表徵** $u=\frac{1}{N_p}\sum_i v_i$（Swin 無 CLS，平均後接分類頭）。&#x20;
* **分類**：對 anchor 與 positive 均各自通過分類頭（softmax），以 **交叉熵**訓練。

---

## 4) 自監督對比學習（實例級 Triplet）

* **嵌入**：取 **anchor/positive/negative** 的 **全域特徵 $u_a,u_p,u_n$**，先 **L2 normalize**。
* **損失**：採 **triplet**（margin β=1），鼓勵 $u_a$ 靠近 $u_p$ 遠離 $u_n$：

  $$
  \mathcal{L}_{icl}=\frac{1}{B}\sum_{i=1}^{B}\max\Big(\,\sigma(u_a^i-u_p^j)-\sigma(u_a^i-u_n^j)+\beta,\,0\Big)
  $$

  （文中以 triplet 取代 InfoNCE，理由：超大量「實例為類別」的設定下，triplet 更合適且較不易過擬合。）&#x20;

---

## 5) 總體目標函數與權重

* **總損失**：

  $$
  \mathcal{L}=\underbrace{\mathcal{L}_{cls}^{(a)}}_{\text{anchor CE}}+\lambda\,\underbrace{\mathcal{L}_{cls}^{(p)}}_{\text{positive CE}}+\gamma\,\underbrace{\mathcal{L}_{icl}}_{\text{實例級 triplet}}
  $$

  論文預設 **$\lambda=\gamma=1$**（CUB 例外），**β=1**。&#x20;

---

## 6) 訓練與推論流程

1. **資料載入** → 依上節生成 *(Ia, Ip, In)* 三元組（In 由 batch 內抽樣）。
2. **前向**：

   * Backbone（timm）抽取 *(Va,ua)*、*(Vp,up)*、*(Vn,un)*；
   * 分類頭：對 *(Ia, Ip)* 產生 logits，計 $\mathcal{L}_{cls}^{(a)},\mathcal{L}_{cls}^{(p)}$；
   * 對比頭：用 $(u_a,u_p,u_n)$ 算 $\mathcal{L}_{icl}$。
3. **反傳**：最小化 $\mathcal{L}$（AdamW、餘弦退火、warmup 皆可）。
4. **推論**：**只餵原圖**，走 backbone→全域平均→分類頭；**不產生 positive/negative**。

---

## 7) PyTorch/timm 參考樣板

> 下列程式碼為**骨幹／資料集／訓練步**的最小可運行雛形（可直接整合到你現有的 repo）。

### 7.1 Dataset（雙視角 + 拼圖遮擋）

```python
# dataset_clevit.py
import random, torch
from PIL import Image
from torchvision import transforms
import numpy as np

class PositiveShuffleMask:
    def __init__(self, mask_ratio=(0.15, 0.45), grid_s=4):
        self.ratio = mask_ratio
        self.s = grid_s

    def __call__(self, img: Image.Image):
        w, h = img.size
        # 1) 隨機遮擋一個矩形
        r = random.uniform(*self.ratio)
        mw, mh = int(w * (r ** 0.5)), int(h * (r ** 0.5))
        x0 = random.randint(0, w - mw); y0 = random.randint(0, h - mh)
        img_np = np.array(img).copy()
        img_np[y0:y0+mh, x0:x0+mw, :] = 0  # mask to 0

        # 2) s×s 區塊打亂
        s = self.s
        gh, gw = h // s, w // s
        tiles = []
        for i in range(s):
            for j in range(s):
                tiles.append(img_np[i*gh:(i+1)*gh, j*gw:(j+1)*gw, :].copy())
        random.shuffle(tiles)
        # 重組
        out = np.zeros_like(img_np)
        k = 0
        for i in range(s):
            for j in range(s):
                out[i*gh:(i+1)*gh, j*gw:(j+1)*gw, :] = tiles[k]; k += 1
        return Image.fromarray(out)

class CLEVITDataset(torch.utils.data.Dataset):
    def __init__(self, items, train=True):
        self.items = items
        self.train = train
        self.resize = (600, 600)
        self.crop_train = transforms.RandomCrop(448)
        self.crop_test = transforms.CenterCrop(448)
        self.std_aug = transforms.Compose([
            transforms.Resize(self.resize),
            self.crop_train if train else self.crop_test,
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(10),
        ])
        self.pos_aug = PositiveShuffleMask(mask_ratio=(0.15, 0.45), grid_s=4)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')

        # anchor
        ia = self.std_aug(img)
        ta = self.to_tensor(ia)

        # positive: 先做相同標準增強，再遮擋+打亂
        ip_base = self.std_aug(img)
        ip = self.pos_aug(ip_base)
        tp = self.to_tensor(ip)

        return ta, tp, label, idx  # 負樣本於 collate/batch 中產生
    def __len__(self): return len(self.items)
```

> **註**：遮擋比例與 $s=4$ 來自論文實作區段。&#x20;

### 7.2 模型（timm Swin-B）與損失

```python
# model_clevit.py
import torch, torch.nn as nn, torch.nn.functional as F
import timm

class CLEVIT(nn.Module):
    def __init__(self, num_classes, backbone='swin_base_patch4_window7_224',
                 lambda_pos=1.0, gamma_icl=1.0, margin=1.0):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='')
        # Swin 無 CLS，取最後特徵圖做 GAP
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.lambda_pos = lambda_pos
        self.gamma_icl = gamma_icl
        self.margin = margin

    def forward_backbone(self, x):
        # timm: forward_features 取得特徵，之後做 GAP
        feat = self.backbone.forward_features(x)   # [B, C, H', W'] 或 [B, N, C]
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)  # N_p 平均
        return feat  # u

    def forward(self, xa, xp, y=None):
        ua = self.forward_backbone(xa)  # anchor 全域特徵
        up = self.forward_backbone(xp)  # positive 全域特徵
        logits_a = self.classifier(ua)
        logits_p = self.classifier(up)

        out = dict(logits_a=logits_a, logits_p=logits_p, ua=ua, up=up)
        if y is not None:
            ce_a = F.cross_entropy(logits_a, y)
            ce_p = F.cross_entropy(logits_p, y)
            out['loss_sup'] = ce_a + self.lambda_pos * ce_p
        return out

def triplet_instance_loss(ua, up, un, margin=1.0):
    # L2 normalize
    ua = F.normalize(ua, dim=1)
    up = F.normalize(up, dim=1)
    un = F.normalize(un, dim=1)
    d_ap = (ua - up).pow(2).sum(1)
    d_an = (ua - un).pow(2).sum(1)
    return torch.clamp(d_ap - d_an + margin, min=0).mean()
```

> **對齊論文**：全域特徵以 patch 平均（$u=\frac{1}{N_p}\sum v_i$）與 CE 損失；對比採 **L2 normalize** 的 **triplet**，**β（margin）=1**。 &#x20;

### 7.3 訓練步驟（產生負樣本、合併總損失）

```python
# train_step.py
import torch

def train_step(model, batch, optimizer, gamma_icl=1.0, margin=1.0):
    xa, xp, y, idx = batch  # [B,3,448,448], ...
    out = model(xa, xp, y)
    loss = out['loss_sup']

    # 以 batch 內「其他 anchor」當負樣本（循環位移避免同索引）
    ua = out['ua']
    up = out['up']
    un = ua.roll(shifts=1, dims=0)  # 簡單負樣本；亦可隨機打亂索引，避免同身份
    licl = triplet_instance_loss(ua, up, un, margin=margin)

    loss = loss + gamma_icl * licl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(), 'ce': out['loss_sup'].item(), 'icl': licl.item()}
```

---

## 8) 推薦超參數（與論文對齊）

* 遮擋比例 $\alpha\in[0.15,0.45]$、**拼圖格數 $s=4$**；**β（margin）=1**；**$\lambda=\gamma=1$**（CUB 例外）；**AdamW，batch=12，lr=1e-3**；**resize 600 → crop 448**。

---

## 9) 評估與消融（建議）

* **主結果**：在 UFG 與文中各資料集上比較 CLE-ViT vs. baseline。
* **消融**：

  * 拿掉實例級對比（baseline）與改成類別級對比（Baseline+CC）之比較；
  * **λ** 權重掃描（論文在 SoyLocal 上顯示 λ=1 最佳）。&#x20;

---

## 10) 與論文關鍵點對齊一覽

* 雙視角（標準 + 遮擋＋打亂）→ backbone 抽特徵 → 分類 + 實例級對比；**推論時移除自監督分支**。
* **正樣本構造**：隨機遮擋（方塊）→ $s\times s$ 區塊隨機打亂。&#x20;
* **負樣本**：同批次其他影像。
* **全域表徵**：patch 平均；**CE** 損失。&#x20;
* **對比損失**：實例級 **triplet**（L2 normalize、β=1），採用 triplet 的動機與 InfoNCE 比較。&#x20;
* **總損失**：$\mathcal{L}=\mathcal{L}_{cls}^{(a)}+\lambda\mathcal{L}_{cls}^{(p)}+\gamma\mathcal{L}_{icl}$。
* **實作超參**：Swin-B（ImageNet-21K）、resize 600→crop 448、AdamW、batch=12、lr=1e-3、$\alpha\in[0.15,0.45], s=4$。

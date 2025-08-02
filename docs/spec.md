好的，下面給你**可直接落地的實作規格與步驟**，以及一套**完整實驗計畫**。內容以 PyTorch/CLE-ViT 為假定環境，核心模組可作為 plug-in head 加到原本 pipeline 中。

---

## A. 系統與程式架構

**目標**：在 CLE-ViT 的雙視角（anchor/positive）token 特徵上，加入

1. **EGO-式關係圖多項式融合（Graph Polynomial Fusion, GPF）**，
2. **圖加權高階統計矩池化（Moment Head）**，
   並輸出「矩 token」與 CLS token 的融合表徵做分類與對比。

**推薦專案結構**

```
project/
  configs/
    ufg_base.yaml
  data/
  src/
    models/
      cle_vit_backbone.py       # 直接沿用/封裝 CLE-ViT backbone 輸出 tokens
      gpf_kernel.py             # Graph Polynomial Fusion (EGO-like)
      moment_head.py            # Weighted MPN-COV + (opt) tensor-sketch head
      classifier_head.py        # CLS + moment concat -> MLP
    losses/
      triplet_loss.py
      kernel_alignment.py       # L_align
    train.py
    eval.py
    utils/
      ops.py                    # iSQRT-COV(牛頓-舒爾茲), half-vectorize, seed等
      viz.py                    # 相似度圖/注意力/混淆矩陣可視化
  scripts/
    train_ufg.sh
    eval_ufg.sh
  README.md
```

**環境**

* Python 3.10+ / PyTorch 2.2+ / CUDA 11.8+
* timm（若 CLE-ViT 依賴）、einops（簡化張量重排）、apex/torch.compile（選用）

---

## B. 核心模組規格（可直接實作）

### B1. Graph Polynomial Fusion（GPF, EGO-like）

**介面**

```python
class GraphPolynomialFusion(nn.Module):
    def __init__(self, degree_p=2, degree_q=2, coeff_init='uniform', 
                 sim='cosine', eps=1e-6):
        """
        degree_p, degree_q: 多項式階數上限 P, Q
        係數參數 A_{pq} 以非負參數化：A_{pq} = softplus(alpha_{pq})
        sim: 'cosine' 或 'dot'，用來計算 token 相似度
        """
    def forward(self, Za, Zp): 
        """
        Za, Zp: 兩視角 tokens, shape = [B, N, d] （來自 CLE-ViT）
        return: G (融合後關係圖), shape = [B, N, N], 保證對稱、非負，近似 PSD
        """
```

**計算步驟**

1. **標準化 tokens**（若 sim='cosine'）：`Z = Z / (||Z||+eps)`。
2. 兩視角相似度：
   `Ra = Za @ Za^T`、`Rp = Zp @ Zp^T`，shape = `[B, N, N]`。
3. 多項式融合（元素乘／Hadamard power）：

   $$
   G(A)=\sum_{p=0}^{P}\sum_{q=0}^{Q} A_{pq}\,(R_a)^{\odot p}\odot(R_p)^{\odot q}
   $$

   * 參數 $A_{pq} = \mathrm{softplus}(\alpha_{pq})$ 以確保非負。
   * 強制對稱：`G = 0.5*(G + G.transpose(-1,-2))`。
   * （選）normalize 到 $[0,1]$ 或以 `torch.clamp(G, min=0)`。
4. 輸出 `G`。理論上（Schur 乘積 + 非負和）使其近似 PSD。

**複雜度**：O(B·N²·P·Q)，建議 `N=196, P=Q=2`（9 個項）起步。

---

### B2. 圖加權高階統計矩池化（Moment Head）

**介面**

```python
class MomentHead(nn.Module):
    def __init__(self, d_in, d_out, use_third=False, 
                 isqrt_iter=5, eps=1e-5, sketch_dim=4096):
        """
        d_in: token 維度
        d_out: 輸出壓縮後維度（最終給分類器）
        use_third: 是否加入近似三階 (Tensor-Sketch)
        isqrt_iter: iSQRT-COV 牛頓-舒爾茲迭代步數
        sketch_dim: 三階近似投影維度
        """
    def forward(self, Z, G):
        """
        Z: [B, N, d] 單一視角或融合後 tokens（建議用 anchor 視角）
        G: [B, N, N] 由 GPF 產生
        return: moment_feat [B, d_out]
        """
```

**計算步驟（二階）**

1. 令 `D = diag(G @ 1)`，`W = D^{-1/2} G D^{-1/2}`（對每個 batch 逐樣本計算）。
2. 帶權均值：$\mu = (Z^\top W \mathbf{1}) / \mathrm{tr}(W)$，shape = `[B, d]`。
3. 帶權二階矩：

   $$
   M_2=(Z-\mu)^\top W (Z-\mu)\quad \in \mathbb{R}^{d\times d}
   $$
4. **iSQRT-COV 正規化**：

   * Trace normalization：`M2 = M2 / trace(M2)`
   * 牛頓-舒爾茲迭代 \~ 3–5 步得到 `M2_isqrt`
5. 將 `M2_isqrt` 做 **上三角 half-vectorize**（含對角），再經 `Linear -> BN -> GELU -> Linear` 壓到 `d_out/2`。

**（選）三階近似**

* 對 `Z - μ` 維度做 **Tensor-Sketch/Compact Bilinear**：
  `phi3 ≈ TS( (Z-μ) ⊗ (Z-μ) ⊗ (Z-μ) )`，再以 `W` 做加權平均：
  `t3 = (phi3^T @ W @ 1) / tr(W)`，接 MLP 壓到 `d_out/2`。
* 最後 concat：`moment_feat = concat(m2_feat, t3_feat)`；若不啟用三階就只用二階。

---

### B3. 分類頭與損失

**ClassifierHead**

```python
class ClassifierHead(nn.Module):
    def __init__(self, d_cls, d_moment, num_classes, p_drop=0.1):
        # concat [CLS] 與 moment_feat -> MLP -> logits
    def forward(self, cls_feat, moment_feat):
        # return logits
```

**損失組合**

* `L = CE(logits, y) + L_triplet(cls_anchor, cls_positive, margin) + λ * L_align(G, Y_batch)`
* `L_align`：minimize `- corr(G, Y)` 或 `||G - Y||_F^2`（Y 為 mini-batch 標籤關係圖：同類 1、異類 0；可標準化到 \[0,1]）。
* `λ` 建議 `0.05 ~ 0.2` 起試。

---

### B4. 與 CLE-ViT 整合（前向流程）

1. 由 CLE-ViT 取兩視角 tokens：`Za, Zp` 與各自 `CLS`。
2. `G = GPF(Za, Zp)`。
3. `moment_feat = MomentHead(Za, G)`（以 anchor tokens 為主；也可用 `avg(Za, Zp)`）。
4. `logits = ClassifierHead(CLS_anchor, moment_feat)`。
5. 計算 `CE`、`Triplet(CLS_anchor, CLS_positive)`、`L_align(G, Y)` 損失。
6. 反傳更新（確保 `A_{pq}` 綁 softplus，梯度穩定）。

---

## C. 設定與超參（建議預設）

* ViT backbone：與 CLE-ViT 相同（Base/Small 視資源）。
* Token 數：`N=196`（14×14）或 `N=256`（16×16）。
* `P=Q=2`；若資源許可可試 `P=3,Q=2`。
* `A_{pq}` 初始化：`uniform(0.0, 0.1)`，經 softplus。
* MomentHead：`d_out=1024`（二階 512 + 三階 512）。
* Triplet margin：`0.2~0.5`。
* Optimizer：`AdamW(lr=3e-4, wd=0.05)`；backbone 可分組較小 lr（如 1e-4）。
* Scheduler：`cosine`，warmup 5 epochs。
* Batch：`64`（AMP 混合精度），訓練 `100~150` epochs。
* Regularization：`dropout 0.1`、`stochastic depth` 依 backbone。
* 記憶體優化：AMP、gradient checkpoint（對 iSQRT-COV 可選用 `torch.compile`）。

---

## D. 關鍵實作細節與陷阱

* **PSD 與數值穩定**：

  * `A_{pq} ≥ 0` 用 softplus；`G = (G+G^T)/2`。
  * `D = diag(G@1)` 需 `clamp(min=eps)`，避免 `D^{-1/2}` 溢出。
* **iSQRT-COV**：

  * 先 `M2 = M2 / trace(M2)` 再迭代；迭代步數 3–5 折衷精度/速度。
  * 迭代中加入微小 `eps*I`。
* **三階近似**：

  * Tensor-Sketch 的隨機哈希/符號要固定種子，`sketch_dim` 2–8k 間選。
* **對齊正則（L\_align）**：

  * `Y` 可做 `Y = Y / ||Y||_F`，`G = G / ||G||_F` 後取 `-<G, Y>`。
* **多 GPU**：

  * `Y` 與 `G` 的 batch 內配對要注意跨卡收集（`all_gather`）才能形成完整關係圖；或改成「每卡各自 mini-batch 的對齊」。

---

## E. 訓練與推論流程（步驟）

1. **資料載入與增強**：依 CLE-ViT 生成兩視角（遮擋 + 打亂/裁切/顏色抖動）；維持與原文一致的正樣本生成策略。
2. **前向**：`Za,Zp, CLS_a, CLS_p -> GPF -> MomentHead -> Classifier -> losses`。
3. **反傳**：AMP + 梯度裁剪（如 1.0）。
4. **評估**：Top-1/Top-5、macro-avg accuracy；保存最佳 ckpt。
5. **推論**：單視角或 multi-crop（如 3-crop）；輸出 logits。

---

## F. 實驗設計（主表 + 消融 + 計算效率）

### F1. 主結果（UFG 主基準）

* **資料**：UFG 全量（\~47k 圖、3.5k 類；官方劃分）。
* **模型**：CLE-ViT（baseline） vs **CLE-ViT + GPF + Moment**（本法）。
* **指標**：Top-1、Top-5、per-class mean（防長尾偏置），標註標準差（3 次不同 seed）。
* **報表**：主表列出：

  * CLE-ViT（原始）
  * * 二階（W 加權 + iSQRT-COV）
  * * 二階 + 三階（Tensor-Sketch）
  * * 二階 + 三階 + L\_align（完整）
      另附 **FLOPs / Params / 訓練時間 / 推論延遲**。

### F2. 消融（Ablation）

1. **多項式階數**：`(P,Q) ∈ {(1,1),(2,1),(2,2),(3,2)}`。
2. **係數學習**：

   * 固定等權 vs **learnable A\_{pq}**（softplus）
   * 加 `L1` 稀疏化於 `A_{pq}`（鼓勵小而有效的項）。
3. **相似度選擇**：`cosine` vs `dot`（dot 前需 LayerNorm）。
4. **是否對稱正規化**：`W = D^{-1/2} G D^{-1/2}` vs 僅用 `G`。
5. **Moment 組件**：

   * 僅二階 vs 二階+三階；
   * iSQRT-COV 迭代步數 {3,5,7}；
   * half-vectorize vs PCA 壓縮。
6. **對齊正則 `λ`**：{0, 0.05, 0.1, 0.2}。
7. **視角選擇**：MomentHead 用 `Za` vs `avg(Za,Zp)`。
8. **Token 數**：`N ∈ {144,196,256}`（patch 大小/裁切導致）。

### F3. 效率與可擴展性

* **時間/空間**：以 batch=64、解析度 224 記錄：

  * 前向/反向時間（每 iter）
  * 額外顯存（相對 CLE-ViT）
* **O(N²) 效率技巧**：

  * 分塊計算 `Ra,Rp`；`torch.cuda.amp.autocast`；
  * `Ra.pow_(p)` 使用 fused kernel 或 `exp(p*log(Ra+eps))`（注意數值穩定）。

### F4. 魯棒性與泛化

* **遮擋比例掃描**：變動 positive 視角的遮擋/打亂強度（0.1\~0.6）。
* **少樣本**：每類 1/2/4-shot 的微調表現（從預訓練權重出發）。
* **跨資料集驗證（選）**：在 UFG 的子集合（如 Cotton80 等）測試。

### F5. 可解釋性與可視化

* `G` 的熱力圖（同/異類對比分布）、`A_{pq}` 的學習曲線。
* t-SNE/UMAP：CLS vs CLS+Moment 的嵌入分佈。
* Token 重要度：以 `W` 的度中心性或 `∑_j W_ij` 對應空間位置可視化。

### F6. 統計檢定

* 每組結果以三個 seed（e.g., 42/3407/2025）。
* 以成對 t-test 檢定本法 vs baseline（Top-1）。

---

## G. 參考指令稿（可直接改）

**configs/ufg\_base.yaml（示例）**

```yaml
dataset: UFG
image_size: 224
batch_size: 64
epochs: 120
optimizer:
  name: adamw
  lr: 3.0e-4
  weight_decay: 0.05
scheduler:
  name: cosine
  warmup_epochs: 5
model:
  backbone: cle_vit_b
  gpf:
    degree_p: 2
    degree_q: 2
    sim: cosine
  moment:
    d_out: 1024
    use_third: true
    isqrt_iter: 5
    sketch_dim: 4096
loss:
  triplet_margin: 0.3
  lambda_align: 0.1
train:
  amp: true
  grad_clip: 1.0
  seed: 42
```

**scripts/train\_ufg.sh**

```bash
python -m src.train \
  --config configs/ufg_base.yaml \
  --output runs/ufg_eGO_moment_$(date +%Y%m%d_%H%M)
```

---

## H. 驗收與風險控管清單

* [ ] 主表顯著優於 CLE-ViT baseline（≥1–2% Top-1，依資料集難度調整期望）。
* [ ] 消融顯示：`(P,Q)=(2,2)` 與 learnable `A_{pq}` 明顯優於低階/固定權。
* [ ] 計算開銷：訓練時間增加 <25%，顯存↑可控（+\~1–2GB/卡）。
* [ ] 可視化：正對在 `G` 中顯著高於負對，`A_{pq}` 收斂到稀疏有效子集。
* [ ] 代碼可重現（3 seeds 方差 <0.5% Top-1）。

---

## I. 下一步我可以幫你的事

* 出一份最小可運行（**MVP**）的 `gpf_kernel.py` 與 `moment_head.py` 參考實作；
* 幫你把 CLE-ViT 的前處理/正樣本生成整合成 **Lightning** 或 **accelerate** 訓練腳本；
* 依你的機器資源（GPU 型號/記憶體）調參，給出**最穩定的首發設定**。

需要我直接生出 `gpf_kernel.py`/`moment_head.py` 的樣板程式碼嗎？我可以用你上面的設定寫出可直接放進 repo 的版本。

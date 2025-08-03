下面把 **EGO（Learnable Expansion of Graph Operators）** 的「**可落地實作方法**」整理成可直接套到你現有 pipeline（含 CLE-ViT/ViT 類輸出特徵）上的規格、步驟與關鍵細節；並附上與原論文對應的依據與式子，確保設計與可行性一致。

---

## 0) 問題設定與符號

* 給定 $M$ 組單位層級（unit-level）特徵 $\{F^{(m)}\}_{m=1}^M$，各 $F^{(m)}\in\mathbb{R}^{N\times d_m}$；$N$ 是 clip/patch/token 數（EGO 在實驗中常用 $N=32$ 作為圖大小）。
* 以**相似度**建立每組的**關係圖** $R^{(m)}\in\mathbb{R}^{N\times N}$，論文實作使用 **cosine similarity**（亦說明距離與相似度的對偶關係）。
* 透過 **元素乘（Hadamard）** 的「圖關係擴張」形成各圖的高階版本 $ \{(R^{(m)})^{\odot p}\}_{p=0}^P$（$R^0=I$）。EGO 的融合本質等同在**逐元素相似度**上做**多線性多項式**的加權組合（式 (6)）。
* 兩圖 $a,b$ 的**可學融合算子**：

  $$
  G \;=\; \sum_{p=0}^{P}\sum_{q=0}^{Q} A_{p,q}\; (R^{(a)})^{\odot p}\;\odot\; (R^{(b)})^{\odot q}\,,
  $$

  其中 $A\in\mathbb{R}^{(P+1)\times(Q+1)}$ 為**可學權重矩陣**；也可退化為向量外積 $A=a\otimes b$ 版本（論文提供由 $a,b$ 到全可學 $A$ 的推導）。

---

## 1) 實作模組設計（PyTorch 取向）

### (A) 特徵前處理與關係圖建構 `RelationshipGraphBuilder`

**輸入**：任一特徵 $F\in\mathbb{R}^{N\times d}$，建議先 L2 normalize。
**步驟**：

1. $S = \text{cosine\_sim}(F,F)\in[-1,1]^{N\times N}$（論文用 cosine）。
2. 對角線處理：設 $R\gets S$，並令 $R_{ii}=1$，保留自連結；另備 $R^0=I$。
3. （可選）將 $S$ 線性映射到 $[0,1]$ 以利高階乘冪穩定：$R\leftarrow (S+1)/2$。

> 依據：EGO 在圖層級以**相似度**表示點對關係，並以 $R^0=I$ 作為初始階（與式 (6) 多項式中 $s^0=1$ 一致）。

### (B) 圖關係擴張 `GraphPowerExpansion`

**輸入**：$R\in[0,1]^{N\times N}$，階數上限 $P$。
**輸出**：$\mathcal{G}=\{R^{\odot 0},R^{\odot 1},\dots,R^{\odot P}\}$。
**作法**：用 `torch.pow(R, p)` 或遞推 `R_p = R_{p-1} ⊙ R`。

> 直觀等同把**逐元素相似度**提升為不同「次方關係」，與式 (6) 中的 $s^p$ 完全對應。

### (C) 可學融合算子 `EGOFusion(A)`

**輸入**：兩組擴張序列 $\mathcal{G}^{(a)},\mathcal{G}^{(b)}$、可學參數 $A\in\mathbb{R}^{(P+1)\times(Q+1)}$。
**前向**：

```text
G = 0
for p in 0..P:
  for q in 0..Q:
    G += A[p,q] * (G_a[p] ⊙ G_b[q])
```

> 公式與推導：式 (5) 與附錄 C 的展開（含 $A=a\otimes b$ 的關係），論文明確主張 **A 可設為完整可學矩陣** 以提升彈性。

### (D) 任務頭 `GraphHead`

EGO 在異常偵測實驗中：

* **分類頭**：兩層全連接 $N\!\to\! N \to 1$，中間 **ReLU**，輸出接 **Sigmoid**；並在實驗中令 **$N=32$**。
* $G$ 轉為節點級分數的方法（常見做法之一）：

  * 先做 row-sum 得到每個節點的 weighted degree 作為節點分數，或
  * 將 $G$ 攤平成向量後餵入上面的 FC。

> 論文文字直接給出頭部設計與 $N=32$ 的設定。

---

## 2) 訓練流程（含隨機取樣與損失）

### (A) 迭代中的**隨機配對融合**

每個 iteration **隨機抽兩張關係圖**（可跨模態/表徵/資料域）做一次 EGO 融合，以維持在**同質的圖空間**中融合、同時提升魯棒性與泛化。

> 這一設計是 EGO 在多表徵/多模態/多域能同時運作的關鍵之一。

### (B) 任務損失

* 以 **BCE** 做最終二分類（正常/異常等）。
* **度數變異正規化（Degree Variance Regularization，式 (7)）**：

  * 對 fused $G$ 的**行（或列）加總**得各節點 weighted degree；
  * 設定**閾值 $\alpha$** 過濾弱連結；在**異常圖**度數中取 **Top-k** 最大值；
  * 最小化「正常圖度數變異」與「異常圖 Top-k 度數變異」之差的平方，權重為 $\lambda$。
  * 直觀：鼓勵**異常節點連結更少**、正常節點度數分布更均衡（文本明說式 (7) 使異常點較少連結）；完整式子與參數說明見。

> 以上正規化與 BCE 可線性組合為總損失；論文亦給出 $\alpha,k,\lambda$ 的探討與建議區間。

---

## 3) 超參數與預設建議

* **圖大小 $N$**：先用 $N=32$（論文實驗配置；計算可控）。
* **階數 $P,Q$**：論文以 **1–10** 做 grid search；效能對 $P,Q$ 敏感，視資料複雜度與特徵品質調整。
* **$\alpha,k,\lambda$**：

  * $\alpha$：資料集依噪聲強弱而異（Ped2/Street 偏高 0.8–0.9；Avenue 較低）。
  * $k$：ShT≈11、Ped2≈17、Avenue≈15；Street 較小（=1）。
  * $\lambda$：Street 偏小（$10^{-4}$），Ped2 較大（≈1）。
* **最佳化**：Adam/AdamW 皆可；學習率依 $A$ 規模與頭部大小微調。
* **訓練回合**：30–50 epochs（論文敘述）。
* **相似度**：用 cosine；必要時將負值shifting至 $[0,1]$ 以強化高階乘冪的數值穩定。

---

## 4) 融合與複雜度

* 前向主耗時是建立高階序列（式 (3) 的擴張），理論成本與 $P{+}Q$ 成正比；元素乘高度可平行化。以 $N=32, P=Q\approx 8$ 時成本極低且可擴展。
* EGO 在多資料集上的**速度**與**表現**比較（對 MTN 等）可參考論文表格，EGO 訓練/測試時間明顯更短。

---

## 5) 可直接落地的程式結構（接口草案）

> 下述為**結構與步驟**，可直接以 PyTorch 撰寫；若需我幫你生成實作檔與測試腳本，我可以下一步把 code 產出。

1. **GraphBuilder**

   * `forward(F) -> R`：L2 norm → cosine sim → diag=1 →（可選）shift 至 \[0,1]。
2. **GraphPowerExpansion(P)**

   * `forward(R) -> [R^0, ..., R^P]`（元素乘冪）。
3. **EGOFusion(P,Q)**

   * 參數：`A`（(P+1)×(Q+1) 可學矩陣）。
   * `forward(Ga_list, Gb_list) -> G` 依式 (5) 彙總。
4. **RandomPairSampler**

   * 從 $\{R^{(m)}\}_{m=1}^M$ 隨機取兩張圖索引 $a,b$（跨模態/表徵/域），組合成一次融合。
5. **GraphHead**

   * `forward(G) -> ŷ`：兩層 FC（$N\!\to\!N$ + ReLU → $N\!\to\!1$ + Sigmoid）。
6. **Loss**

   * `bce_loss(ŷ, y)` + `lambda * degree_var_reg(G, y, alpha, k)`；
   * `degree_var_reg` 依式 (7) 實作（Row-sum→過濾 $s_{ij}\ge\alpha$ → 計算 Var，異常取 Top-k）。
7. **訓練 Loop**

   * 取 batch → 對每個樣本建 $R^{(m)}$ → 隨機抽 $a,b$ → 擴張→融合 $G$ → 頭部→損失（BCE+正規化）→ 反向傳遞更新 $A$、頭部與可學部分。

---

## 6) 與 CLE-ViT / ViT 類 backbone 的接點

* 你已有 ViT/CLE-ViT/timm backbone：輸出 token/patch 的 unit-level 特徵即是 $F^{(m)}$。
* 你可同時使用「**不同層/不同視角**」的 ViT 特徵做多表徵，或**跨模態**（如文字嵌入）做 EGO 融合，再把 $G$ 送至任務頭。
* 若做分類（非異常偵測），可拿 **節點分數的統計量** 或 **$G$ 的向量化** 接到分類頭與 cross-entropy；EGO 的融合核心與式 (5)/(6) 不變。

---

## 7) 驗證清單（快速對照論文式子/設計）

* **關係圖建構**（cosine，相似度與距離關係）：✔
* **圖關係擴張**（元素冪次，$R^0=I$）：✔（與式 (6) 的 $s^p$ 對應）
* **融合算子**（式 (5) 與可學 $A$）：✔
* **隨機雙圖融合策略**：✔
* **度數變異正規化（式 (7)）**：✔（含 $\alpha,k,\lambda$ 與直覺解釋）
* **分類頭與 $N=32$**：✔
* **複雜度與可擴展性**：✔

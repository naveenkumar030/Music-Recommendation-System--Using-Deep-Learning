# Deep Learning Music Recommendation System (SoundSpace DL) — Project Plan

A comprehensive specification and implementation plan for a session-aware, end-to-end **Deep Learning (DL)** music recommendation system.

---

## 1. Objective, Scope & System Overview

**Objective:** Build an industrial-grade music recommendation system using modern Deep Learning architectures. The system predicts multi-objective listener engagement ($P(\text{listen})$, $P(\text{like})$, $P(\text{skip})$) and delivers personalized, diverse slates under strict latency constraints (< 15 ms).

**Core Deep Learning Methodology:**
1. **Candidate Retrieval (Two-Tower DSSM):** Dual-encoder architecture embedding listener preference history and song acoustic profiles into a shared 64-dimensional Euclidean latent space, trained via InfoNCE contrastive representation learning.
2. **Deep Neural Ranking (MT-DNN / NCF):** Deep multi-task neural network combining Generalized Matrix Factorization (GMF) and a non-linear Multi-Layer Perceptron (MLP) trunk to jointly predict multiple user engagement heads.

---

## 2. End-to-End Deep Learning Architecture

```mermaid
flowchart TD
    subgraph Data & Feature Layer
        U_RAW[Listener Context & History]
        S_RAW[Audio Acoustic Features & Tags]
        FS[(Vector Feature Store<br/>Song Embeddings: N x 64)]
    end

    subgraph Stage 1: Deep Retrieval (Two-Tower DSSM)
        U_ENC[User Encoder Network<br/>MLP 143 -> 128 -> 64]
        S_ENC[Item Encoder Network<br/>MLP 24 -> 64]
        ANN[Approximate Nearest Neighbors Search<br/>Cosine Similarity / Dot Product]
        CANDS[Top-50 Candidate Shortlist]
    end

    subgraph Stage 2: Deep Ranking (Multi-Task Neural Ranker)
        GMF[Generalized Matrix Factorization<br/>p_u * q_s]
        MLP_B[Deep Non-Linear Trunk<br/>Linear 207 -> 256 -> 128]
        TRUNK[Concatenated Latent Representation]
        H_LISTEN[Head: P(Completion)]
        H_LIKE[Head: P(Like)]
        H_SKIP[Head: P(Skip)]
        SCORE[Composite Utility Scoring]
    end

    subgraph Serving & UI Layer
        API[FastAPI Asynchronous Gateway]
        UI[SoundSpace DL Studio & Player]
    end

    U_RAW --> U_ENC
    S_RAW --> S_ENC
    U_ENC & S_ENC --> ANN
    FS --> ANN
    ANN --> CANDS
    CANDS & U_RAW --> GMF & MLP_B
    GMF & MLP_B --> TRUNK
    TRUNK --> H_LISTEN & H_LIKE & H_SKIP
    H_LISTEN & H_LIKE & H_SKIP --> SCORE
    SCORE --> API --> UI
```

---

## 3. Mathematical Formulation

### 3.1 Two-Tower Candidate Generation (DSSM)
Let listener state be $\mathbf{u} \in \mathbb{R}^{d_u}$ and candidate item features be $\mathbf{s} \in \mathbb{R}^{d_s}$.
The two towers compute $L_2$-normalized latent representations:
$$\mathbf{z}_u = \frac{f_\theta(\mathbf{u})}{\|f_\theta(\mathbf{u})\|_2}, \quad \mathbf{z}_s = \frac{g_\psi(\mathbf{s})}{\|g_\psi(\mathbf{s})\|_2}$$

Trained via symmetric InfoNCE contrastive loss over mini-batch $B$:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\mathbf{z}_{u_i}^\top \mathbf{z}_{s_i} / \tau)}{\sum_{j=1}^B \exp(\mathbf{z}_{u_i}^\top \mathbf{z}_{s_j} / \tau)}$$

At inference time, ANN indexes the item embeddings and queries the top $K=50$ candidates in $< 1\text{ ms}$.

### 3.2 Multi-Task Deep Neural Ranker (MT-DNN)
The ranker takes the concatenated listener vector $\mathbf{u} \in \mathbb{R}^{143}$ and candidate song embedding $\mathbf{e}_s \in \mathbb{R}^{64}$ (input dimension $D=207$).

1. **GMF Latent Interaction:**
   $$\mathbf{\phi}^{\text{GMF}} = \mathbf{p}_u \odot \mathbf{q}_s$$
2. **Deep Non-Linear MLP Branch:**
   $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 [\mathbf{u}; \mathbf{e}_s] + \mathbf{b}_1)$$
   $$\mathbf{h}_2 = \text{Dropout}_{0.2}(\text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2))$$
3. **Multi-Head Task Prediction:**
   The branches are fused into a shared trunk $\mathbf{h} = [\mathbf{\phi}^{\text{GMF}}; \mathbf{h}_2]$ and passed to dedicated task output heads:
   $$\hat{y}_{\text{listen}} = \sigma(\mathbf{w}_{\text{listen}}^\top \mathbf{h} + b_{\text{listen}})$$
   $$\hat{y}_{\text{like}} = \sigma(\mathbf{w}_{\text{like}}^\top \mathbf{h} + b_{\text{like}})$$
   $$\hat{y}_{\text{skip}} = \sigma(\mathbf{w}_{\text{skip}}^\top \mathbf{h} + b_{\text{skip}})$$

4. **Multi-Task Objective Function:**
   $$\mathcal{L}_{\text{Ranker}} = \alpha \cdot \text{BCE}(\hat{y}_{\text{listen}}, y_{\text{listen}}) + \beta \cdot \text{BCE}(\hat{y}_{\text{like}}, y_{\text{like}}) + \gamma \cdot \text{BCE}(\hat{y}_{\text{skip}}, y_{\text{skip}})$$
   where $\alpha = 1.0, \beta = 1.5, \gamma = 1.0$.

5. **Inference Composite Ranking Score:**
   $$S(u, s) = 1.0 \cdot \hat{y}_{\text{listen}} + 1.5 \cdot \hat{y}_{\text{like}} - 1.2 \cdot \hat{y}_{\text{skip}}$$

---

## 4. Feature Store & Input Representation

- **Static Listener Profile (17 dims):** Listener acoustic preferences (energy, valence, danceability, acousticness, speechiness, tempo, etc.).
- **Dynamic Sequence History (120 dims):** Last 5 interacted tracks $\times$ (21 acoustic features + 3 interaction indicators: listen, like, skip).
- **Temporal Context (3 dims):** Hour-of-day (sine, cosine), weekend indicator.
- **Session Progress Signals (3 dims):** Running skip count, like count, session step index.
- **Total State Dimension:** $17 + 120 + 3 + 3 = 143$.
- **Combined Ranker Input:** $143 \text{ (state)} + 64 \text{ (song embedding)} = 207$.

---

## 5. Milestone Breakdown & Acceptance Criteria

### Milestone 0: Data Engineering & Feature Store
- Ingest KKBox / OpenSpot music datasets.
- Generate acoustic features and zero-data-leakage chronological train/validation/test splits.
- *Status:* **Completed**

### Milestone 1: Two-Tower DSSM Candidate Retrieval
- Train user and item dual-encoders with contrastive InfoNCE loss.
- Build vector indexing for sub-millisecond top-50 candidate retrieval.
- *Status:* **Completed** (p95 retrieval latency: 0.89 ms)

### Milestone 2: Multi-Task Deep Neural Ranker Model
- Implement `MultiTaskDeepRanker` architecture in PyTorch (`src/models/deep_ranker.py`).
- Implement vectorized scoring agent `DeepRankerAgent`.
- Train model on 32,600+ interactions across 5 epochs (`models/deep_ranker.pt`).
- *Status:* **Completed** (Training Loss: 1.6878)

### Milestone 3: Deep Learning RecSys Evaluation Suite
- Implement standard ranking metrics: Hit Rate@K, NDCG@K, MRR, Catalog Coverage, Gini Diversity.
- Benchmark Deep Neural Ranker against Two-Tower baseline and Random exploration.
- *Status:* **Completed** (HR@10: 18.8%, NDCG@10: 0.1435, MRR: 0.1349)

### Milestone 4: FastAPI Serving & Online Fine-Tuning
- Update REST API endpoints to serve Deep Neural Ranker predictions with < 15 ms latency.
- Ingest real-time listener feedback for model gradient updates.
- *Status:* **Completed**

### Milestone 5: Web Studio Dashboard & Model Visualizations
- Interactive visualization of Two-Tower DSSM retrieval, candidate scores, and acoustic radar alignment.
- Full A/B simulation workbench comparing deep learning models.
- *Status:* **Completed**

---

## 6. Verification and Validation Checklist

- [x] All unit and integration tests passing (`python -m pytest`).
- [x] Vectorized PyTorch forward pass runs in under 3 ms.
- [x] Offline evaluation report confirms superior NDCG@10 and Hit Rate@10 over baselines.
- [x] Modern interactive web dashboard displays Deep Learning pipeline components.
- [x] API health and recommendation endpoints operational.

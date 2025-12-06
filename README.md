# Intent Discovery and Taxonomy Refinement for Customer Support Chats

## 1. Approach and Reasoning

The goal is to discover **missing or split‑worthy intents** from real customer chats, and use them to refine the existing intent taxonomy (primary + secondary intents).  

Instead of asking an LLM to invent intents from scratch, the system first uses **unsupervised clustering on sentence embeddings** to find coherent conversation patterns, then uses an LLM only to label those clusters and propose human‑readable intents.  

This separation has three benefits:  
- Clusters are driven by real user behavior, not model hallucinations.  
- LLM work is cheap and controllable (one call per cluster, JSON output).  
- The same workflow scales from hundreds to millions of messages by changing only configuration values, not the design.  

---

## 2. Workflow Architecture

### 2.1 Inputs and Preprocessing

**Inputs:**  
- `intent_mapper`: primary + secondary intents and definitions (existing taxonomy).  
- `customer_messages`: list of chats (`history`, `current_human_message`).  

**Preprocessing:**  
- Flatten `history + current_human_message` per record to keep context plus the latest user question.  
- Normalize text: lowercase, remove URLs, punctuation, and extra whitespace.  

This gives a clean, comparable text representation per message while preserving enough context for clustering.

### 2.2 Embeddings and Sampling

- Use `all-mpnet-base-v2` from SentenceTransformers to embed each normalized message into a dense vector, batched for efficiency.  
- For small datasets (≤ 20k), all points are used for clustering; for larger datasets, a **random sample** of messages (e.g. 50k) is selected for clustering and structure discovery, while all messages are still embedded.  

### 2.3 Dimensionality Reduction and Clustering

**Dimensionality Reduction:**  
- **Preferred:** UMAP to 10 dimensions with cosine metric, which is effective for clustering high‑dimensional embeddings.  
- **Fallback:** TruncatedSVD with L2 normalization if UMAP is not available or too slow.  

**Clustering (on the sample):**  
- Use **MiniBatchKMeans** for scalable centroid‑based clustering.  
- Search over k in a small range (e.g. 3–20) and select the k with the best silhouette score on the reduced sample.  

### 2.4 Centroids, FAISS Assignment, and Metrics

- Compute centroids in the **original embedding space** using the sample and its cluster labels.  
- Use FAISS (inner‑product index) to assign all embeddings, including non‑sampled points, to the nearest centroid via cosine similarity. This lets the same set of centroids represent hundreds of thousands or millions of messages efficiently.  

**Metrics:**  
- Global and per‑cluster silhouette on the reduced sample, to measure cluster separation.  
- Intra‑cluster coherence: average pairwise cosine similarity within each cluster (using a subsample per cluster), to measure tightness in embedding space.  

### 2.5 TF‑IDF, Novelty, and LLM Labeling

**For each cluster:**  
- Compute top TF‑IDF terms (unigrams and bigrams) across normalized texts within the cluster.  
- Collect 2–6 representative messages for qualitative inspection.  

**Novelty vs existing taxonomy:**  
- Extract all **existing secondary intent names** from the intent mapper.  
- Mark "novel" terms as those that are not substrings of any existing secondary name.  
- Only consider clusters with at least 2 novel terms as candidates for new or refined intents.  

**LLM Labeling:**  
- Prompt Groq LLM with the top terms and example messages, asking for a single JSON object:  
  - `label` (2–5 words)  
  - `id` (snake_case)  
  - `description` (≤ 25 words)  
  - `confidence` ∈ [0, 1]  
- Parse and normalize the JSON (ensure snake_case id, clamp confidence to [0, 1]).  

### 2.6 Guardrails and Filtering

Before a cluster becomes an "intent proposal", it must pass:

**Size thresholds:**  
- Hard minimum: ignore clusters with fewer than 6 messages.  
- Proposal minimum: require size ≥ 10 to propose a new/refined intent.  

**Novelty threshold:**  
- At least 2 novel TF‑IDF terms vs existing secondary names.  

**Confidence threshold:**  
- A simple `proposal_confidence` score built from cluster silhouette and coherence must be ≥ 0.3.  

Each surviving cluster is emitted as a proposal with:  
- Cluster id, size, and share of total messages.  
- Top terms and novel terms.  
- Sample messages.  
- Silhouette, coherence, proposal_confidence.  
- LLM `label/id/description/confidence`.  
- `suggested_primary` inferred from terms (e.g. `logistics` vs `specific_product`).  

---

## 3. Findings: Proposed New / Refined Intents

All examples below come from the final run (`proposals-8.json`, 200 messages).

### 3.1 Track Order (Refine Order Status)

**Cluster 1:** size 16 (8% of messages)

**Metrics:**  
- Silhouette ≈ 0.98  
- Coherence ≈ 0.93  
- Proposal confidence ≈ 0.96  

**Content:**  
- Top terms: `track order`, `track`, `order`, `help track`  
- Example messages are all variants of "How to track order?"  

**LLM Label:**  
- `label`: `Track Order`  
- `id`: `track_order`  
- `description`: "Requesting information about the status of a shipment."  
- `confidence`: 1.0  

**Mapping and Rationale:**  
- Primary: `logistics`  
- Secondary: strong, pure refinement of existing `Order Status` intent, highlighting that "track order" is a very specific, frequent sub‑pattern that deserves an explicit label and possibly tailored flows.  

### 3.2 Anti‑dandruff Product Details (Split Product Info)

**Cluster 2:** size 22 (11%)

**Metrics:**  
- Silhouette ≈ 0.63  
- Coherence ≈ 0.61  
- Proposal confidence ≈ 0.63  

**Content:**  
- Top terms: `dandruff`, `shampoo`, `anti dandruff`, `Bare Anatomy`, `use`, `acid biotin`, `shampoo salicylic`  
- Messages ask:  
  - What is special about the anti‑dandruff shampoo  
  - Ingredients and differences between variants  
  - How to use scalp exfoliating scrub and anti‑dandruff shampoo  
  - How well it works for severe dandruff  

**LLM Label:**  
- `label`: `Ask about product details`  
- `id`: `ask_product_details`  
- `description`: "Request information about product ingredients, usage, or other details."  
- `confidence`: 0.8  

**Mapping and Rationale:**  
- Primary: `specific_product`  
- This cluster supports:  
  - Splitting generic `productinfo` into more focused secondaries such as:  
    - `product_usage` (how to use step by step)  
    - `product_details_and_ingredients`  
    - `product_effectiveness_for_dandruff`  
  - The traffic is concentrated around anti‑dandruff and scalp treatment products, suggesting that a dedicated secondary like `dandruff_treatment_info` could also be justified.  

### 3.3 Other Logistics and Product Themes

- Additional proposals from clusters 3, 4, 5, 6, 7 (sizes between 21 and 39) capture:  
  - Broader "order status + contact details + missing product" flows  
  - General hair‑care product usage/effectiveness themes  
- These clusters provide further refinement candidates, but even if only two or three are used explicitly, they demonstrate that the pipeline consistently finds **interpretable, taxonomy‑relevant structure**.  

---

## 4. Failure Cases, Limitations, and Mitigations

### 4.1 Mixed Conversations / Multi‑Intent Messages

**Problem:**  
- Many chats mix multiple intents (e.g. recommendations → order status → discount).  
- Clustering operates on flattened text, so a point can contain multiple behaviors.  

**Mitigation:**  
- Cluster purity is validated via:  
  - Top terms  
  - Example messages  
  - Silhouette and coherence scores  
- Only high‑volume, high‑quality clusters with a clear dominant theme are proposed as new/refined intents.  

### 4.2 Small Clusters and Over‑Fragmentation

**Problem:**  
- Unsupervised clustering can create many small groups that are not stable enough to be intents.  

**Mitigation:**  
- Hard minimum cluster size for consideration (≥ 6) and a higher threshold (≥ 10) for actual proposals  
- Confidence threshold based on silhouette + coherence  
- Novelty check vs existing taxonomy  

### 4.3 Language Mix (English + Hinglish + Hindi)

**Problem:**  
- Some conversations mix English with transliterated Hindi, which can reduce embedding quality slightly.  

**Mitigation:**  
- Use a strong sentence‑transformer that handles mixed language reasonably well  
- Rely on example messages in proposals to ensure a human can confirm that the semantic pattern is genuine before making taxonomy changes  

### 4.4 Ambiguous and Very Short Messages

**Problem:**  
- Messages like "Ok", "Thanks", or single product names are common but not very informative for new intents.  

**Mitigation:**  
- These tend to form low‑coherence or tiny clusters and fail the guardrails, so they are not proposed as new intents  
- Existing `basic_interactions` or similar catch‑all intents can cover them in production  

---

## 5. Guardrails, Fallback Strategies, and Advanced Thinking

### 5.1 Guardrails and Boundary Conditions

**Cluster-level guardrails:**  
- Size thresholds as described above  
- Novelty vs existing secondary names  
- Confidence threshold  

**LLM guardrails:**  
- Fixed JSON schema (label, id, description, confidence)  
- Post‑processing normalization (snake_case id, clamped confidence)  
- Human review required before updating the mapper  

These ensure that the system **suggests** intents rather than silently rewriting the taxonomy.

### 5.2 Fallbacks and Scalability

**Dimensionality Reduction Fallback:**  
- UMAP → SVD if UMAP is not available or too slow  

**Clustering Scalability:**  
- For small N: cluster on all points  
- For large N: cluster on a sample, then assign all points with FAISS; only centroids and assignments are needed, not full pairwise distances  

This allows the same pipeline to work unchanged from 200 to 2,000,000+ messages by adjusting only CONFIG values:  
- `max_points_for_full_clustering`: threshold above which sampling kicks in  
- `sample_size_for_clustering`: size of the sample for clustering  
- `embedding_batch_size`: batch size for embeddings  
- `k_min`, `k_max`: range for k-search in MiniBatchKMeans  

### 5.3 Evaluation Beyond the Assignment

The framework allows more advanced evaluation:

- **Track cluster centroids over time** to monitor intent drift (how topics move between runs)  
- **Compute acceptance rate** of proposals after human review as a quality KPI  
- **Add alternative scoring functions** (e.g. using stability or external business feedback) to rank proposed intents  
- **Monitor embedding quality** by checking average silhouette over time to detect data distribution shifts  

---

## 6. Engineering and ML Thinking

### Engineering Thinking

- **Batching:** Embeddings computed in batches for GPU efficiency  
- **ANN (FAISS):** Scales assignment to millions of messages without pairwise distance computation  
- **Sampling:** Clustering on a subset keeps computational cost linear in cluster complexity, not data size  
- **Configuration-driven:** No code rewrites; adjust CONFIG for different data sizes  

### ML Thinking

- **Sampling strategy:** Representative sampling preserves cluster structure while reducing compute  
- **Metrics:** Silhouette and coherence validate cluster quality from both geometric and semantic perspectives  
- **Stability:** Bootstrap subsampling (if enabled) can measure cluster robustness  
- **Drift detection:** Centroid comparison across runs detects when intent themes change over time  

### Out-of-the-Box Thinking

- **Hierarchical clustering refinement:** After discovering k clusters, apply hierarchical clustering within the largest cluster to find sub-intents  
- **User feedback loop:** Mark proposals as "accepted", "rejected", or "modified" and track which features (top terms, examples) correlated with acceptance  
- **Custom scoring:** Beyond silhouette + coherence, add business metrics (e.g. how many support tickets per cluster, avg resolution time) to rank proposals  
- **Multilingual representation:** Embed descriptions of existing intents and compare embedding-space distance to cluster centroids for more accurate `suggested_primary` mapping  

---

## 7. Summary

This intent discovery pipeline combines:  
- **Unsupervised learning** (clustering) to find real patterns in data  
- **Explainability** (top terms, examples, metrics) so humans can understand and review proposals  
- **Scalability** (batching, sampling, ANN) so it works from hundreds to millions of messages  
- **Safety** (guardrails, novelty checks, human review loop) so it doesn't blindly rewrite the taxonomy  

The final run (`proposals.json`) surfaces clear, high-quality intent candidates like "Track Order" and "Ask about product details", backed by metrics and examples, and organized with fallback strategies and boundary conditions. This demonstrates both systematic design and attention to practical constraints that real ML systems must respect.
## 8. How to Use This System

### For Small Datasets (200–2,000 messages)
Run as-is; no configuration changes needed
out = run_intent_discovery_pipeline()

### For Medium Datasets (2,000–20,000 messages)

Optionally reduce batch size or UMAP components:

CONFIG["embedding_batch_size"] = 32

CONFIG["umap_n_neighbors"] = 10

### For Large Datasets (20,000–2,000,000+ messages)
CONFIG["max_points_for_full_clustering"] = 20000

CONFIG["sample_size_for_clustering"] = 50000

CONFIG["embedding_model"] = "all-MiniLM-L6-v2" # lighter model if needed

CONFIG["embedding_batch_size"] = 64

All steps scale automatically; no code changes required.

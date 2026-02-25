# Multimodal GNN on DMG777K 

This project solves a complex data problem: How do you teach a computer to categorize items in a massive database when the information is a mix of text, images, and relationships?

Most AI models look at just text or just images. This pipeline uses a Multimodal Graph Neural Network to "see" and "read" node attributes (using CLIP) while simultaneously understanding how those nodes are connected (using Graph Convolutional Networks).

Standard databases (Knowledge Graphs) like **DMG777K** are messy. They contain "entities" (like a person) connected to "literals" (like a photo of that person or a written description).

My approach combines three distinct AI strengths:

 1) Computer Vision: Uses OpenAI’s CLIP model to "understand" image attributes.

 2) Natural Language Processing (NLP): Uses CLIP’s text encoder to transform descriptions into mathematical vectors.

 3) Graph Intelligence: Uses an R-GCN (Relational Graph Convolutional Network) to process the "social network" of these items—understanding that who you are connected to is just as important as what you look like.

---

## Table of Contents
- [Task Overview](#task-overview)
- [Key Idea](#key-idea)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Experiments](#experiments)
  - [Baseline: Bag-of-Relations (BoR) + R-GCN](#baseline-bag-of-relations-bor--r-gcn)
  - [Multimodal: CLIP Embeddings + R-GCN](#multimodal-clip-embeddings--r-gcn)
  - [Subset Strategy (Why it’s Needed)](#subset-strategy-why-its-needed)
- [Embedding Timing (Offline vs Online)](#embedding-timing-offline-vs-online)
- [Bonus: Joint Fine-tuning (Planned/POC)](#bonus-joint-fine-tuning-plannedpoc)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Task Overview
The assignment asks to: [file:2]
- Perform EDA on `dmg777k` (node/edge types, available node information). [file:2]
- Build a GNN for node classification. [file:2]
- Use simple baselines (e.g., BoW/BoR) and then **use Hugging Face pre-trained models** to embed multimodal node information. [file:2]
- Think about **when** to compute embeddings (offline vs online). [file:2]
- **Bonus:** jointly fine-tune the pre-trained models with the GNN. [file:2]

---

## Key Idea
DMG777K contains different node “kinds”: entities plus literal nodes such as **images** and **text strings**. [file:2]

This repo builds:
- **Structural baseline**: Bag-of-Relations features from the KG triples → R-GCN.
- **Multimodal pipeline**:
  1. Extract images/text literals from the KG.
  2. Embed them using **CLIP** (vision + text encoders from Hugging Face).
  3. Insert those embeddings into a global node feature matrix `x_multimodal`.
  4. Train an **R-GCN** for node classification.

---

## Setup
Recommended platform: **Google Colab (GPU)**. [file:2]

### Install dependencies
In Colab:

```bash 
!pip install kgbench-loader torch_geometric transformers
```

Running the Notebook
1) Load the dataset
The notebook loads DMG777K using kgbench:
```
python
import kgbench as kg
data = kg.load('dmg777k', torch=True, final=True)
```
2) Build edges
Construct edge_index and edge_type from data.triples:
```
python
num_nodes = data.num_entities
mask_valid = (data.triples[:,0] < num_nodes) & (data.triples[:,2] < num_nodes)

edge_index = torch.stack([data.triples[:,0], data.triples[:,2]], dim=0)[:, mask_valid]
edge_type  = data.triples[:,1][mask_valid]
```

3) (Optional but recommended) Cache embeddings
CLIP embedding extraction is expensive. You can save x_multimodal to Google Drive and reload after restarts.

## Experiments
### Baseline: Bag-of-Relations (BoR) + R-GCN
A fast, strong structural baseline:

Create BoR feature vector for each node by counting incoming relations.

Train an R-GCN using these features.

This baseline uses only KG structure (no images/text).

### Multimodal: CLIP Embeddings + R-GCN
Multimodal pipeline:

Use data.get_images() and data.get_strings('http://www.w3.org/2001/XMLSchema#string')

Use CLIP to embed images/text

Fill the global feature matrix x_multimodal

Train R-GCN on those features

### Subset Strategy (Why it’s Needed)
Full-batch R-GCN on the entire DMG777K graph often exceeds Colab GPU memory. [file:2]

Therefore, the notebook uses a subset/subgraph strategy:

Build a bidirectional edge list (add reverse edges).

Sample a k-hop subgraph around a set of train/test seeds.

Ensure the subgraph includes multimodal literals (non-zero CLIP fraction).

Train on the sampled subgraph.

This aligns with the assignment’s instruction to create a subset if runtime constraints occur. [file:2]

## Embedding Timing (Offline vs Online)
The assignment explicitly asks when to compute multimodal embeddings. [file:2]

Offline (used here)
Compute CLIP embeddings once.

Store them in x_multimodal.

Train the GNN using static embeddings.

Pros: fast training, less VRAM usage.
Cons: CLIP does not adapt to the graph task.

Online (bonus direction)
Keep CLIP inside the forward pass.

Backprop through CLIP + GNN jointly.

Pros: end-to-end task adaptation.
Cons: expensive (memory/time), usually needs neighbor sampling.

## Bonus: Joint Fine-tuning (Planned/POC)
The bonus task is to jointly fine-tune pre-trained models with the GNN. [file:2]

A proof-of-concept approach:

Use a small sampled subgraph.

Unfreeze only CLIP projection layers (or use LoRA adapters).

Backprop classification loss into CLIP + R-GCN.

This is computationally intensive; a few epochs on a small subgraph are acceptable as proof-of-concept. [file:2]

## Results
Results depend on:

subset size, hop count, and seed selection

class imbalance in the sampled test set

whether multimodal literal nodes appear in the sampled subgraph

Reported metrics:

Accuracy on withheld/test nodes in the sampled subgraph

Label distribution diagnostics to detect majority-class collapse

Troubleshooting
CUDA Out Of Memory (OOM)
Common on full DMG777K.

Delete CLIP model after embedding extraction (del model; torch.cuda.empty_cache()).

Use subgraph sampling instead of full-batch training. [file:2]

Reduce hidden dim, reduce num_bases, use float16 storage.

Accuracy stuck at majority baseline
If predictions collapse to one class:

Check test label distribution.

Use class-weighted loss.

Ensure reverse edges are present before k-hop sampling so literals can be reached.

Fraction of nodes with non-zero CLIP feats: 0.0
Your sampled subgraph contains no literal nodes with CLIP embeddings.

Build a bidirectional graph before k-hop sampling.

Adjust seeds/hops to include entity↔literal neighborhoods.

## Acknowledgements

DMG777K dataset via the kgbench ecosystem (Peter Bloem’s KG benchmark tooling).

CLIP model from Hugging Face Transformers.​

Author Notes: This implementation demonstrates that competitive node classification is achievable on knowledge graphs without node features by leveraging relational structure and proper feature engineering. The manual sampling approach ensures compatibility across environments without requiring compiled extensions.


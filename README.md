# VTT Innovation Resolution

Youtube Link: https://www.youtube.com/watch?v=yKNr22bu9Yc

Streamlit app link: https://test-vtt.streamlit.app

This project addresses the challenge of identifying and consolidating innovation disclosures from VTT's collaboration partnerships.

## Challenge Description

The challenge focuses on two main tasks:

1. **Innovation Resolution**: Identify when different sources are discussing the same innovation by analyzing associated organizations, descriptions, and source text.

2. **Innovation Relationship**: Create a representation that aggregates information describing innovations without losing source attribution.

## Data Overview

The dataset contains two main sources:
- Company websites mentioning VTT collaborations
- VTT's website mentioning company collaborations

The data has been preprocessed into structured graph documents containing:
- Nodes (Organizations and Innovations)
- Relationships (DEVELOPED_BY and COLLABORATION)

## Project Structure

```
Innovation-Duplication/
├── data/
│   ├── dataframes/             # Source dataframes
│   ├── entity_glossary/        # Organization name resolution data
│   ├── graph_docs/             # Original extracted relationship data
│   ├── graph_docs_names_resolved/  # Data with resolved organization names
│   ├── graph_docs_vtt_domain/  # VTT domain source data
│   ├── graph_docs_vtt_domain_names_resolved/  # VTT domain data with resolved names
│   └── keys/                   # API keys for OpenAI (needs to be obtained)
├── evaluation/                 # Evaluation files directory
│   ├── gold_entities.json      # Gold standard entities for evaluation (optional)
│   ├── gold_relations.json     # Gold standard relations for evaluation (optional)
│   ├── pred_entities.json      # Predicted entities
│   ├── pred_relations.json     # Predicted relations
│   ├── consistency_sample.csv  # Samples for manual consistency checking
│   ├── qa_examples.json        # Example QA queries and results
│   └── evaluation_results.json # Comprehensive evaluation metrics
├── results/                    # Output directory for analysis results
├── introduction_data.ipynb     # Notebook introducing the dataset
├── introduction_data.py        # Python script version of the introduction
├── local_entity_processing.py  # Data models for graph documents
├── innovation_resolution.py    # Main script for innovation resolution
├── innovation_utils.py         # Utility functions for innovation resolution
├── evaluation.py               # Evaluation module for quality assessment
├── requirements.txt            # Project dependencies (pip)
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Setup

### Using pip

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```


### Configuration file

1. Obtain API keys for OpenAI models by asking at the VTT stand
2. Place your API key configuration in `data/keys/azure_config.json` with the following structure:
   ```json
   {
   "azure-ai-search": {
      "api_key": "YOUR_SEARCH_KEY",
      "azure_endpoint": "https://your-search.search.windows.net",
      "index_name": "innovation-index"
   },
     "gpt-4o-mini": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT",
       "api_version": "API_VERSION",
       "deployment": "gpt-4o-mini",
       "eval_deployment": "gpt-4o-mini",
       "emb_deployment": "text-embedding-3-large"
     },
     "gpt-4.1-mini": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT",
       "api_version": "API_VERSION",
       "deployment": "gpt-4.1-mini",
       "eval_deployment": "gpt-4.1-mini",
       "emb_deployment": "text-embedding-3-large"
     },
     "gpt-4.1": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT", 
       "api_version": "API_VERSION",
       "deployment": "gpt-4.1",
       "eval_deployment": "gpt-4.1",
       "emb_deployment": "text-embedding-3-large"
     }
   }
   ```

## Usage

1. Run the introduction notebook to understand the data:
   ```
   jupyter notebook introduction_data.ipynb
   ```

2. Execute the innovation resolution solution:
   ```
   python innovation_resolution.py
   ```

3. Start Streamlit app locally:
   ```
   streamlit run app.py
   ```

The script will perform the following steps:
1. Load and combine data from both company websites and VTT domain
2. Initialize OpenAI client for generating embeddings
3. Resolve innovation duplicates using semantic similarity
4. Create a consolidated knowledge graph
5. Analyze the innovation network
6. Visualize the results
7. Export the results to the `results/` directory
8. Run evaluation metrics on the results

### Command Line Options

The script supports various command line options for configuring the caching system and evaluation:

```
python innovation_resolution.py [options]

Options:
  --cache-type TYPE      Cache type to use (default: embedding)
  --cache-backend TYPE   Cache backend type (json or memory, default: json)
  --cache-path PATH      Path to cache file (default: ./embedding_vectors.json)
  --no-cache             Disable caching
  --skip-eval            Skip evaluation step
  --auto-label           Automatically label consistency samples and generate gold standard files
```

Examples:
```bash
# Use default configuration (JSON file caching)
python innovation_resolution.py

# Use in-memory caching (faster but not persistent)
python innovation_resolution.py --cache-backend memory

# Disable caching (regenerate embeddings each time)
python innovation_resolution.py --no-cache

# Custom cache file location
python innovation_resolution.py --cache-path "./data/cache/embeddings.json"

# Skip evaluation metrics
python innovation_resolution.py --skip-eval

# Use automatic labeling for evaluation (no manual labeling required)
python innovation_resolution.py --auto-label
```

## Solution Details

# Azure Integration Details

This section provides detailed instructions and configurations for integrating with Azure services.

---

## Azure Services Used

### 1. Azure OpenAI

Used for:

* Embedding generation via `AzureOpenAIEmbeddings`
* Language understanding and chat responses via `AzureChatOpenAI`

#### Purpose of `AzureOpenAIEmbeddings`

The `AzureOpenAIEmbeddings` component is responsible for converting innovation-related textual descriptions into high-dimensional vector representations (dimension = 3072). These embeddings capture semantic meaning and are used to measure similarity between different innovations. The resulting vectors are stored in Azure AI Search for later retrieval and clustering. This forms the core of the semantic deduplication and search pipeline.

Deployment requirements:

* `text-embedding-3-large`&#x20;

#### Purpose of `AzureChatOpenAI`

The `AzureChatOpenAI` model is designed to answer user questions about innovation relationships. It retrieves the most relevant context from the vectorized innovation database stored in Azure AI Search and generates an intelligent, context-aware response. It uses LangChain's prompt templating and pipeline to ensure answers are grounded in retrieved information.

Deployment requirements:

* ⚠️ Resource type should be `Azure OpenAI` instead of `Azure Foundry` , in this way you get full access to OpenAI model

- `gpt-4.1-mini` or similar deployed under your Azure OpenAI resource. Change the name in **Configuration File** accordingly
- The inference task of model is `ChatCompletion`

### 2. Azure AI Search

Used for:

* Storing and querying innovation information embeddings
* Supporting similarity search and  information retrieve

Required setup:

* Create a search service instance
* Copy your API Key and Endpoint of Azure AI Search to **Configuration File**

---

## Coming Soon

* Support for Azure CosmosDB (for graph persistence)
* Automated Azure Search index provisioning
* Azure Functions / Logic Apps integration


### Innovation Resolution

The solution uses semantic similarity through embeddings to identify when different sources are discussing the same innovation:

1. For each innovation, we create a feature representation combining:
   - Innovation name
   - Innovation descriptions
   - Organizations that developed it

2. These features are converted to embeddings using OpenAI's embedding API or using TF-IDF as fallback

3. Cosine similarity is computed between all innovation pairs

4. Innovations with similarity above a threshold (default: 0.85) are considered duplicates

5. Duplicate innovations are mapped to a canonical innovation ID


### Innovation Duplicate Detection & Knowledge Graph

A lightweight pipeline to detect semantically duplicate innovations via embedding‐based clustering (HDBSCAN) and build a consolidated innovation–organization knowledge graph.

1. **HDBSCAN Overview**
HDBSCAN (Hierarchical Density‐Based Spatial Clustering of Applications with Noise)
	-	No preset k: Automatically determines cluster count.
	-	Variable‐density & arbitrary‐shape clusters: Handles small, unevenly sized duplicate groups.
	-	Noise detection: Unique (non‐duplicate) innovations remain unclustered.
	-	Membership probability: Each point obtains a “strength” score for belonging to its cluster.
	-	Why HDBSCAN works here
	-	Duplicate innovation records often form small, dense pockets in 1536-dimensional text‐embedding space.
	-	HDBSCAN isolates outliers (non-duplicates) without forcing them into a cluster.
	-	We normalize embeddings (cosine → Euclidean) and tune min_cluster_size=2 by default.

2. **Experience result**
  - Total nodes from all Grah: 4076
  - Total edges: 9287
  - Innovations: 1735(HDBSCAN) to 2000(setting needed for some clusters way)
  - Organizations: 2490

Baseline: Raw Threshold Clustering (Connected Components): Created knowledge graph with 1911 innovations, 2490 organizations, and 12502 relationships
HDBSCAN (min_cluster_size=2, cosine metric): Created knowledge graph with 1735 innovations, 2490 organizations, and 12341 relationships
K-Means (k = 1911): Created knowledge graph with 1911 innovations, 2490 organizations, and 12544 relationships
Agglomerative Hierarchical Clustering (n_clusters = 1911): Created knowledge graph with 1911 innovations, 2490 organizations, and 12544 relationships
Spectral Clustering (n_clusters = 1911, n_neighbors = 15): Created knowledge graph with 1911 innovations, 2490 organizations, and 12612 relationships

3. **API utils**
ALL the cluster we try have been packaged into API form, which includes HDBSCAN, K-Means, Agglomerative Hierarchical Clustering, Spectral Clustering, Faiss‐Based Nearest Neighbors + Graph Clustering and Pure Graph Clustering (Threshold + Connected Components / k-Core). Using Pure Graph Clustering to get the initial cluster numbers for all the cluster methods that needed to be preinputed first.
 

### Caching System

The solution uses a modular caching system for embeddings to improve performance:

1. **Architecture**:
   - Abstract `CacheBackend` protocol for different backend implementations
   - `JsonFileCache`: Persistent file-based caching (default)
   - `MemoryCache`: Fast in-memory caching
   - `EmbeddingCache`: Unified front-end with configurable backend
   - `CacheFactory`: Factory for creating cache instances

2. **Features**:
   - Automatic embedding caching to avoid redundant API calls
   - Configurable cache backend (file-based or in-memory)
   - Option to disable caching completely
   - Automatic recovery from cache loading errors

3. **Extensibility**:
   - Designed to be easily extended with new cache backends
   - Compatible with both OpenAI embeddings and TF-IDF fallback

### Knowledge Graph Consolidation

Once duplicates are identified, we consolidate the knowledge graph:

1. All information about duplicate innovations is merged into a single representation
2. The consolidated graph maintains:
   - Multiple names for the same innovation
   - Multiple descriptions
   - All organizations involved in development
   - All source documents mentioning the innovation
   - Original IDs of the duplicate innovations

### Network Analysis

The solution analyzes the consolidated innovation network to extract insights:

1. Basic statistics about innovations and organizations
2. Identification of innovations mentioned in multiple sources
3. Key organizations based on network centrality
4. Visualization of the innovation network

### Evaluation Module

The solution includes a comprehensive evaluation module that assesses the quality of the innovation resolution process through four main components:

1. **Consistency Checking**:
   - Randomly samples merged innovations for human verification
   - Generates a CSV file with innovation IDs, aliases, and source snippets
   - Allows human labelers to mark whether the merged items are truly the same innovation
   - Calculates the overall consistency rate (percentage of correctly merged innovations)
   - With `--auto-label`, automatically labels samples using LLM or heuristic methods

2. **Entity & Relation Extraction Accuracy**:
   - Compares automatically extracted entities and relations against human-annotated gold standards
   - Calculates precision, recall, and F1 score for both entity and relation extraction
   - Requires gold standard files (`gold_entities.json` and `gold_relations.json`) in the `evaluation` directory
   - With `--auto-label`, automatically generates gold standard files by sampling from predictions

3. **Knowledge Graph Structure Metrics**:
   - Calculates comprehensive graph statistics including:
     - Node and edge counts by type
     - Average degree and graph density
     - Connected component analysis
     - Connectivity ratio

4. **End-to-End QA Testing**:
   - Performs sample queries on the knowledge graph, such as:
     - "Which organizations developed innovation X?"
     - "What innovations are associated with organization Y?"
   - Saves example query results for manual inspection

### Template Files for Evaluation

The repository includes template files in the `evaluation` directory to help users get started with the evaluation process:

1. **`gold_entities_template.json`**: 
   - Example format for gold standard entity annotations
   - Copy this file to `evaluation/gold_entities.json` and expand with your own annotations

2. **`gold_relations_template.json`**: 
   - Example format for gold standard relation annotations
   - Copy this file to `evaluation/gold_relations.json` and expand with your own annotations

3. **`consistency_sample_template.csv`**: 
   - Example of the consistency checking CSV with sample labels
   - Shows how to fill in the `human_label` column with "Yes" or "No"

4. **`qa_examples_template.json`**: 
   - Example of the QA testing results format
   - Shows the expected structure of innovation-organization relationships

To use these templates:

```bash
# For entity evaluation
cp evaluation/gold_entities_template.json evaluation/gold_entities.json
# Edit evaluation/gold_entities.json with your annotations

# For relation evaluation
cp evaluation/gold_relations_template.json evaluation/gold_relations.json
# Edit evaluation/gold_relations.json with your annotations

# For consistency checking (after first run generates the sample)
# Edit evaluation/consistency_sample.csv adding Yes/No in the human_label column
# Then run the script again to calculate consistency rate
```

To use the evaluation module, you can:

1. **Prepare Gold Standards** (optional):
   - Create `evaluation/gold_entities.json` with the format: `[{"name": "...", "type": "Innovation"}, ...]`
   - Create `evaluation/gold_relations.json` with the format: `[{"innovation": "...", "organization": "...", "relation": "DEVELOPED_BY"}, ...]`

2. **Run the Evaluation**:
   - Execute `python innovation_resolution.py` to run the complete pipeline with evaluation
   - For the first run, consistency checking samples will be generated
   - Fill in the `human_label` column in the generated CSV file
   - Run the script again to calculate the consistency rate

3. **Review Results**:
   - Evaluation metrics are printed to console during execution
   - Complete evaluation results are saved to `evaluation/evaluation_results.json`
   - QA examples are saved to `evaluation/qa_examples.json`

## Results

The solution produces the following outputs:

1. In the `results/` directory:
   - `canonical_mapping.json`: Mapping from original innovation IDs to canonical IDs
   - `consolidated_graph.json`: Complete consolidated knowledge graph
   - `innovation_stats.json`: Statistics about the innovation network
   - `multi_source_innovations.json`: Details about innovations mentioned in multiple sources
   - `key_nodes.json`: Key organizations and innovations based on network analysis
   - Visualizations:
     - `innovation_network.png`: Network visualization
     - `innovation_network_3d.html`: Interactive 3D network visualization
     - `innovation_stats.png`: Summary statistics visualization
     - `top_organizations.png`: Top organizations by innovation count

2. In the `evaluation/` directory:
   - `consistency_sample.csv`: Samples for manual consistency checking
   - `evaluation_results.json`: Comprehensive evaluation metrics
   - `qa_examples.json`: Example QA queries and results
   - `pred_entities.json`: Predicted entities
   - `pred_relations.json`: Predicted relations

## Dependencies

Main dependencies include:
- pandas, numpy: Data processing
- pydantic: Data modeling
- langchain-openai, openai: API integration with OpenAI models
- networkx, matplotlib, seaborn, plotly: Visualization and network analysis
- scikit-learn: Machine learning utilities for TF-IDF fallback

See `requirements.txt` for the complete list.

## Contributors

This project is part of the AaltoAI Hackathon in collaboration with VTT.


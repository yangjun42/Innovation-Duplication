# VTT Innovation Resolution

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
├── results/                    # Output directory for analysis results
├── introduction_data.ipynb     # Notebook introducing the dataset
├── introduction_data.py        # Python script version of the introduction
├── local_entity_processing.py  # Data models for graph documents
├── innovation_resolution.py    # Main script for innovation resolution
├── innovation_utils.py         # Utility functions for innovation resolution
├── requirements.txt            # Project dependencies (pip)
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Setup

### Option 1: Using pip

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. Clone this repository
2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate vtt-innovation
   ```

### API Keys

1. Obtain API keys for OpenAI models by asking at the VTT stand
2. Place your API key configuration in `data/keys/azure_config.json` with the following structure:
   ```json
   {
     "gpt-4o-mini": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT",
       "api_version": "API_VERSION"
     },
     "gpt-4.1-mini": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT",
       "api_version": "API_VERSION"
     },
     "gpt-4.1": {
       "api_key": "YOUR_API_KEY",
       "api_base": "YOUR_AZURE_ENDPOINT", 
       "api_version": "API_VERSION"
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

The script will perform the following steps:
1. Load and combine data from both company websites and VTT domain
2. Initialize OpenAI client for generating embeddings
3. Resolve innovation duplicates using semantic similarity
4. Create a consolidated knowledge graph
5. Analyze the innovation network
6. Visualize the results
7. Export the results to the `results/` directory

## Solution Details

### Innovation Resolution

The solution uses semantic similarity through embeddings to identify when different sources are discussing the same innovation:

1. For each innovation, we create a feature representation combining:
   - Innovation name
   - Innovation description
   - Organizations that developed it

2. These features are converted to embeddings using OpenAI's embedding API

3. Cosine similarity is computed between all innovation pairs

4. Innovations with similarity above a threshold (default: 0.85) are considered duplicates

5. Duplicate innovations are mapped to a canonical innovation ID

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

## Results

The solution produces the following outputs in the `results/` directory:

1. `canonical_mapping.json`: Mapping from original innovation IDs to canonical IDs
2. `consolidated_graph.json`: Complete consolidated knowledge graph
3. `innovation_stats.json`: Statistics about the innovation network
4. `multi_source_innovations.json`: Details about innovations mentioned in multiple sources
5. `key_nodes.json`: Key organizations and innovations based on network analysis
6. Visualizations:
   - `innovation_network.png`: Network visualization
   - `innovation_stats.png`: Summary statistics visualization
   - `top_organizations.png`: Top organizations by innovation count

## Dependencies

Main dependencies include:
- pandas, numpy: Data processing
- pydantic: Data modeling
- langchain-openai, openai: API integration with OpenAI models
- networkx, matplotlib, seaborn: Visualization and network analysis
- scikit-learn: Machine learning utilities

See `requirements.txt` or `environment.yml` for the complete list.

## Contributors

This project is part of the AaltoAI Hackathon in collaboration with VTT and DataCrunch.


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
├── introduction_data.ipynb     # Notebook introducing the dataset
├── introduction_data.py        # Python script version of the introduction
├── local_entity_processing.py  # Data models for graph documents
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

## Solution Approach

The solution implements:

1. **Feature extraction and embedding** of innovations using OpenAI models
2. **Similarity-based clustering** to identify duplicate innovations
3. **Knowledge graph consolidation** to create a unified view of innovations
4. **Network analysis** to discover patterns in VTT's innovation ecosystem

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


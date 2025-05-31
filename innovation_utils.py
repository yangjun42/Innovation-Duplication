#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Innovation Resolution Challenge.

This module contains helper functions for loading data, working with
embeddings, and processing innovation and organization entities.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

# Import local modules if needed
try:
    from local_entity_processing import Node, Relationship, Document, GraphDocument
except ImportError:
    pass  # Allow the module to be imported without the local modules


def load_entity_glossary(glossary_path: str) -> Dict:
    """
    Load the entity glossary which maps entity IDs to their aliases.
    
    Args:
        glossary_path: Path to the entity glossary JSON file
        
    Returns:
        Dict: Entity glossary mapping
    """
    try:
        with open(glossary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading entity glossary: {e}")
        return {}


def get_entity_name_by_id(entity_id: str, glossary: Dict) -> Optional[str]:
    """
    Get the primary name for an entity ID using the glossary.
    
    Args:
        entity_id: Entity ID to lookup
        glossary: Entity glossary mapping
        
    Returns:
        str: Primary name for the entity or None if not found
    """
    if entity_id in glossary and 'alias' in glossary[entity_id] and glossary[entity_id]['alias']:
        # Return the first alias as the primary name
        return glossary[entity_id]['alias'][0]
    return None


def create_innovation_features(innovation_id: str, 
                              name: str, 
                              description: str, 
                              developed_by: List[str] = None) -> str:
    """
    Create a feature string for an innovation by combining its attributes.
    
    Args:
        innovation_id: ID of the innovation
        name: Name of the innovation
        description: Description of the innovation
        developed_by: List of organizations that developed the innovation
        
    Returns:
        str: Feature string for the innovation
    """
    features = f"{name}: {description}"
    
    if developed_by and len(developed_by) > 0:
        features += f" Developed by: {', '.join(developed_by)}"
    
    return features


def compute_similarity_matrix(embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute a similarity matrix for a set of embeddings.
    
    Args:
        embeddings: Dictionary mapping entity IDs to their embeddings
        
    Returns:
        pd.DataFrame: Similarity matrix
    """
    # Convert dictionary to list of tuples (id, embedding)
    embedding_items = list(embeddings.items())
    ids = [item[0] for item in embedding_items]
    embeddings_array = np.array([item[1] for item in embedding_items])
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Convert to dataframe
    return pd.DataFrame(similarity_matrix, index=ids, columns=ids)


def find_potential_duplicates(similarity_matrix: pd.DataFrame, 
                             threshold: float = 0.85) -> List[Tuple[str, str, float]]:
    """
    Find potential duplicate innovations using a similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix of innovations
        threshold: Similarity threshold for considering duplicates
        
    Returns:
        List[Tuple[str, str, float]]: List of potential duplicates with their similarity scores
    """
    potential_duplicates = []
    
    # Get upper triangle indices to avoid redundant pairs
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            similarity = similarity_matrix.iloc[i, j]
            if similarity > threshold:
                id1 = similarity_matrix.index[i]
                id2 = similarity_matrix.columns[j]
                potential_duplicates.append((id1, id2, similarity))
    
    # Sort by similarity (highest first)
    potential_duplicates.sort(key=lambda x: x[2], reverse=True)
    
    return potential_duplicates


def calculate_innovation_statistics(consolidated_graph: Dict) -> Dict:
    """
    Calculate statistics about the innovation network.
    
    Args:
        consolidated_graph: Consolidated knowledge graph
        
    Returns:
        Dict: Statistics about the innovation network
    """
    innovations = consolidated_graph['innovations']
    organizations = consolidated_graph['organizations']
    relationships = consolidated_graph['relationships']
    
    # Count innovations by data source
    source_counts = {}
    for inno_id, inno in innovations.items():
        for source in inno['data_sources']:
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
    
    # Count innovations by number of sources
    source_distribution = {}
    for inno_id, inno in innovations.items():
        num_sources = len(inno['sources'])
        if num_sources not in source_distribution:
            source_distribution[num_sources] = 0
        source_distribution[num_sources] += 1
    
    # Count innovations by number of developers
    developer_distribution = {}
    for inno_id, inno in innovations.items():
        num_developers = len(inno['developed_by'])
        if num_developers not in developer_distribution:
            developer_distribution[num_developers] = 0
        developer_distribution[num_developers] += 1
    
    # Count relationships by type
    relationship_types = {}
    for rel in relationships:
        rel_type = rel['type']
        if rel_type not in relationship_types:
            relationship_types[rel_type] = 0
        relationship_types[rel_type] += 1
    
    return {
        'total_innovations': len(innovations),
        'total_organizations': len(organizations),
        'total_relationships': len(relationships),
        'source_counts': source_counts,
        'source_distribution': source_distribution,
        'developer_distribution': developer_distribution,
        'relationship_types': relationship_types,
        'multi_source_count': sum(1 for i in innovations.values() if len(i['sources']) > 1),
        'multi_developer_count': sum(1 for i in innovations.values() if len(i['developed_by']) > 1)
    }


def find_similar_text(text1: str, text2: str) -> float:
    """
    Calculate the similarity between two text strings using a simple Jaccard similarity.
    This is a fallback method when embeddings are not available.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def merge_innovation_records(innovations: List[Dict]) -> Dict:
    """
    Merge multiple innovation records into a single consolidated record.
    
    Args:
        innovations: List of innovation records to merge
        
    Returns:
        Dict: Consolidated innovation record
    """
    if not innovations:
        return {}
    
    # Use the first innovation as the base
    consolidated = {
        'id': innovations[0]['id'],
        'names': set(),
        'descriptions': set(),
        'developed_by': set(),
        'sources': set(),
        'source_ids': set(),
        'data_sources': set()
    }
    
    # Merge all innovations
    for inno in innovations:
        consolidated['names'].update(inno.get('names', []))
        consolidated['descriptions'].update(inno.get('descriptions', []))
        consolidated['developed_by'].update(inno.get('developed_by', []))
        consolidated['sources'].update(inno.get('sources', []))
        consolidated['source_ids'].update(inno.get('source_ids', []))
        consolidated['data_sources'].update(inno.get('data_sources', []))
    
    return consolidated


def get_primary_innovation_name(innovation: Dict) -> str:
    """
    Get the primary name for an innovation from its various names.
    
    Args:
        innovation: Innovation record
        
    Returns:
        str: Primary name for the innovation
    """
    names = list(innovation['names'])
    if not names:
        return innovation['id']
    
    # Sort by length (prefer longer, more descriptive names)
    names.sort(key=len, reverse=True)
    return names[0] 
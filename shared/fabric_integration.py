"""
Neural Fabric Integration Layer
=================================

This module provides integration between the multi-scale neural fabric and
the existing SkinTwin cognitive systems (AtomSpace, PLN, MOSES, ESN).

The fabric serves as the foundational platform that embeds and connects all
cognitive components, enabling:
- Knowledge representation in AtomSpace mapped to fabric embeddings
- PLN reasoning operating over fabric-embedded concepts
- MOSES pattern discovery guided by fabric structure
- ESN temporal predictions anchored in fabric state
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .neural_fabric import NeuralFabric, SkinScale


class FabricIntegrationLayer:
    """
    Integration layer connecting the neural fabric with cognitive systems.
    """
    
    def __init__(self, fabric: NeuralFabric):
        """
        Initialize the integration layer.
        
        Args:
            fabric: The neural fabric instance to integrate with
        """
        self.fabric = fabric
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize integration configurations for each cognitive system."""
        
        # AtomSpace integration: map Atoms to fabric embeddings
        self.fabric.integrate_cognitive_system('atomspace', {
            'type': 'knowledge_graph',
            'embedding_scale': SkinScale.TISSUE,  # Primary scale for knowledge
            'concept_mapping': {},  # Maps concept names to component IDs
            'relation_encoding': 'cross_scale'  # How relations are encoded
        })
        
        # PLN (Probabilistic Logic Networks) integration
        self.fabric.integrate_cognitive_system('pln', {
            'type': 'reasoning_engine',
            'truth_value_encoding': 'activation',  # Truth values as activations
            'inference_scale': SkinScale.TISSUE,
            'confidence_threshold': 0.7
        })
        
        # MOSES (Pattern Mining) integration
        self.fabric.integrate_cognitive_system('moses', {
            'type': 'pattern_discovery',
            'search_scales': [SkinScale.CELLULAR, SkinScale.TISSUE],
            'fitness_metric': 'embedding_coherence',
            'population_encoding': 'multi_scale'
        })
        
        # ESN (Echo State Networks) integration
        self.fabric.integrate_cognitive_system('esn', {
            'type': 'temporal_prediction',
            'temporal_scale': SkinScale.SYSTEM,
            'reservoir_dimension': self.fabric.embedding_dimension,
            'readout_scales': SkinScale.ALL_SCALES
        })
        
        # ECAN (Economic Attention Networks) integration
        self.fabric.integrate_cognitive_system('ecan', {
            'type': 'attention_allocation',
            'attention_currency': 'activation',
            'allocation_scales': SkinScale.ALL_SCALES
        })
    
    def atomspace_to_fabric(self, concept_name: str, 
                           concept_features: Optional[np.ndarray] = None,
                           scale: str = SkinScale.TISSUE) -> int:
        """
        Map an AtomSpace concept to a fabric component.
        
        Args:
            concept_name: Name of the concept
            concept_features: Optional feature vector for the concept
            scale: Scale at which to embed the concept
            
        Returns:
            Component ID in the fabric
        """
        config = self.fabric.cognitive_integrations['atomspace']
        
        # Check if concept already mapped
        if concept_name in config['concept_mapping']:
            return config['concept_mapping'][concept_name]
        
        # Allocate new component
        scale_emb = self.fabric.scale_embeddings[scale]
        
        # Find unused component or use next available
        existing_ids = set(config['concept_mapping'].values())
        component_id = 0
        while component_id in existing_ids and component_id < scale_emb.num_components:
            component_id += 1
        
        if component_id >= scale_emb.num_components:
            raise ValueError(f"No available components at scale {scale}")
        
        # Store mapping
        config['concept_mapping'][concept_name] = component_id
        
        # Set metadata
        scale_emb.set_metadata(component_id, {
            'concept_name': concept_name,
            'source': 'atomspace',
            'type': 'knowledge_concept'
        })
        
        # Initialize embedding with features if provided
        if concept_features is not None:
            if len(concept_features) == self.fabric.embedding_dimension:
                scale_emb.embeddings[component_id] = concept_features
        
        return component_id
    
    def fabric_to_atomspace(self, component_id: int, 
                           scale: str = SkinScale.TISSUE) -> Dict[str, Any]:
        """
        Extract AtomSpace-compatible information from a fabric component.
        
        Args:
            component_id: Fabric component ID
            scale: Scale of the component
            
        Returns:
            Dictionary with concept information
        """
        embedding = self.fabric.get_embedding(scale, component_id)
        metadata = self.fabric.scale_embeddings[scale].get_metadata(component_id)
        
        # Find related concepts through similarity
        similar_components = self.fabric.query_fabric(embedding, scale, top_k=5)
        
        return {
            'embedding': embedding,
            'metadata': metadata,
            'similar_concepts': [
                {
                    'component_id': comp_id,
                    'similarity': sim,
                    'metadata': self.fabric.scale_embeddings[scale].get_metadata(comp_id)
                }
                for comp_id, sim in similar_components if comp_id != component_id
            ]
        }
    
    def pln_inference_on_fabric(self, premise_concepts: List[str],
                               conclusion_concept: str,
                               scale: str = SkinScale.TISSUE) -> Dict[str, float]:
        """
        Perform PLN-style inference using fabric embeddings.
        
        Args:
            premise_concepts: List of premise concept names
            conclusion_concept: Conclusion concept name
            scale: Scale for inference
            
        Returns:
            Dictionary with truth value and confidence
        """
        config = self.fabric.cognitive_integrations['pln']
        concept_mapping = self.fabric.cognitive_integrations['atomspace']['concept_mapping']
        
        # Get embeddings for premises
        premise_embeddings = []
        for concept in premise_concepts:
            if concept in concept_mapping:
                comp_id = concept_mapping[concept]
                premise_embeddings.append(self.fabric.get_embedding(scale, comp_id))
        
        if not premise_embeddings:
            return {'truth_value': 0.0, 'confidence': 0.0}
        
        # Combine premises (simple average)
        combined_premise = np.mean(premise_embeddings, axis=0)
        
        # Get conclusion embedding
        if conclusion_concept in concept_mapping:
            conclusion_id = concept_mapping[conclusion_concept]
            conclusion_emb = self.fabric.get_embedding(scale, conclusion_id)
        else:
            return {'truth_value': 0.0, 'confidence': 0.0}
        
        # Compute similarity as proxy for truth value
        norm_premise = np.linalg.norm(combined_premise)
        norm_conclusion = np.linalg.norm(conclusion_emb)
        
        if norm_premise > 0 and norm_conclusion > 0:
            truth_value = float(np.dot(combined_premise, conclusion_emb) / 
                              (norm_premise * norm_conclusion))
            # Normalize to [0, 1]
            truth_value = (truth_value + 1) / 2
        else:
            truth_value = 0.0
        
        # Confidence based on number of premises and their consistency
        premise_consistency = 0.0
        if len(premise_embeddings) > 1:
            for i in range(len(premise_embeddings)):
                for j in range(i + 1, len(premise_embeddings)):
                    norm_i = np.linalg.norm(premise_embeddings[i])
                    norm_j = np.linalg.norm(premise_embeddings[j])
                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(premise_embeddings[i], premise_embeddings[j]) / \
                              (norm_i * norm_j)
                        premise_consistency += (sim + 1) / 2
            
            premise_consistency /= (len(premise_embeddings) * (len(premise_embeddings) - 1) / 2)
        else:
            premise_consistency = 1.0
        
        confidence = min(1.0, premise_consistency * (len(premise_embeddings) / 10))
        
        return {
            'truth_value': float(truth_value),
            'confidence': float(confidence)
        }
    
    def moses_search_on_fabric(self, target_pattern: np.ndarray,
                              search_scale: str = SkinScale.CELLULAR,
                              population_size: int = 10) -> List[Tuple[int, float]]:
        """
        Perform MOSES-style evolutionary search on the fabric.
        
        Args:
            target_pattern: Target pattern to match
            search_scale: Scale to search at
            population_size: Number of candidates to return
            
        Returns:
            List of (component_id, fitness_score) tuples
        """
        # Use fabric query to find components matching the pattern
        results = self.fabric.query_fabric(target_pattern, search_scale, 
                                          top_k=population_size)
        
        # Fitness is similarity score
        return results
    
    def esn_predict_on_fabric(self, history_states: List[Dict[str, Any]],
                             prediction_horizon: int = 1) -> Dict[str, Any]:
        """
        Use ESN-style temporal prediction anchored in fabric state.
        
        Args:
            history_states: List of historical fabric states
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            Predicted future state
        """
        config = self.fabric.cognitive_integrations['esn']
        
        if not history_states:
            return {'prediction': None, 'confidence': 0.0}
        
        # Extract temporal patterns from history
        # Simplified: predict next state as weighted average of recent states
        # In a full implementation, this would use ESN reservoir dynamics
        
        weights = np.exp(-np.arange(len(history_states))[::-1] / 3.0)
        weights /= weights.sum()
        
        # Get embeddings from historical states
        scale = config['temporal_scale']
        
        prediction = {
            'predicted_activations': {},
            'confidence': 0.7,  # Placeholder
            'horizon': prediction_horizon
        }
        
        return prediction
    
    def ecan_allocate_attention(self, current_activations: Dict[str, np.ndarray],
                               budget: float = 1.0) -> Dict[str, Dict[int, float]]:
        """
        Allocate attention across fabric components using ECAN principles.
        
        Args:
            current_activations: Current activation levels at each scale
            budget: Total attention budget to allocate
            
        Returns:
            Attention allocation per scale and component
        """
        config = self.fabric.cognitive_integrations['ecan']
        
        allocation = {}
        total_activation = 0.0
        
        # Compute total activation across all scales
        for scale in SkinScale.ALL_SCALES:
            if scale in current_activations:
                total_activation += np.sum(np.abs(current_activations[scale]))
        
        if total_activation == 0:
            # Uniform allocation if no activations
            for scale in SkinScale.ALL_SCALES:
                scale_emb = self.fabric.scale_embeddings[scale]
                uniform_attention = budget / sum(
                    emb.num_components for emb in self.fabric.scale_embeddings.values()
                )
                allocation[scale] = {
                    i: uniform_attention 
                    for i in range(scale_emb.num_components)
                }
        else:
            # Proportional allocation based on activation
            for scale in SkinScale.ALL_SCALES:
                if scale in current_activations:
                    activations = current_activations[scale]
                    scale_budget = budget * (np.sum(np.abs(activations)) / total_activation)
                    
                    # Normalize activations to allocate scale budget
                    abs_activations = np.abs(activations)
                    if abs_activations.sum() > 0:
                        normalized = abs_activations / abs_activations.sum()
                        allocation[scale] = {
                            i: float(scale_budget * normalized[i])
                            for i in range(len(activations))
                        }
        
        return allocation
    
    def cross_system_reasoning(self, query: str, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-system reasoning using multiple cognitive components.
        
        This demonstrates how the fabric enables integration of AtomSpace,
        PLN, MOSES, and ESN working together.
        
        Args:
            query: Query or problem to reason about
            context: Additional context information
            
        Returns:
            Integrated reasoning results
        """
        results = {
            'query': query,
            'fabric_state': self.fabric.get_fabric_state(),
            'reasoning_steps': []
        }
        
        # Step 1: Map query to fabric representation (via AtomSpace)
        # This is simplified - real implementation would parse query
        if 'concept' in context:
            concept = context['concept']
            try:
                comp_id = self.atomspace_to_fabric(concept)
                embedding = self.fabric.get_embedding(SkinScale.TISSUE, comp_id)
                
                results['reasoning_steps'].append({
                    'step': 'atomspace_mapping',
                    'concept': concept,
                    'component_id': comp_id
                })
                
                # Step 2: Use MOSES to find related patterns
                related = self.moses_search_on_fabric(embedding, 
                                                     search_scale=SkinScale.TISSUE,
                                                     population_size=5)
                
                results['reasoning_steps'].append({
                    'step': 'moses_pattern_search',
                    'related_components': related
                })
                
                # Step 3: Use PLN for inference if premises provided
                if 'premises' in context and 'conclusion' in context:
                    inference = self.pln_inference_on_fabric(
                        context['premises'],
                        context['conclusion']
                    )
                    results['reasoning_steps'].append({
                        'step': 'pln_inference',
                        'inference_result': inference
                    })
                
            except Exception as e:
                results['error'] = str(e)
        
        return results


def create_integrated_platform(embedding_dimension: int = 128) -> Tuple[NeuralFabric, FabricIntegrationLayer]:
    """
    Create a complete integrated cognitive platform with neural fabric.
    
    Args:
        embedding_dimension: Dimensionality of embeddings
        
    Returns:
        Tuple of (NeuralFabric, FabricIntegrationLayer)
    """
    from .neural_fabric import create_default_fabric
    
    fabric = create_default_fabric(embedding_dimension)
    integration = FabricIntegrationLayer(fabric)
    
    return fabric, integration

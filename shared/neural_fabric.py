"""
Multi-Scale Skin Neural Fabric
================================

A learnable neural fabric of tensor embeddings that represents the multi-scale
model of the skin. This forms the foundational platform within which all other
cognitive systems (AtomSpace, PLN, MOSES, ESN) are embedded.

The neural fabric represents skin at multiple scales:
- Cellular level (microscopic)
- Tissue level (dermal layers)
- Organ level (skin regions)
- System level (whole body skin)

Each scale is represented as learnable tensor embeddings that can capture
hierarchical relationships and enable cross-scale reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class SkinScale:
    """Represents a specific scale in the multi-scale skin model."""
    
    CELLULAR = "cellular"      # Microscopic: cells, proteins, molecular
    TISSUE = "tissue"          # Dermal layers: epidermis, dermis, hypodermis
    REGION = "region"          # Body regions: face, arms, torso, etc.
    SYSTEM = "system"          # Whole body integration
    
    ALL_SCALES = [CELLULAR, TISSUE, REGION, SYSTEM]


class TensorEmbedding:
    """
    A learnable tensor embedding that represents features at a specific scale.
    
    Attributes:
        scale: The scale this embedding represents
        dimension: Dimensionality of the embedding space
        embeddings: The actual tensor values (numpy array)
        metadata: Additional information about this embedding
    """
    
    def __init__(self, scale: str, dimension: int, num_components: int, 
                 learning_rate: float = 0.01):
        """
        Initialize a tensor embedding.
        
        Args:
            scale: One of SkinScale constants
            dimension: Dimensionality of embedding vectors
            num_components: Number of distinct components at this scale
            learning_rate: Learning rate for gradient-based updates
        """
        self.scale = scale
        self.dimension = dimension
        self.num_components = num_components
        self.learning_rate = learning_rate
        
        # Initialize embeddings with small random values
        self.embeddings = np.random.randn(num_components, dimension) * 0.01
        
        # Gradient accumulator for learning
        self.gradients = np.zeros_like(self.embeddings)
        
        # Metadata for each component
        self.metadata: Dict[int, Dict[str, Any]] = {}
        
    def get_embedding(self, component_id: int) -> np.ndarray:
        """Retrieve embedding vector for a specific component."""
        if 0 <= component_id < self.num_components:
            return self.embeddings[component_id].copy()
        raise ValueError(f"Component ID {component_id} out of range")
    
    def update_embedding(self, component_id: int, gradient: np.ndarray):
        """
        Update embedding using gradient descent.
        
        Args:
            component_id: The component to update
            gradient: Gradient vector for the update
        """
        if 0 <= component_id < self.num_components:
            self.embeddings[component_id] -= self.learning_rate * gradient
            
    def set_metadata(self, component_id: int, metadata: Dict[str, Any]):
        """Set metadata for a specific component."""
        self.metadata[component_id] = metadata
        
    def get_metadata(self, component_id: int) -> Dict[str, Any]:
        """Get metadata for a specific component."""
        return self.metadata.get(component_id, {})
    
    def compute_similarity(self, component_a: int, component_b: int) -> float:
        """Compute cosine similarity between two components."""
        emb_a = self.embeddings[component_a]
        emb_b = self.embeddings[component_b]
        
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


class NeuralFabric:
    """
    The multi-scale neural fabric that serves as the foundational platform
    for the SkinTwin cognitive architecture.
    
    This fabric maintains learnable tensor embeddings at multiple scales and
    provides mechanisms for:
    - Cross-scale information flow
    - Hierarchical representation
    - Integration with cognitive systems (AtomSpace, PLN, MOSES, ESN)
    """
    
    def __init__(self, embedding_dimension: int = 128):
        """
        Initialize the neural fabric.
        
        Args:
            embedding_dimension: Dimensionality of embedding vectors across all scales
        """
        self.embedding_dimension = embedding_dimension
        
        # Initialize embeddings at each scale with appropriate component counts
        self.scale_embeddings: Dict[str, TensorEmbedding] = {
            SkinScale.CELLULAR: TensorEmbedding(
                SkinScale.CELLULAR, embedding_dimension, num_components=1000
            ),
            SkinScale.TISSUE: TensorEmbedding(
                SkinScale.TISSUE, embedding_dimension, num_components=50
            ),
            SkinScale.REGION: TensorEmbedding(
                SkinScale.REGION, embedding_dimension, num_components=20
            ),
            SkinScale.SYSTEM: TensorEmbedding(
                SkinScale.SYSTEM, embedding_dimension, num_components=5
            ),
        }
        
        # Cross-scale transformation matrices (learnable)
        self.cross_scale_transforms: Dict[Tuple[str, str], np.ndarray] = {}
        self._initialize_cross_scale_transforms()
        
        # Integration points for cognitive systems
        self.cognitive_integrations: Dict[str, Any] = {}
        
    def _initialize_cross_scale_transforms(self):
        """Initialize learnable transformation matrices between scales."""
        scales = SkinScale.ALL_SCALES
        
        for i, scale_from in enumerate(scales):
            for scale_to in scales[i+1:]:
                # Create bidirectional transforms
                key_up = (scale_from, scale_to)
                key_down = (scale_to, scale_from)
                
                # Initialize with identity + small noise
                transform = np.eye(self.embedding_dimension) + \
                           np.random.randn(self.embedding_dimension, 
                                         self.embedding_dimension) * 0.01
                
                self.cross_scale_transforms[key_up] = transform.copy()
                self.cross_scale_transforms[key_down] = transform.T.copy()
    
    def get_embedding(self, scale: str, component_id: int) -> np.ndarray:
        """
        Get embedding for a specific component at a given scale.
        
        Args:
            scale: Scale identifier (from SkinScale)
            component_id: Component identifier at that scale
            
        Returns:
            Embedding vector
        """
        if scale not in self.scale_embeddings:
            raise ValueError(f"Unknown scale: {scale}")
        
        return self.scale_embeddings[scale].get_embedding(component_id)
    
    def transform_across_scales(self, embedding: np.ndarray, 
                               from_scale: str, to_scale: str) -> np.ndarray:
        """
        Transform an embedding from one scale to another.
        
        Args:
            embedding: Source embedding vector
            from_scale: Source scale
            to_scale: Target scale
            
        Returns:
            Transformed embedding vector
        """
        if from_scale == to_scale:
            return embedding.copy()
        
        key = (from_scale, to_scale)
        if key in self.cross_scale_transforms:
            return self.cross_scale_transforms[key] @ embedding
        
        # If direct transform doesn't exist, use intermediate scales
        # For now, use composition through adjacent scales
        scales = SkinScale.ALL_SCALES
        from_idx = scales.index(from_scale)
        to_idx = scales.index(to_scale)
        
        result = embedding.copy()
        step = 1 if to_idx > from_idx else -1
        
        for i in range(from_idx, to_idx, step):
            current_scale = scales[i]
            next_scale = scales[i + step]
            key = (current_scale, next_scale)
            
            if key in self.cross_scale_transforms:
                result = self.cross_scale_transforms[key] @ result
        
        return result
    
    def integrate_cognitive_system(self, system_name: str, 
                                   integration_config: Dict[str, Any]):
        """
        Register a cognitive system (AtomSpace, PLN, MOSES, ESN) with the fabric.
        
        Args:
            system_name: Name of the cognitive system
            integration_config: Configuration for how the system integrates
        """
        self.cognitive_integrations[system_name] = integration_config
        
    def query_fabric(self, query_embedding: np.ndarray, scale: str, 
                    top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Query the fabric for similar components at a given scale.
        
        Args:
            query_embedding: Query embedding vector
            scale: Scale to query
            top_k: Number of top matches to return
            
        Returns:
            List of (component_id, similarity_score) tuples
        """
        if scale not in self.scale_embeddings:
            raise ValueError(f"Unknown scale: {scale}")
        
        scale_emb = self.scale_embeddings[scale]
        similarities = []
        
        # Compute similarity with all components at this scale
        for comp_id in range(scale_emb.num_components):
            comp_emb = scale_emb.embeddings[comp_id]
            
            # Cosine similarity
            norm_query = np.linalg.norm(query_embedding)
            norm_comp = np.linalg.norm(comp_emb)
            
            if norm_query > 0 and norm_comp > 0:
                sim = float(np.dot(query_embedding, comp_emb) / 
                          (norm_query * norm_comp))
                similarities.append((comp_id, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def propagate_signal(self, source_scale: str, source_component: int,
                        target_scale: str, signal_strength: float = 1.0) -> Dict[int, float]:
        """
        Propagate a signal from one scale to another, computing activation
        at the target scale.
        
        Args:
            source_scale: Scale of the source
            source_component: Component ID at source scale
            target_scale: Scale to propagate to
            signal_strength: Strength of the signal
            
        Returns:
            Dictionary mapping target component IDs to activation levels
        """
        # Get source embedding
        source_emb = self.get_embedding(source_scale, source_component)
        
        # Transform to target scale
        target_emb = self.transform_across_scales(source_emb, source_scale, target_scale)
        
        # Compute activations at target scale
        activations = {}
        target_scale_emb = self.scale_embeddings[target_scale]
        
        for comp_id in range(target_scale_emb.num_components):
            comp_emb = target_scale_emb.embeddings[comp_id]
            
            # Compute activation as scaled dot product
            norm_target = np.linalg.norm(target_emb)
            norm_comp = np.linalg.norm(comp_emb)
            
            if norm_target > 0 and norm_comp > 0:
                activation = signal_strength * np.dot(target_emb, comp_emb) / \
                           (norm_target * norm_comp)
                activations[comp_id] = float(activation)
        
        return activations
    
    def update_from_observation(self, scale: str, component_id: int,
                               observation: np.ndarray, learning_rate: float = 0.01):
        """
        Update fabric embeddings based on new observations.
        
        Args:
            scale: Scale of the observation
            component_id: Component being observed
            observation: Observation vector
            learning_rate: Learning rate for update
        """
        if scale not in self.scale_embeddings:
            raise ValueError(f"Unknown scale: {scale}")
        
        scale_emb = self.scale_embeddings[scale]
        current_emb = scale_emb.get_embedding(component_id)
        
        # Compute gradient (simple: direction towards observation)
        gradient = observation - current_emb
        
        # Update embedding
        scale_emb.update_embedding(component_id, -learning_rate * gradient)
    
    def save_fabric(self, filepath: str):
        """Save the neural fabric to disk."""
        fabric_data = {
            'embedding_dimension': self.embedding_dimension,
            'scale_embeddings': {},
            'cross_scale_transforms': {},
            'cognitive_integrations': self.cognitive_integrations
        }
        
        # Save embeddings
        for scale, emb in self.scale_embeddings.items():
            fabric_data['scale_embeddings'][scale] = {
                'embeddings': emb.embeddings.tolist(),
                'metadata': emb.metadata
            }
        
        # Save transforms
        for key, transform in self.cross_scale_transforms.items():
            fabric_data['cross_scale_transforms'][f"{key[0]}->{key[1]}"] = \
                transform.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(fabric_data, f, indent=2)
    
    def load_fabric(self, filepath: str):
        """Load the neural fabric from disk."""
        with open(filepath, 'r') as f:
            fabric_data = json.load(f)
        
        self.embedding_dimension = fabric_data['embedding_dimension']
        
        # Load embeddings
        for scale, data in fabric_data['scale_embeddings'].items():
            emb_array = np.array(data['embeddings'])
            num_components, dimension = emb_array.shape
            
            self.scale_embeddings[scale] = TensorEmbedding(
                scale, dimension, num_components
            )
            self.scale_embeddings[scale].embeddings = emb_array
            self.scale_embeddings[scale].metadata = data['metadata']
        
        # Load transforms
        for key_str, transform_list in fabric_data['cross_scale_transforms'].items():
            from_scale, to_scale = key_str.split('->')
            self.cross_scale_transforms[(from_scale, to_scale)] = \
                np.array(transform_list)
        
        self.cognitive_integrations = fabric_data['cognitive_integrations']
    
    def get_fabric_state(self) -> Dict[str, Any]:
        """
        Get current state of the neural fabric for monitoring/debugging.
        
        Returns:
            Dictionary with fabric statistics and state information
        """
        state = {
            'embedding_dimension': self.embedding_dimension,
            'scales': {},
            'cognitive_systems': list(self.cognitive_integrations.keys())
        }
        
        for scale, emb in self.scale_embeddings.items():
            state['scales'][scale] = {
                'num_components': emb.num_components,
                'embedding_norm_mean': float(np.mean(np.linalg.norm(
                    emb.embeddings, axis=1
                ))),
                'embedding_norm_std': float(np.std(np.linalg.norm(
                    emb.embeddings, axis=1
                )))
            }
        
        return state


def create_default_fabric(embedding_dimension: int = 128) -> NeuralFabric:
    """
    Create a neural fabric with default configuration.
    
    Args:
        embedding_dimension: Dimensionality of embeddings
        
    Returns:
        Initialized NeuralFabric instance
    """
    fabric = NeuralFabric(embedding_dimension)
    
    # Initialize with basic skin component metadata
    _initialize_default_metadata(fabric)
    
    return fabric


def _initialize_default_metadata(fabric: NeuralFabric):
    """Initialize fabric with basic dermatological metadata."""
    
    # Cellular scale metadata (example components)
    cellular_components = [
        "keratinocyte", "melanocyte", "langerhans_cell", "merkel_cell",
        "fibroblast", "collagen", "elastin", "sebaceous_gland"
    ]
    for i, component in enumerate(cellular_components[:8]):
        if i < fabric.scale_embeddings[SkinScale.CELLULAR].num_components:
            fabric.scale_embeddings[SkinScale.CELLULAR].set_metadata(i, {
                'name': component,
                'type': 'cell' if 'cell' in component or 'cyte' in component else 'structure'
            })
    
    # Tissue scale metadata
    tissue_components = [
        "stratum_corneum", "stratum_lucidum", "stratum_granulosum",
        "stratum_spinosum", "stratum_basale", "papillary_dermis",
        "reticular_dermis", "hypodermis"
    ]
    for i, component in enumerate(tissue_components):
        if i < fabric.scale_embeddings[SkinScale.TISSUE].num_components:
            fabric.scale_embeddings[SkinScale.TISSUE].set_metadata(i, {
                'name': component,
                'layer': 'epidermis' if 'stratum' in component else 'dermis'
            })
    
    # Region scale metadata
    region_components = [
        "face", "scalp", "neck", "chest", "back", "arms", "hands",
        "abdomen", "legs", "feet"
    ]
    for i, component in enumerate(region_components):
        if i < fabric.scale_embeddings[SkinScale.REGION].num_components:
            fabric.scale_embeddings[SkinScale.REGION].set_metadata(i, {
                'name': component,
                'body_part': component
            })
    
    # System scale metadata
    system_components = [
        "barrier_function", "immune_response", "thermal_regulation",
        "sensory_perception", "vitamin_synthesis"
    ]
    for i, component in enumerate(system_components):
        if i < fabric.scale_embeddings[SkinScale.SYSTEM].num_components:
            fabric.scale_embeddings[SkinScale.SYSTEM].set_metadata(i, {
                'name': component,
                'function': component
            })

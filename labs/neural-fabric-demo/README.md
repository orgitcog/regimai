# Multi-Scale Neural Fabric for SkinTwin

The multi-scale neural fabric is a learnable platform of tensor embeddings that forms the foundational layer of the SkinTwin cognitive architecture. All cognitive systems (AtomSpace, PLN, MOSES, ESN, ECAN) are embedded within this fabric.

## Overview

The neural fabric provides:

- **Multi-Scale Representation**: Hierarchical organization from cellular to system level
- **Learnable Embeddings**: Tensor embeddings that adapt through learning
- **Cross-Scale Information Flow**: Seamless propagation between scales
- **Cognitive System Integration**: Platform for all SkinTwin cognitive components
- **Persistent State**: Save and load learned representations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Multi-Scale Neural Fabric                      │
│                                                                  │
│  Scale Hierarchy:                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Cellular │ → │  Tissue  │ → │  Region  │ → │  System  │    │
│  │ (micro)  │   │ (layers) │   │ (body)   │   │ (whole)  │    │
│  │ 1000     │   │   50     │   │   20     │   │    5     │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                                  │
│  Each component: 128-dimensional learnable tensor embedding     │
│  Cross-scale: Learnable transformation matrices                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │  Cognitive Systems Embedded in Fabric: │
         │  • AtomSpace - Knowledge graphs        │
         │  • PLN - Probabilistic reasoning       │
         │  • MOSES - Pattern mining              │
         │  • ESN - Temporal prediction           │
         │  • ECAN - Attention allocation         │
         └────────────────────────────────────────┘
```

## Four Scales of Skin Representation

### 1. Cellular Scale (Microscopic)
- **Components**: 1000 embeddings
- **Represents**: Individual cells, proteins, molecular structures
- **Examples**: Keratinocytes, melanocytes, fibroblasts, collagen, elastin

### 2. Tissue Scale (Dermal Layers)
- **Components**: 50 embeddings
- **Represents**: Skin layers and tissue structures
- **Examples**: Epidermis layers (stratum corneum, basale), dermis, hypodermis

### 3. Region Scale (Body Areas)
- **Components**: 20 embeddings
- **Represents**: Body regions and anatomical areas
- **Examples**: Face, scalp, arms, hands, torso, legs

### 4. System Scale (Whole-Body Functions)
- **Components**: 5 embeddings
- **Represents**: Systemic functions and processes
- **Examples**: Barrier function, immune response, thermoregulation, sensory perception

## Key Components

### Neural Fabric Core (`shared/neural_fabric.py`)

The core fabric implementation provides:

- `SkinScale`: Scale enumeration (CELLULAR, TISSUE, REGION, SYSTEM)
- `TensorEmbedding`: Learnable embedding container for each scale
- `NeuralFabric`: Main fabric class with multi-scale management
- `create_default_fabric()`: Factory function for initialized fabric

### Integration Layer (`shared/fabric_integration.py`)

The integration layer connects cognitive systems:

- `FabricIntegrationLayer`: Main integration class
- AtomSpace concept mapping
- PLN reasoning on fabric embeddings
- MOSES pattern discovery
- ESN temporal prediction
- ECAN attention allocation
- Cross-system reasoning

## Usage

### Basic Setup

```python
from shared.neural_fabric import create_default_fabric, SkinScale
from shared.fabric_integration import create_integrated_platform

# Create fabric with default configuration
fabric = create_default_fabric(embedding_dimension=128)

# Or create integrated platform with all cognitive systems
fabric, integration = create_integrated_platform(embedding_dimension=128)
```

### Get Embeddings

```python
# Get embedding for a specific component at a scale
cellular_embedding = fabric.get_embedding(SkinScale.CELLULAR, component_id=0)
tissue_embedding = fabric.get_embedding(SkinScale.TISSUE, component_id=5)

# Get metadata for a component
metadata = fabric.scale_embeddings[SkinScale.TISSUE].get_metadata(0)
print(f"Component: {metadata.get('name')}")
```

### Cross-Scale Transformations

```python
# Transform embedding from one scale to another
tissue_emb = fabric.get_embedding(SkinScale.TISSUE, 0)
region_emb = fabric.transform_across_scales(
    tissue_emb,
    from_scale=SkinScale.TISSUE,
    to_scale=SkinScale.REGION
)
```

### Signal Propagation

```python
# Propagate signal across scales
activations = fabric.propagate_signal(
    source_scale=SkinScale.CELLULAR,
    source_component=0,
    target_scale=SkinScale.TISSUE,
    signal_strength=1.0
)

# activations is dict: {component_id: activation_level}
```

### Similarity Search

```python
# Find similar components at a scale
query_embedding = fabric.get_embedding(SkinScale.TISSUE, 0)
similar = fabric.query_fabric(
    query_embedding,
    scale=SkinScale.TISSUE,
    top_k=5
)

# similar is list of (component_id, similarity_score) tuples
```

### AtomSpace Integration

```python
# Map knowledge concept to fabric
concept_id = integration.atomspace_to_fabric(
    "melanoma",
    scale=SkinScale.TISSUE
)

# Retrieve concept information
concept_info = integration.fabric_to_atomspace(
    concept_id,
    scale=SkinScale.TISSUE
)
```

### PLN Reasoning

```python
# Perform probabilistic inference
result = integration.pln_inference_on_fabric(
    premise_concepts=["inflammation", "immune_response"],
    conclusion_concept="dermatitis",
    scale=SkinScale.TISSUE
)

print(f"Truth Value: {result['truth_value']}")
print(f"Confidence: {result['confidence']}")
```

### MOSES Pattern Discovery

```python
# Search for patterns matching a target
target_pattern = fabric.get_embedding(SkinScale.TISSUE, 5)
patterns = integration.moses_search_on_fabric(
    target_pattern=target_pattern,
    search_scale=SkinScale.CELLULAR,
    population_size=10
)

# patterns is list of (component_id, fitness_score) tuples
```

### Learning and Adaptation

```python
# Update fabric from observations
observation = np.random.randn(128)  # Example observation
fabric.update_from_observation(
    scale=SkinScale.TISSUE,
    component_id=0,
    observation=observation,
    learning_rate=0.01
)
```

### Persistence

```python
# Save fabric state
fabric.save_fabric("skin_fabric.json")

# Load fabric state
fabric.load_fabric("skin_fabric.json")
```

## Demo Notebook

See the [neural-fabric-demo.ipynb](labs/neural-fabric-demo/neural-fabric-demo.ipynb) notebook for a comprehensive demonstration including:

1. Fabric initialization
2. Scale exploration
3. Cross-scale transformations
4. Signal propagation
5. Similarity search
6. Cognitive system integration
7. AtomSpace concept mapping
8. PLN reasoning
9. MOSES pattern discovery
10. Learning and adaptation
11. Cross-system reasoning
12. Visualizations
13. Persistence

## Integration with SkinTwin Cognitive Architecture

The neural fabric serves as the foundational platform for all SkinTwin cognitive systems:

### AtomSpace (Knowledge Representation)
- Concepts mapped to fabric embeddings
- Relations encoded as cross-scale connections
- Semantic similarity via embedding proximity

### PLN (Probabilistic Logic Networks)
- Inference over fabric-embedded concepts
- Truth values as embedding similarities
- Confidence from embedding consistency

### MOSES (Pattern Mining)
- Evolutionary search in embedding space
- Fitness as embedding coherence
- Pattern discovery across scales

### ESN (Echo State Networks)
- Temporal dynamics anchored in fabric state
- Reservoir mapped to embedding space
- Multi-scale temporal prediction

### ECAN (Attention Allocation)
- Attention as activation over fabric
- Dynamic focus across scales
- Resource allocation based on embedding activation

## Design Principles

1. **Multi-Scale by Design**: Built from the ground up for hierarchical representation
2. **Learning-First**: All embeddings are learnable and adapt through experience
3. **Cognitive Substrate**: Forms the platform that contains all other systems
4. **Interoperability**: Seamless integration between cognitive components
5. **Persistence**: State can be saved, shared, and reloaded
6. **Scalability**: Designed to handle large-scale dermatological knowledge

## Technical Details

- **Embedding Dimension**: 128 (configurable)
- **Initialization**: Small random values (0.01 std)
- **Learning**: Gradient-based updates
- **Similarity**: Cosine similarity
- **Transformations**: Learnable linear matrices
- **Storage**: JSON serialization

## Requirements

```
numpy
matplotlib
```

See `requirements.txt` for complete dependencies.

## Future Enhancements

- [ ] GPU acceleration for large-scale fabric
- [ ] Advanced learning algorithms (meta-learning, few-shot)
- [ ] Graph neural network integration for relationships
- [ ] Attention mechanisms for cross-scale focus
- [ ] Distributed fabric for federated learning
- [ ] Real-time learning from clinical data streams
- [ ] Integration with medical imaging pipelines
- [ ] Explainability and interpretability tools

## Medical Applications

The neural fabric enables advanced dermatological AI applications:

- **Disease Diagnosis**: Multi-scale pattern recognition
- **Treatment Planning**: Cross-scale reasoning about interventions
- **Progression Modeling**: Temporal prediction of condition evolution
- **Personalized Medicine**: Patient-specific fabric adaptation
- **Drug Discovery**: Pattern mining for treatment targets
- **Clinical Decision Support**: Integrated reasoning across knowledge sources

## References

- **OpenCog AtomSpace**: https://wiki.opencog.org/w/AtomSpace
- **PLN**: https://wiki.opencog.org/w/PLN
- **MOSES**: https://wiki.opencog.org/w/MOSES
- **Echo State Networks**: Reservoir computing for temporal dynamics
- **Multi-Scale Modeling**: Hierarchical representation in biology

## License

See LICENSE.md in repository root.

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

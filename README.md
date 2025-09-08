# UnparametrizeGeom_DON

A deep learning framework for solving Partial Differential Equations (PDEs) on arbitrary 2D geometries using DeepONet (Deep Operator Networks) combined with geometry parameterization via Signed Distance Functions (SDFs).

## Overview

This repository implements a neural operator approach to solve PDEs on varying geometric domains. The key innovation is the ability to train a single model that can generalize across different geometric shapes without retraining. The framework combines:

- **DeepONet**: Neural operators that learn mappings between function spaces
- **Geometry Parameterization**: Using Signed Distance Functions (SDFs) to represent arbitrary 2D shapes
- **Meta-Learning**: Training on multiple geometries to achieve cross-domain generalization
- **PDE Solving**: Focus on Poisson equations with varying boundary conditions and source terms

## Key Features

- Solve PDEs on arbitrary 2D geometries without mesh-specific training
- Generate synthetic datasets for various geometric shapes (squares with holes, polygons, stars)
- Meta-learning approach for generalization across different domains
- Efficient inference on new, unseen geometries
- Comprehensive visualization and evaluation tools

## Installation

### Prerequisites

The framework requires the following main dependencies:

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Firedrake (for PDE solving and data generation)
- tqdm
- json

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Limekaaa/UnparametrizeGeom_DON.git
cd UnparametrizeGeom_DON
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Firedrake (required for PDE data generation):
```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install
```

4. Activate the Firedrake environment:
```bash
source firedrake/bin/activate
```

**Note**: Firedrake is essential for the finite element computations and PDE data generation. Make sure to activate the Firedrake environment before running any scripts.

## Project Structure

```
UnparametrizeGeom_DON/
├── UDON/                          # Core library
│   ├── data.py                    # Data loading and processing utilities
│   ├── inference.py               # Model inference functions
│   └── workspace.py               # Project workspace management
├── data_generators/               # Data generation modules
│   ├── Poisson2D_random_shape.py  # Poisson equation solver
│   ├── msh_unit_square_with_holes.py  # Square geometry generator
│   ├── msh_polygons.py            # Polygon geometry generator
│   └── msh_stars.py               # Star geometry generator
├── networks/                      # Neural network architectures
│   ├── DeepONet2DGeometry_upgraded.py  # Main DeepONet implementation
│   ├── MetaLearningDON.py         # Meta-learning variant
│   ├── VanillaDON.py              # Standard DeepONet
│   └── deep_sdf_decoder.py        # SDF decoder network
├── experiments/                   # Experiment configurations
│   └── unit_squares_1_3_holes/    # Example experiment
│       ├── specs.json             # Model configuration
│       └── specs_data.json        # Data generation configuration
├── training_DON.py                # Main training script
├── generate_sdf_data.py           # SDF data generation
├── generate_2D_data.py            # PDE data generation
├── generate_training_vectors.py   # Training data preparation
├── plot_pde.py                    # Visualization utilities
└── plot_log.py                    # Training progress visualization
```

## Quick Start

To get started quickly with a working example:

```bash
# 1. Generate geometric data
python generate_sdf_data.py -e experiments/unit_squares_1_3_holes/

# 2. Generate PDE solutions
python generate_2D_data.py -e experiments/unit_squares_1_3_holes/ -b 32

# 3. Prepare training data
python generate_training_vectors.py -e experiments/unit_squares_1_3_holes/

# 4. Train the model (this may take several hours)
python training_DON.py -e experiments/unit_squares_1_3_holes/

# 5. Visualize results
python plot_pde.py -e experiments/unit_squares_1_3_holes/ -cd latest -cs latest -s test --n_reconstructions 3
```

This will train a model to solve Poisson equations on unit squares with 1-3 randomly placed holes.

## Usage

### 1. Data Generation

#### Generate Geometric Data (SDF)

First, generate the geometric representations using Signed Distance Functions:

```bash
python generate_sdf_data.py -e experiments/unit_squares_1_3_holes/
```

This creates:
- Mesh files (.msh) for finite element computation
- SDF samples (.npz) for neural network training
- Split files (.json) defining train/test data divisions

**Generated Data Structure:**
```
data/
├── unit_squares_with_holes_test/
│   ├── [geometry_id].msh          # Firedrake mesh files
│   ├── [geometry_id].npz          # SDF samples
│   └── [geometry_id]_[coeff_id]/  # PDE solutions
│       └── pde_data.npz
└── ...
```

#### Generate PDE Data

Generate solutions to the Poisson equation on the geometries:

```bash
python generate_2D_data.py -e experiments/unit_squares_1_3_holes/ -b 64
```

Parameters:
- `-e`: Experiment directory containing configuration files
- `-b`: Batch size for data generation (optional)

#### Prepare Training Vectors

Convert the generated data into training-ready format:

```bash
python generate_training_vectors.py -e experiments/unit_squares_1_3_holes/
```

### 2. Model Training

Train a DeepONet model on the generated data:

```bash
python training_DON.py -e experiments/unit_squares_1_3_holes/
```

To continue training from a specific checkpoint:

```bash
python training_DON.py -e experiments/unit_squares_1_3_holes/ -c 1000
```

Parameters:
- `-e`: Experiment directory containing configuration files
- `-c`: Checkpoint to continue from (optional)

The training script will:
- Load configuration from `specs.json` and `specs_data.json`
- Initialize the neural networks (branch and trunk networks)
- Train the model using the specified learning rate schedule
- Save model checkpoints at regular intervals
- Log training progress

**Expected Output:**
```
INFO:root:epoch 0: train loss: 0.024513, test loss: 0.019832, lr: 0.0005
INFO:root:epoch 10: train loss: 0.015642, test loss: 0.012456, lr: 0.0005
...
```

**Saved Files:**
```
experiments/unit_squares_1_3_holes/
├── DeepONet/
│   └── ModelParameters/
│       ├── latest.pth              # Latest checkpoint
│       ├── best.pth                # Best validation loss
│       └── [epoch_num].pth         # Epoch-specific checkpoints
└── Logs/
    ├── loss.txt                    # Training/validation losses
    └── normalized_err.txt          # Normalized errors
```

#### Training Configuration

Key parameters in `specs.json`:

```json
{
    "DeepONet": {
        "NetworkSpecs": {
            "num_branch_inputs": 1,      # Input dimension for branch network
            "num_basis_functions": 128,   # Number of basis functions
            "num_trunk_inputs": 2,        # Coordinate dimensions (x, y)
            "branch_dims": [128, 128, 128, 128],  # Branch network architecture
            "trunk_dims": [128, 128, 128, 128],   # Trunk network architecture
            "latent_dim": 32             # Geometry encoding dimension
        },
        "NumEpochs": 2000,
        "ScenesPerBatch": 64,
        "SamplesPerScene": 200,
        "LearningRateSchedule": [
            {
                "Type": "Step",
                "Initial": 0.0005,
                "Interval": 500,
                "Factor": 0.5
            }
        ]
    }
}
```

### 3. Model Evaluation

#### Visualize PDE Solutions

Plot and compare predicted vs. ground truth solutions:

```bash
python plot_pde.py -e experiments/unit_squares_1_3_holes/ -cd latest -cs latest -s test --n_reconstructions 5
```

Parameters:
- `-e`: Experiment directory
- `-cd`: DeepONet checkpoint to load (e.g., "latest", "1000")
- `-cs`: DeepSDF checkpoint to load (e.g., "latest", "1000")
- `-s`: Data split to use ("train" or "test")
- `--n_reconstructions`: Number of test cases to visualize

This generates:
- Solution heatmaps
- Error distributions
- Quantitative metrics (L2 error, etc.)

#### Monitor Training Progress

Visualize training logs and metrics:

```bash
python plot_log.py -e experiments/unit_squares_1_3_holes/
```

### 4. Solving PDEs on New Geometries

#### Inference on Unseen Shapes

The trained model can solve PDEs on new geometries without retraining:

1. Generate SDF data for the new geometry
2. Use the inference module:

```python
import torch
from UDON.inference import load_model, solve_pde
from networks.MetaLearningDON import DeepONet

# Load trained model
model = load_model("experiments/unit_squares_1_3_holes/", checkpoint="latest")

# Define new geometry (SDF samples)
geometry_encoding = generate_geometry_encoding(new_shape_sdf)

# Define PDE parameters
rhs_function = lambda x, y: torch.sin(x) * torch.cos(y)  # Right-hand side
boundary_conditions = 0.0  # Dirichlet BC

# Solve PDE
solution = solve_pde(model, geometry_encoding, rhs_function, boundary_conditions)
```

## Configuration

### Data Generation Configuration (`specs_data.json`)

```json
{
    "dataset_name": "unit_squares_with_holes_test",
    "SDFDataGenerator": "msh_unit_square_with_holes",
    "PDEDataGenerator": "Poisson2D_random_shape",
    "SDFData": {
        "num_samples": 5,           # Number of geometric variations
        "min_hole": 1,              # Minimum number of holes
        "max_hole": 3,              # Maximum number of holes
        "min_radius": 0.05,         # Minimum hole radius
        "SamplesPerScene": 10201    # SDF samples per geometry
    },
    "PDEData": {
        "n_coeffs": 100,            # Number of RHS variations
        "bc_val": 0.0,              # Boundary condition value
        "rhs": "lambda x,y: sin(y * x[0]) * sin(y * x[1])"  # RHS function
    }
}
```

### Model Configuration (`specs.json`)

The configuration includes:
- **Network Architecture**: Layer dimensions, activation functions
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Data Loading**: Memory management, threading
- **Regularization**: Dropout, weight decay
- **Checkpointing**: Save frequency, snapshot management

## Examples

### Example 1: Training on Squares with Holes

```bash
# Generate data
python generate_sdf_data.py -e experiments/unit_squares_1_3_holes/
python generate_2D_data.py -e experiments/unit_squares_1_3_holes/ -b 32
python generate_training_vectors.py -e experiments/unit_squares_1_3_holes/

# Train model
python training_DON.py -e experiments/unit_squares_1_3_holes/

# Evaluate results
python plot_pde.py -e experiments/unit_squares_1_3_holes/ -cd latest -cs latest -s test --n_reconstructions 3
```

### Example 2: Custom Geometry

To work with custom geometries:

1. Create a new data generator in `data_generators/`
2. Implement SDF computation for your geometry
3. Update configuration files
4. Follow the standard workflow

### Example 3: Different PDE Types

To solve different PDEs:

1. Modify the PDE generator in `data_generators/`
2. Implement your PDE solver using Firedrake
3. Update the RHS function and boundary conditions
4. Retrain the model

## Advanced Features

### Meta-Learning

The framework supports meta-learning across different geometry families:

- Train on multiple geometry types simultaneously
- Learn geometry-invariant representations
- Fast adaptation to new geometric domains

### Custom Loss Functions

Implement custom loss functions for specific PDE types or physics constraints:

```python
def physics_informed_loss(pred, target, pde_residual):
    mse_loss = torch.nn.MSELoss()(pred, target)
    physics_loss = torch.mean(pde_residual**2)
    return mse_loss + 0.1 * physics_loss
```

### Distributed Training

For large-scale experiments:

```bash
python -m torch.distributed.launch --nproc_per_node=4 training_DON.py -e experiments/large_experiment/
```

## Troubleshooting

### Common Issues

1. **Firedrake Installation**: Ensure Firedrake is properly installed and activated
   ```bash
   source firedrake/bin/activate
   python -c "from firedrake import *; print('Firedrake working!')"
   ```

2. **Memory Issues**: Reduce batch size or number of samples per scene
   ```json
   "ScenesPerBatch": 32,     // Reduce from 64
   "SamplesPerScene": 100    // Reduce from 200
   ```

3. **Convergence Problems**: Adjust learning rate schedule or network architecture
   ```json
   "LearningRateSchedule": [{
       "Type": "Step",
       "Initial": 0.0001,    // Lower initial learning rate
       "Interval": 1000,     // Longer intervals
       "Factor": 0.5
   }]
   ```

4. **CUDA Errors**: Ensure PyTorch CUDA version matches your CUDA installation
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Import Errors**: Ensure you're in the correct directory and environment
   ```bash
   cd UnparametrizeGeom_DON
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is available for faster training
  ```python
  import torch
  print(f"CUDA available: {torch.cuda.is_available()}")
  ```
- **Memory Management**: Reduce batch size if encountering out-of-memory errors
- **Data Loading**: Use multiple workers for faster data loading (set in specs.json)
- **Checkpointing**: Save intermediate results to resume training if interrupted

### Hardware Requirements

- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM (e.g., RTX 3070, V100)
- **Storage**: 10-50GB depending on dataset size

### Training Time Estimates

- **Small dataset** (5 geometries, 100 PDE coefficients): ~30 minutes on GPU
- **Medium dataset** (50 geometries, 1000 PDE coefficients): ~3-5 hours on GPU  
- **Large dataset** (500+ geometries): ~10+ hours on GPU

## Frequently Asked Questions

**Q: Can I use this for 3D geometries?**
A: The current implementation is designed for 2D geometries. Extending to 3D would require modifications to the data generators and network architectures.

**Q: How do I add a new type of PDE?**
A: Create a new file in `data_generators/` following the pattern of `Poisson2D_random_shape.py`. Implement your PDE solver using Firedrake and update the configuration files.

**Q: Can I use pre-existing mesh files?**
A: Yes, modify the data generators to load your mesh files instead of generating new ones. Ensure they're in Firedrake-compatible format.

**Q: How do I visualize training progress?**
A: Use `python plot_log.py -e experiments/your_experiment/` to see loss curves and metrics over time.

**Q: What if my geometries are very different from the training examples?**
A: The model generalizes best to geometries similar to the training distribution. For very different shapes, consider including representative examples in your training set.

**Q: How do I reduce memory usage?**
A: Reduce `ScenesPerBatch`, `SamplesPerScene`, or set `LoadRam` to false in the configuration to load data on-demand.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{unparametrize_geom_don,
  title={Unparametrized Geometry Deep Operator Networks for PDE Solving},
  author={[Author Names]},
  journal={[Journal]},
  year={[Year]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Firedrake team for the finite element framework
- DeepONet original authors for the neural operator concept
- PyTorch team for the deep learning framework
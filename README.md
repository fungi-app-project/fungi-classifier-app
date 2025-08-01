
# Dependable FungAI - README

## Overview

This project "Dependable FungAI" implements an **Enhanced Hybrid Concept-Based Mushroom Classification System** using advanced machine learning techniques including **CLIP (ViT-L/14)** for visual feature extraction and **Multi-Layer Perceptron (MLP)** networks. The system classifies mushrooms as either **edible** or **poisonous** based on both visual features and textual concept descriptions. The project generates comprehensive **Training Bill of Materials (TBOM)** and **Inference Bill of Materials (IBOM)** for complete transparency, traceability, and explainability.

## Key Features

- **Enhanced IBOM Analysis**: Interactive inference with concept categorization by morphological features
- **TBOM Visualization**: Comprehensive training documentation and performance metrics
- **Interactive Concept Modification**: Real-time adjustment of concept scores with impact visualization
- **Educational Mode**: Detailed explanations, safety guidance, and identification tips
- **DSSE Signature Support**: Supply chain security with in-toto attestation
- **Export Functionality**: Multiple download formats (JSON, CSV, comprehensive reports)
- **Advanced Safety Features**: Conflict detection and biological impossibility validation

## System Architecture

### Training Components (TBOM)
- **Enhanced Hybrid MLP Classifier** with concept-based learning
- **CLIP ViT-L/14** backbone for visual feature extraction
- **117 mushroom-specific concepts** generated from morphological features
- **Cross-validation** with comprehensive performance tracking

### Inference Components (IBOM)
- **Interactive concept categorization** by morphological features:
  - Cap properties (color, shape, surface)
  - Gill properties (color, spacing, attachment)
  - Stalk properties (shape, color, texture)
  - Sensory properties (odor detection)
  - Environmental context (habitat, growth patterns)
  - Reproductive features (spore characteristics)

## Key Terms Defined

- **TBOM (Training Bill of Materials)**: Comprehensive documentation of the training process, model architecture, performance metrics, and validation results
- **IBOM (Inference Bill of Materials)**: Detailed analysis of individual predictions including concept contributions, uncertainty visualization, and safety assessments
- **DSSE (Dead Simple Signing Envelope)**: Cryptographic signing standard for supply chain security and integrity verification
- **Concept Categorization**: Organization of mushroom identification features by morphological categories for educational clarity
- **Uncertainty Visualization**: Interactive charts showing prediction confidence, decision boundaries, and risk assessment
- **Conflict Detection**: Automated identification of biologically impossible feature combinations

## Requirements

### Data Set

1. For **CSV Dataset**: https://www.kaggle.com/datasets/uciml/mushroom-classification
2. For **Image Classification:** https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images
3. For **Multi- Modal Approach** (Images dataset): 
git clone: https://www.kaggle.com/datasets/derekkunowilliams/mushrooms

This has four folders: Edible, Conditionally Edible, Poisonous, and Deadly Poisonous. Merge Edible and Conditionally Edible into one Edible folder, and Poisonous and Deadly Poisonous into one Poisonous folder, for a binary classification setup.

### Core Dependencies

pip install streamlit torch clip-by-openai pillow plotly pandas numpy scikit-learn opencv-python


### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, falls back to CPU)
- 8GB+ RAM recommended for CLIP model loading

## Installation & Setup

1. **Clone the repository**:

git clone https://github.com/umaima786/dependable-fungai.git
cd dependable-fungai


2. **Install dependencies**:

pip install -r requirements.txt


3. **Set up environment variables** (optional but recommended):

export TBOM_PATH="/path/to/TBOM.json"
export MODEL_PATH="/path/to/final_model.pt"
export CSV_PATH="/path/to/mushrooms.csv"


## Usage Instructions

### 1. Training the Model (TBOM Generation)

**Run the training script**:

python TBOM.py


**Interactive prompts will ask for**:
- Path to mushroom CSV dataset (or press Enter for default)
- Path to mushroom image dataset (or press Enter for default)

**Outputs**:
- `TBOM.json`: Complete training documentation
- `final_model.pt`: Trained model weights
- Comprehensive performance metrics and validation results

### 2. Running the Interactive Dashboard

**Start the Streamlit application**:

streamlit run app.py


**Features available**:
- **TBOM Analysis Tabs**:
  - Data Summary: Dataset statistics and class distribution
  - Overview: Performance metrics and comparison charts
  - Performance: Confusion matrices, ROC/PR curves, training progress
  - Architecture: Model structure and component visualization
  - Concepts: Concept analysis and model interpretation
  - Technical: Environment details and raw TBOM data

- **Enhanced IBOM Tab**:
  - Interactive image upload and analysis
  - Real-time concept score modification by category
  - Uncertainty visualization with confidence intervals
  - Educational mode with safety guidance
  - Export functionality for analysis results

### 3. Running Inference with IBOM Generation

**Command-line inference**:

python IBOM.py --image_file mushroom.jpg --dsse --educational_mode


**Batch processing**:

python IBOM.py --image_dir images/ --output analysis.json --export_summary


**Parameters**:
- `--image_file`: Single mushroom image for analysis
- `--image_dir`: Directory containing multiple images
- `--output`: Custom output filename
- `--dsse`: Include DSSE signature for security
- `--educational_mode`: Enable detailed explanations
- `--export_summary`: Generate additional summary report

## Safety & Educational Features

### Critical Safety Notices
- **NEVER consume any mushroom based solely on AI analysis**
- Always consult certified mycologists for positive identification
- Misidentification can be fatal

### Educational Enhancements
- **Morphological categorization** with biological context
- **Interactive learning** through concept modification
- **Conflict detection** for impossible feature combinations
- **Uncertainty quantification** with confidence interpretation
- **Safety guidance** with risk assessment

## Advanced Features

### Security & Traceability
- **DSSE signatures** for supply chain integrity
- **Cryptographic hashing** of analysis results
- **Complete audit trail** from training to inference

### Export & Integration
- **JSON exports** for programmatic access
- **CSV data** for spreadsheet analysis
- **Comprehensive reports** for documentation
- **API-ready format** for system integration

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Run `python TBOM.py` first to generate required files
   - Set environment variables for custom paths

2. **Memory errors**:
   - Reduce batch size in training
   - Use CPU instead of GPU if VRAM insufficient

3. **Interactive prompts in production**:
   - Set all environment variables to avoid prompts
   - Use `--no_interactive` flag for automated deployment

### Getting Help

- Check the **Technical Details** tab in the dashboard for system information
- Enable **detailed error display** in the IBOM interface for debugging
- Consult the **Educational Mode** for feature explanations



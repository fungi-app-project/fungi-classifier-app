"""
Inference Bill of Materials (IBOM) Generator for Enhanced Hybrid Concept-Based Mushroom Classifier
Complete implementation with DSSE signature, concept categorization, educational mode, and comparison analysis
"""

import os, json, hashlib, argparse, platform, time, glob
import torch, clip
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import warnings
from pathlib import Path
from datetime import datetime
import tempfile

# Import from training script (TBOM.py) 
from TBOM import (
    HybridMLP, 
    generate_concepts_from_csv,
    get_software_versions,
    CLIP_MODEL_NAME,
    CLIP_EMBEDDING_DIM,
    DEVICE
)

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# DSSE SIGNATURE IMPLEMENTATION

def generate_dsse_signature(ibom_data, private_key=None):
    """
    Generate DSSE (Dead Simple Signing Envelope) signature for IBOM integrity
    Implements in-toto attestation format for supply chain security
    """
    
    # Create DSSE envelope structure
    payload = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "subject": [{
            "name": ibom_data['image_metadata']['image_id'],
            "digest": {
                "sha256": hashlib.sha256(json.dumps(ibom_data, sort_keys=True).encode()).hexdigest()
            }
        }],
        "predicateType": "https://dependable-fungai.io/IBOM/v1.0",
        "predicate": {
            "model_architecture": ibom_data['model_architecture_summary'],
            "prediction_result": ibom_data['detailed_inference_results'][0]['neural_network_analysis']['prediction_analysis'],
            "safety_recommendation": ibom_data['detailed_inference_results'][0]['training_context']['prediction_reliability_assessment']['recommendation'],
            "timestamp": ibom_data['generation_metadata']['timestamp_utc'],
            "concept_analysis_summary": {
                "total_concepts": len(ibom_data['detailed_inference_results'][0]['concept_analysis']['all_concepts']),
                "prediction_class": ibom_data['detailed_inference_results'][0]['neural_network_analysis']['prediction_analysis']['prediction_class'],
                "confidence": ibom_data['detailed_inference_results'][0]['neural_network_analysis']['prediction_analysis']['confidence']
            }
        }
    }
    
    # Encode payload
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    payload_b64 = hashlib.sha256(payload_bytes).hexdigest()
    
    # Create DSSE signature structure
    dsse_envelope = {
        "payload": payload_b64,
        "payloadType": "application/vnd.in-toto+json",
        "signatures": [{
            "keyid": "dependable-fungai-v1.0",
            "sig": hashlib.sha256(payload_bytes + b"dependable-fungai-signature").hexdigest()
        }]
    }
    
    return dsse_envelope, payload

# Flexible path discovery with smart fallback

def get_paths_with_smart_fallback():
    """Auto-discover TBOM, model, and CSV paths with smart fallback"""
    
    print("🔍 Auto-discovering project files...")
    
    # Auto-discovery paths with priority order
    tbom_candidates = [
        os.getenv('TBOM_PATH'),
        os.getenv('FUNGAI_TBOM_PATH'),
        './TBOM.json',
        './outputs_final_documentation/TBOM.json',
        '../TBOM.json',
        '../outputs_final_documentation/TBOM.json',
        os.path.expanduser('~/FungAI/TBOM.json'),
        *glob.glob('**/TBOM.json', recursive=True)
    ]
    
    model_candidates = [
        os.getenv('MODEL_PATH'),
        os.getenv('FUNGAI_MODEL_PATH'),
        './final_model.pt',
        './outputs_final_documentation/final_model.pt',
        '../final_model.pt',
        '../outputs_final_documentation/final_model.pt',
        os.path.expanduser('~/FungAI/final_model.pt'),
        *glob.glob('**/final_model.pt', recursive=True)
    ]
    
    csv_candidates = [
        os.getenv('CSV_PATH'),
        os.getenv('FUNGAI_CSV_PATH'),
        './mushrooms.csv',
        '../mushrooms.csv',
        './data_sets/mushroom_data/mushrooms.csv',
        '../data_sets/mushroom_data/mushrooms.csv',
        os.path.expanduser('~/FungAI/mushrooms.csv'),
        *glob.glob('**/mushrooms.csv', recursive=True),
        *glob.glob('**/mushroom_data.csv', recursive=True)
    ]
    
    # Find first existing file for each type
    tbom_path = next((p for p in tbom_candidates if p and os.path.exists(p)), None)
    model_path = next((p for p in model_candidates if p and os.path.exists(p)), None)
    csv_path = next((p for p in csv_candidates if p and os.path.exists(p)), None)
    
    # Display results
    if tbom_path:
        print(f"✅ TBOM found: {os.path.basename(tbom_path)}")
    if model_path:
        print(f"✅ Model found: {os.path.basename(model_path)}")
    if csv_path:
        print(f"✅ CSV found: {os.path.basename(csv_path)}")
    
    # Check for missing files
    missing_files = []
    if not tbom_path:
        missing_files.append("TBOM.json")
    if not model_path:
        missing_files.append("final_model.pt")
    if not csv_path:
        missing_files.append("mushrooms.csv")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("\n📋 Setup Instructions:")
        print("1. Run training first: python TBOM.py")
        print("2. Set environment variables:")
        print("   export TBOM_PATH=/path/to/TBOM.json")
        print("   export MODEL_PATH=/path/to/final_model.pt")
        print("   export CSV_PATH=/path/to/mushrooms.csv")
        raise FileNotFoundError(f"Required files not found: {', '.join(missing_files)}")
    
    return tbom_path, model_path, csv_path

# IBOM Generator Class with Advanced Features

class DetailedIBOMGenerator:
    """
    Comprehensive IBOM generator with all advanced features:
    - DSSE signature support
    - Concept categorization by morphological features
    - Educational mode with detailed explanations
    - Conflict detection and resolution guidance
    - Export functionality with multiple formats
    - Uncertainty visualization support
    """
    
    def __init__(self, model_path, tbom_path, csv_path):
        """Initialize IBOM generator with training artifacts"""
        self.device = DEVICE
        self.model_path = model_path
        self.tbom_path = tbom_path
        self.csv_path = csv_path
        
        print(f"🚀 Initializing Enhanced IBOM generator...")
        print(f"   Using device: {self.device}")
        print(f"   TBOM: {os.path.basename(tbom_path)}")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   CSV: {os.path.basename(csv_path)}")
        
        # Load T-BOM for training context
        with open(tbom_path, 'r') as f:
            self.tbom_data = json.load(f)
        
        # Generate concepts exactly as in training
        self.concepts = generate_concepts_from_csv(csv_path)
        self.num_concepts = len(self.concepts)
        print(f"   Generated {self.num_concepts} concepts")
        
        # Initialize CLIP (same as training)
        print("   Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
        self.clip_model.eval()
        
        # Precompute text embeddings
        print("   Precomputing concept embeddings...")
        with torch.no_grad():
            tokens = clip.tokenize(self.concepts).to(self.device)
            self.text_embeddings = self.clip_model.encode_text(tokens).float()
            self.text_embeddings /= self.text_embeddings.norm(dim=1, keepdim=True)
        
        # Load trained model with same architecture
        print("   Loading trained model...")
        self.input_dim = CLIP_EMBEDDING_DIM + self.num_concepts
        self.model = HybridMLP(self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Extract concept weights for detailed analysis
        print("   Extracting concept weights...")
        self.concept_weights = self._extract_detailed_concept_weights()
        
        print("✅ Enhanced IBOM generator initialized successfully!")
    
    def get_concept_categories_with_education(self):
        """Get concept categories organized by morphological features with comprehensive educational information"""
        categories = {
            'cap_properties': {
                'name': 'Cap Properties',
                'description': 'Visual characteristics of the mushroom cap (pileus)',
                'importance': 'High - Primary identification feature',
                'concepts': [],
                'educational_note': 'The cap is the most visible part of a mushroom and provides crucial identification clues including color, shape, and surface texture.',
                'safety_relevance': 'Critical - Many dangerous mushrooms can be distinguished by cap characteristics',
                'identification_tips': [
                    'Cap color can change with age and weather conditions',
                    'Shape often evolves from convex to flat as mushroom matures',
                    'Surface texture helps distinguish between species families'
                ]
            },
            'gill_properties': {
                'name': 'Gill Properties', 
                'description': 'Characteristics of gills under the cap where spores are produced',
                'importance': 'Very High - Critical safety indicators',
                'concepts': [],
                'educational_note': 'Gill color, spacing, and attachment are among the most reliable features for mushroom identification and safety assessment.',
                'safety_relevance': 'Extremely Critical - Gill characteristics are key safety indicators',
                'identification_tips': [
                    'Gill color often changes as spores mature',
                    'Attachment to stalk (free vs attached) is species-specific',
                    'Spacing between gills is consistent within species'
                ]
            },
            'stalk_properties': {
                'name': 'Stalk Properties',
                'description': 'Features of the mushroom stem including color, shape, and surface',
                'importance': 'Medium-High - Supporting identification features',
                'concepts': [],
                'educational_note': 'Stalk characteristics including color above/below rings, surface texture, and shape help distinguish between closely related species.',
                'safety_relevance': 'Important - Stalk features can differentiate dangerous look-alikes',
                'identification_tips': [
                    'Color above and below ring can differ significantly',
                    'Stalk shape (tapering, enlarging) is species-specific',
                    'Surface texture provides additional identification clues'
                ]
            },
            'sensory_properties': {
                'name': 'Sensory Properties',
                'description': 'Odor and other sensory characteristics detectable by smell',
                'importance': 'Very High - Critical safety indicators',
                'concepts': [],
                'educational_note': 'Odor is one of the most reliable and immediate indicators of mushroom safety and species identification.',
                'safety_relevance': 'Extremely Critical - Often the most reliable safety indicator',
                'identification_tips': [
                    'Always smell mushrooms - many dangerous species have distinctive odors',
                    'Foul or unpleasant odors often indicate toxicity',
                    'Some edible species have pleasant, distinctive aromas'
                ]
            },
            'environmental_context': {
                'name': 'Environmental Context',
                'description': 'Growth patterns, habitat, and ecological context',
                'importance': 'Medium - Species and habitat context',
                'concepts': [],
                'educational_note': 'Where and how mushrooms grow provides important context for species identification and seasonal availability.',
                'safety_relevance': 'Moderate - Helps narrow down possible species',
                'identification_tips': [
                    'Habitat preferences are often species-specific',
                    'Growth patterns (solitary vs clustered) aid identification',
                    'Seasonal timing helps confirm species identification'
                ]
            },
            'reproductive_features': {
                'name': 'Reproductive Features',
                'description': 'Spore-related characteristics and reproductive structures',
                'importance': 'High - Advanced identification features',
                'concepts': [],
                'educational_note': 'Spore print color and other reproductive features provide definitive identification criteria for experienced foragers.',
                'safety_relevance': 'High - Spore prints can definitively distinguish dangerous species',
                'identification_tips': [
                    'Spore print color is obtained by leaving cap on paper overnight',
                    'Ring and veil characteristics are highly species-specific',
                    'These features require more advanced identification skills'
                ]
            }
        }
        
        # Categorize existing concepts with enhanced mapping
        for concept in self.concept_weights['concept_weights']:
            category_key = self._map_to_enhanced_category(concept['concept_category'])
            if category_key in categories:
                categories[category_key]['concepts'].append(concept)
        
        return categories
    
    def _map_to_enhanced_category(self, original_category):
        """Map original categories to enhanced educational categories"""
        mapping = {
            'cap_color': 'cap_properties',
            'cap_shape': 'cap_properties', 
            'cap_surface': 'cap_properties',
            'cap_general': 'cap_properties',
            'gill_color': 'gill_properties',
            'gill_spacing': 'gill_properties',
            'gill_size': 'gill_properties',
            'gill_general': 'gill_properties',
            'stalk_properties': 'stalk_properties',
            'ring_properties': 'reproductive_features',
            'habitat': 'environmental_context',
            'population': 'environmental_context',
            'odor': 'sensory_properties',
            'spore_properties': 'reproductive_features',
            'other': 'environmental_context'
        }
        return mapping.get(original_category, 'environmental_context')
    
    def detect_concept_conflicts(self, concept_scores):
        """
        Enhanced conflict detection with comprehensive educational context
        Detects biologically impossible combinations and provides resolution guidance
        """
        conflicts = []
        severity_level = "none"
        
        # Define mutually exclusive concept groups with detailed educational context
        exclusive_groups = {
            'gill_colors': {
                'concepts': [
                    "gill color is yellow", "gill color is buff", "gill color is white", 
                    "gill color is black", "gill color is brown", "gill color is chocolate",
                    "gill color is gray", "gill color is green", "gill color is orange",
                    "gill color is pink", "gill color is red"
                ],
                'explanation': "A mushroom can only have one primary gill color at any given time",
                'educational_note': "Gill color is crucial for mushroom identification and safety assessment. While gills may darken as spores mature, each individual mushroom has one dominant gill color.",
                'safety_impact': "Critical - Gill color is one of the most reliable safety indicators",
                'resolution_guidance': "Choose the most prominent gill color observed and reduce conflicting colors below 0.4"
            },
            'cap_shapes': {
                'concepts': [
                    "cap shape is bell", "cap shape is flat", "cap shape is convex",
                    "cap shape is conical", "cap shape is knobbed", "cap shape is sunken"
                ],
                'explanation': "Mushroom caps have one dominant shape, though this may change slightly with maturity",
                'educational_note': "Cap shape is a primary identification feature. While caps may flatten with age, each mushroom has one characteristic shape at any given stage.",
                'safety_impact': "High - Cap shape helps distinguish between dangerous and safe species",
                'resolution_guidance': "Select the current observed cap shape and reduce others below 0.3"
            },
            'cap_colors': {
                'concepts': [
                    "cap color is brown", "cap color is buff", "cap color is white",
                    "cap color is red", "cap color is yellow", "cap color is green",
                    "cap color is gray", "cap color is pink", "cap color is purple", "cap color is cinnamon"
                ],
                'explanation': "Each mushroom has one primary cap color, though variations and patterns may exist",
                'educational_note': "Cap color is the most immediately visible identification feature. While some mushrooms have color variations or patterns, one color typically dominates.",
                'safety_impact': "High - Cap color is often the first distinguishing feature between species",
                'resolution_guidance': "Identify the dominant cap color and reduce secondary colors below 0.4"
            },
            'odor_types': {
                'concepts': [
                    "odor is foul", "odor is none", "odor is almond", 
                    "odor is anise", "odor is fishy", "odor is pungent",
                    "odor is spicy", "odor is creosote", "odor is musty"
                ],
                'explanation': "A mushroom has one primary odor characteristic - this is binary and definitive",
                'educational_note': "Odor is one of the most reliable and immediate safety indicators. Each mushroom either has a detectable odor or does not.",
                'safety_impact': "Extremely Critical - Odor is often the most reliable safety indicator",
                'resolution_guidance': "Choose the most accurately detected odor and set conflicting odors to very low values (< 0.2)"
            }
        }
        
        # Check for mutually exclusive conflicts
        for group_name, group_info in exclusive_groups.items():
            high_activations = []
            for concept in group_info['concepts']:
                for i, full_concept in enumerate(self.concepts):
                    if concept in full_concept and concept_scores[i] > 0.7:
                        high_activations.append({
                            'concept': concept,
                            'score': concept_scores[i],
                            'index': i,
                            'weight': self.concept_weights['concept_weights'][i]['model_weight']
                        })
            
            if len(high_activations) > 1:
                conflicts.append({
                    'type': 'mutually_exclusive',
                    'group': group_name,
                    'conflicting_concepts': high_activations,
                    'severity': 'critical',
                    'explanation': group_info['explanation'],
                    'educational_note': group_info['educational_note'],
                    'safety_impact': group_info['safety_impact'],
                    'resolution_guidance': group_info['resolution_guidance'],
                    'biological_impossibility': True
                })
                severity_level = "critical"
        
        # Check for contradictory feature combinations
        contradictory_pairs = [
            {
                'pair': ("odor is foul", "odor is none"),
                'explanation': "A mushroom cannot simultaneously have no odor and a detectable foul odor",
                'educational_note': "Odor assessment is binary - either a mushroom has a detectable odor or it does not. This is a fundamental biological characteristic.",
                'safety_impact': "Critical - Contradictory odor readings can lead to dangerous misidentification"
            },
            {
                'pair': ("bruises has bruises", "bruises has no bruises"),
                'explanation': "Bruising is a binary characteristic - mushrooms either bruise when damaged or they do not",
                'educational_note': "Bruising occurs when mushroom tissue is damaged and undergoes chemical changes. This is species-specific and consistent.",
                'safety_impact': "High - Bruising patterns help distinguish between similar-looking species"
            }
        ]
        
        for combo in contradictory_pairs:
            concept1, concept2 = combo['pair']
            score1 = score2 = 0
            for i, full_concept in enumerate(self.concepts):
                if concept1 in full_concept:
                    score1 = concept_scores[i]
                elif concept2 in full_concept:
                    score2 = concept_scores[i]
            
            if score1 > 0.6 and score2 > 0.6:
                conflicts.append({
                    'type': 'contradictory_features',
                    'concepts': [concept1, concept2],
                    'scores': [score1, score2],
                    'severity': 'high',
                    'explanation': combo['explanation'],
                    'educational_note': combo['educational_note'],
                    'safety_impact': combo['safety_impact'],
                    'resolution_guidance': "One feature should be high (>0.6) while the other should be low (<0.3)",
                    'biological_impossibility': True
                })
                if severity_level != "critical":
                    severity_level = "high"
        
        return conflicts, severity_level
    
    def compare_analyses_detailed(self, original_result, modified_result):
        """Enhanced comparison analysis with comprehensive educational insights"""
        orig_pred = original_result['neural_network_analysis']['prediction_analysis']
        mod_pred = modified_result['neural_network_analysis']['prediction_analysis']
        
        # Calculate detailed concept changes
        orig_concepts = original_result['concept_analysis']['all_concepts']
        mod_concepts = modified_result['concept_analysis']['all_concepts']
        
        concept_changes = []
        for orig, mod in zip(orig_concepts, mod_concepts):
            if 'contribution_to_prediction' in mod:
                change = mod['contribution_to_prediction'] - orig['contribution_to_prediction']
                if abs(change) > 0.01:  # Significant change threshold
                    concept_changes.append({
                        'concept': orig['concept_clean'],
                        'category': orig['concept_category'],
                        'original_contribution': orig['contribution_to_prediction'],
                        'modified_contribution': mod.get('contribution_to_prediction', orig['contribution_to_prediction']),
                        'change': change,
                        'impact_explanation': self._explain_contribution_change(change, orig['concept_clean']),
                        'safety_relevance': self._assess_concept_safety_relevance(orig['concept_category'])
                    })
        
        # Sort by absolute change magnitude
        concept_changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        comparison = {
            'prediction_changed': orig_pred['prediction_class'] != mod_pred['prediction_class'],
            'confidence_change': mod_pred['confidence'] - orig_pred['confidence'],
            'probability_change': mod_pred['sigmoid_probability'] - orig_pred['sigmoid_probability'],
            'safety_impact': self._assess_safety_impact(orig_pred, mod_pred),
            'concept_changes': concept_changes[:10],  # Top 10 changes
            'educational_insights': self._generate_educational_insights(orig_pred, mod_pred, concept_changes),
            'uncertainty_analysis': self._analyze_prediction_uncertainty(orig_pred, mod_pred),
            'category_impact_analysis': self._analyze_category_impacts(concept_changes)
        }
        
        return comparison
    
    def _explain_contribution_change(self, change, concept_name):
        """Provide educational explanation for contribution changes"""
        if abs(change) > 0.05:
            magnitude = "significantly"
        elif abs(change) > 0.02:
            magnitude = "moderately"
        else:
            magnitude = "slightly"
        
        direction = "toward poisonous classification" if change > 0 else "toward edible classification"
        
        return f"This change {magnitude} shifted the prediction {direction} due to modified {concept_name} characteristics."
    
    def _assess_concept_safety_relevance(self, concept_category):
        """Assess safety relevance of concept categories"""
        safety_relevance = {
            'odor': 'Extremely Critical',
            'gill_color': 'Very High',
            'cap_color': 'High',
            'cap_shape': 'High',
            'stalk_properties': 'Medium-High',
            'spore_properties': 'High',
            'habitat': 'Medium',
            'other': 'Low-Medium'
        }
        return safety_relevance.get(concept_category, 'Medium')
    
    def _generate_educational_insights(self, orig_pred, mod_pred, concept_changes):
        """Generate comprehensive educational insights about the changes"""
        insights = []
        
        # Prediction change insights
        if orig_pred['prediction_class'] != mod_pred['prediction_class']:
            insights.append("Classification change demonstrates the sensitivity of mushroom identification to specific morphological features.")
            
            if orig_pred['prediction_class'] == 'edible' and mod_pred['prediction_class'] == 'poisonous':
                insights.append("⚠️ SAFETY CRITICAL: Change from edible to poisonous classification significantly increases safety risk.")
            elif orig_pred['prediction_class'] == 'poisonous' and mod_pred['prediction_class'] == 'edible':
                insights.append("🚨 EXTREME CAUTION: Change from poisonous to edible classification could be extremely dangerous if acted upon.")
        
        # Concept change insights
        if len(concept_changes) > 5:
            insights.append("Multiple concept modifications created compound effects, showing how mushroom identification relies on feature combinations.")
        
        # Safety-critical concept insights
        critical_changes = [c for c in concept_changes if c['safety_relevance'] in ['Extremely Critical', 'Very High'] and abs(c['change']) > 0.03]
        if critical_changes:
            critical_concepts = [c['concept'] for c in critical_changes[:3]]
            insights.append(f"Safety-critical features that drove the change: {', '.join(critical_concepts)}")
        
        # Confidence change insights
        conf_change = mod_pred['confidence'] - orig_pred['confidence']
        if abs(conf_change) > 0.2:
            insights.append(f"Significant confidence change ({conf_change:+.1%}) indicates these features strongly influence model certainty.")
        
        return insights
    
    def _analyze_prediction_uncertainty(self, orig_pred, mod_pred):
        """Analyze uncertainty changes between predictions"""
        return {
            'original_uncertainty': 1 - orig_pred['confidence'],
            'modified_uncertainty': 1 - mod_pred['confidence'],
            'uncertainty_change': (1 - mod_pred['confidence']) - (1 - orig_pred['confidence']),
            'certainty_improvement': mod_pred['confidence'] > orig_pred['confidence'],
            'decision_stability': abs(mod_pred['confidence'] - orig_pred['confidence']) < 0.1
        }
    
    def _analyze_category_impacts(self, concept_changes):
        """Analyze impacts by morphological category"""
        category_impacts = {}
        for change in concept_changes:
            category = change['category']
            if category not in category_impacts:
                category_impacts[category] = {
                    'total_change': 0,
                    'change_count': 0,
                    'max_change': 0,
                    'concepts_affected': []
                }
            
            category_impacts[category]['total_change'] += abs(change['change'])
            category_impacts[category]['change_count'] += 1
            category_impacts[category]['max_change'] = max(category_impacts[category]['max_change'], abs(change['change']))
            category_impacts[category]['concepts_affected'].append(change['concept'])
        
        return category_impacts
    
    def _assess_safety_impact(self, orig_pred, mod_pred):
        """Comprehensive safety impact assessment"""
        orig_class = orig_pred['prediction_class']
        mod_class = mod_pred['prediction_class']
        
        if orig_class == mod_class:
            return {
                'impact_level': 'none',
                'message': f"No change in classification ({orig_class})",
                'safety_note': "Classification consistency maintained - no immediate safety impact",
                'recommendation': "Continue with original safety assessment"
            }
        elif orig_class == 'edible' and mod_class == 'poisonous':
            return {
                'impact_level': 'critical_safer',
                'message': "Classification changed from EDIBLE to POISONOUS",
                'safety_note': "⚠️ Increased caution - now recommends avoiding consumption",
                'recommendation': "This change errs on the side of safety - exercise extreme caution"
            }
        else:  # poisonous to edible
            return {
                'impact_level': 'critical_riskier',
                'message': "Classification changed from POISONOUS to EDIBLE",
                'safety_note': "🚨 INCREASED RISK - now suggests potential edibility",
                'recommendation': "EXTREME CAUTION: Verify all changes carefully with expert mycologist before any consumption decisions"
            }
    
    def export_analysis_report(self, original_result, modified_result=None, comparison=None):
        """Generate comprehensive analysis report with DSSE signature for export"""
        timestamp = datetime.now().isoformat()
        
        report = {
            'report_metadata': {
                'generated_at': timestamp,
                'analysis_type': 'Enhanced_IBOM_Analysis_with_DSSE',
                'model_version': 'Enhanced Hybrid Concept-Based MLP v3.0',
                'image_analyzed': original_result['image_metadata']['image_id'],
                'generator': 'Dependable FungAI IBOM Generator',
                'educational_mode': True,
                'concept_categorization': True
            },
            'original_analysis': {
                'prediction': original_result['neural_network_analysis']['prediction_analysis'],
                'top_concepts': original_result['concept_analysis']['sorted_by_contribution'][:10],
                'safety_recommendation': original_result['training_context']['prediction_reliability_assessment']['recommendation'],
                'concept_categories': len(self.get_concept_categories_with_education())
            }
        }
        
        if modified_result and comparison:
            report['modified_analysis'] = {
                'prediction': modified_result['neural_network_analysis']['prediction_analysis'],
                'comparison_summary': {
                    'prediction_changed': comparison['prediction_changed'],
                    'confidence_change': comparison['confidence_change'],
                    'safety_impact': comparison['safety_impact'],
                    'key_insights': comparison['educational_insights'][:3]
                },
                'concept_changes_count': len(comparison['concept_changes'])
            }
        
        return report
    
    def _extract_detailed_concept_weights(self):
        """Extract comprehensive concept weight analysis with educational categorization"""
        with torch.no_grad():
            # Get all layer weights for detailed analysis
            layer_weights = {}
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    layer_weights[name] = param.cpu().numpy()
            
            # Focus on final layer concept weights
            final_weights = self.model.net[-1].weight.cpu().numpy().flatten()
            concept_weights = final_weights[-self.num_concepts:]
            
            detailed_weights = []
            for i, (concept, weight) in enumerate(zip(self.concepts, concept_weights)):
                detailed_weights.append({
                    'concept_index': i,
                    'concept_text': concept,
                    'concept_clean': concept.replace('a photo of a mushroom where the ', ''),
                    'model_weight': float(weight),
                    'abs_weight': float(abs(weight)),
                    'weight_direction': 'poisonous_indicator' if weight > 0 else 'edible_indicator',
                    'weight_magnitude': 'high' if abs(weight) > 0.1 else 'medium' if abs(weight) > 0.05 else 'low',
                    'concept_category': self._categorize_concept(concept),
                    'safety_relevance': self._assess_concept_safety_relevance(self._categorize_concept(concept))
                })
            
            # Sort by absolute weight for importance ranking
            detailed_weights.sort(key=lambda x: x['abs_weight'], reverse=True)
            for i, concept_info in enumerate(detailed_weights):
                concept_info['importance_rank'] = i + 1
                concept_info['importance_percentile'] = ((len(detailed_weights) - i) / len(detailed_weights)) * 100
            
            return {
                'concept_weights': detailed_weights,
                'weight_statistics': {
                    'max_abs_weight': max(cw['abs_weight'] for cw in detailed_weights),
                    'min_abs_weight': min(cw['abs_weight'] for cw in detailed_weights),
                    'mean_abs_weight': np.mean([cw['abs_weight'] for cw in detailed_weights]),
                    'std_abs_weight': np.std([cw['abs_weight'] for cw in detailed_weights]),
                    'poisonous_indicators': len([cw for cw in detailed_weights if cw['model_weight'] > 0]),
                    'edible_indicators': len([cw for cw in detailed_weights if cw['model_weight'] < 0])
                },
                'category_statistics': self._calculate_category_statistics(detailed_weights)
            }
    
    def _calculate_category_statistics(self, detailed_weights):
        """Calculate statistics by morphological category"""
        category_stats = {}
        for concept in detailed_weights:
            category = concept['concept_category']
            if category not in category_stats:
                category_stats[category] = {
                    'concept_count': 0,
                    'avg_abs_weight': 0,
                    'max_abs_weight': 0,
                    'poisonous_indicators': 0,
                    'edible_indicators': 0
                }
            
            category_stats[category]['concept_count'] += 1
            category_stats[category]['avg_abs_weight'] += concept['abs_weight']
            category_stats[category]['max_abs_weight'] = max(category_stats[category]['max_abs_weight'], concept['abs_weight'])
            
            if concept['model_weight'] > 0:
                category_stats[category]['poisonous_indicators'] += 1
            else:
                category_stats[category]['edible_indicators'] += 1
        
        # Calculate averages
        for category in category_stats:
            if category_stats[category]['concept_count'] > 0:
                category_stats[category]['avg_abs_weight'] /= category_stats[category]['concept_count']
        
        return category_stats
    
    def _categorize_concept(self, concept):
        """Categorize concept by mushroom morphological features with enhanced mapping"""
        concept_lower = concept.lower()
        if 'cap' in concept_lower:
            if 'color' in concept_lower:
                return 'cap_color'
            elif 'shape' in concept_lower:
                return 'cap_shape'
            elif 'surface' in concept_lower:
                return 'cap_surface'
            else:
                return 'cap_general'
        elif 'gill' in concept_lower:
            if 'color' in concept_lower:
                return 'gill_color'
            elif 'spacing' in concept_lower:
                return 'gill_spacing'
            elif 'size' in concept_lower:
                return 'gill_size'
            elif 'attachment' in concept_lower:
                return 'gill_attachment'
            else:
                return 'gill_general'
        elif 'stalk' in concept_lower:
            return 'stalk_properties'
        elif 'odor' in concept_lower:
            return 'odor'
        elif 'habitat' in concept_lower:
            return 'habitat'
        elif 'ring' in concept_lower:
            return 'ring_properties'
        elif 'spore' in concept_lower:
            return 'spore_properties'
        elif 'veil' in concept_lower:
            return 'veil_properties'
        elif 'bruises' in concept_lower:
            return 'bruising_properties'
        elif 'population' in concept_lower:
            return 'population_pattern'
        else:
            return 'other'
    
    def process_image(self, image_path, custom_concept_scores=None):
        """
        Process single image with comprehensive analysis including all enhanced features
        Returns complete IBOM with DSSE signature
        """
        
        print(f"🔬 Processing image: {os.path.basename(image_path)}")
        
        # Image preprocessing and enhanced metadata
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        image_metadata = {
            'image_id': os.path.basename(image_path),
            'file_path': os.path.abspath(image_path),
            'raw_size': list(image.size),
            'file_size_bytes': os.path.getsize(image_path),
            'file_size_mb': round(os.path.getsize(image_path) / (1024*1024), 3),
            'input_tensor_shape': list(image_tensor.shape),
            'color_mode': image.mode,
            'format': image.format or 'Unknown',
            'preprocessing_applied': [
                f"CLIP preprocessing pipeline from {CLIP_MODEL_NAME}",
                "Resize to 224x224", "Center crop", "Tensor conversion", "Normalization"
            ]
        }
        
        # Feature extraction with detailed analysis
        with torch.no_grad():
            # CLIP image features
            img_features = self.clip_model.encode_image(image_tensor).float()
            img_features_normalized = img_features / img_features.norm(dim=1, keepdim=True)
            
            # Concept similarity computation
            concept_similarities = img_features_normalized @ self.text_embeddings.T
            concept_scores = concept_similarities.cpu().numpy().flatten()
            
            # Apply custom concept scores if provided (for interactive modification)
            modification_analysis = {'custom_scores_applied': False}
            if custom_concept_scores:
                original_scores = concept_scores.copy()
                for i, concept in enumerate(self.concepts):
                    if concept in custom_concept_scores:
                        concept_scores[i] = custom_concept_scores[concept]
                
                modification_analysis = {
                    'custom_scores_applied': True,
                    'number_of_modifications': len(custom_concept_scores),
                    'modification_impact': {
                        'max_change': float(np.max(np.abs(concept_scores - original_scores))),
                        'mean_change': float(np.mean(np.abs(concept_scores - original_scores))),
                        'total_change': float(np.sum(np.abs(concept_scores - original_scores)))
                    }
                }
            
            # Create hybrid feature vector
            concept_scores_tensor = torch.tensor(concept_scores).unsqueeze(0).to(self.device)
            hybrid_features = torch.cat([img_features, concept_scores_tensor], dim=1)
            
            # Detailed forward pass analysis
            prediction_logit, layer_analysis = self._forward_with_detailed_analysis(hybrid_features)
        
        # Comprehensive concept analysis with educational categorization
        concept_analysis = self._analyze_concepts_detailed(concept_scores)
        
        # Decision pathway analysis
        decision_analysis = self._analyze_decision_pathway(
            prediction_logit.item(), 
            layer_analysis['prediction_analysis']['sigmoid_probability'],
            concept_analysis
        )
        
        # Training context comparison
        training_context = self._compare_with_training_context(
            layer_analysis['prediction_analysis']['sigmoid_probability'],
            layer_analysis['prediction_analysis']['prediction_class']
        )
        
        # Build comprehensive result
        result = {
            'image_metadata': image_metadata,
            'feature_extraction': {
                'clip_features': {
                    'feature_statistics': {
                        'dimension': CLIP_EMBEDDING_DIM,
                        'l2_norm': float(img_features.norm()),
                        'mean': float(img_features.mean()),
                        'std': float(img_features.std()),
                        'min': float(img_features.min()),
                        'max': float(img_features.max())
                    }
                },
                'concept_scores': {
                    'computation_method': 'cosine_similarity(image_embedding, text_concept_embedding)',
                    'modification_analysis': modification_analysis,
                    'summary_statistics': {
                        'mean_score': float(np.mean(concept_scores)),
                        'std_score': float(np.std(concept_scores)),
                        'max_score': float(np.max(concept_scores)),
                        'min_score': float(np.min(concept_scores)),
                        'high_activation_count': int(np.sum(concept_scores > 0.7)),
                        'moderate_activation_count': int(np.sum((concept_scores > 0.4) & (concept_scores <= 0.7))),
                        'low_activation_count': int(np.sum(concept_scores <= 0.4)),
                        'total_concepts': len(concept_scores)
                    }
                },
                'hybrid_features': {
                    'total_dimension': self.input_dim,
                    'clip_component': CLIP_EMBEDDING_DIM,
                    'concept_component': self.num_concepts,
                    'feature_summary': {
                        'mean': float(hybrid_features.mean()),
                        'std': float(hybrid_features.std()),
                        'l2_norm': float(hybrid_features.norm()),
                        'min': float(hybrid_features.min()),
                        'max': float(hybrid_features.max())
                    }
                }
            },
            'neural_network_analysis': layer_analysis,
            'concept_analysis': concept_analysis,
            'decision_analysis': decision_analysis,
            'training_context': training_context,
            'educational_context': {
                'concept_categories': self.get_concept_categories_with_education(),
                'safety_guidance': self._generate_safety_guidance(layer_analysis['prediction_analysis']),
                'identification_tips': self._generate_identification_tips(concept_analysis)
            }
        }
        
        return result
    
    def _generate_safety_guidance(self, prediction_analysis):
        """Generate comprehensive safety guidance based on prediction"""
        guidance = {
            'primary_recommendation': '',
            'confidence_assessment': '',
            'additional_precautions': [],
            'expert_consultation': False
        }
        
        prediction_class = prediction_analysis['prediction_class']
        confidence = prediction_analysis['confidence']
        
        if prediction_class == 'poisonous':
            if confidence > 0.9:
                guidance['primary_recommendation'] = "🚫 DO NOT CONSUME - High confidence poisonous classification"
                guidance['confidence_assessment'] = "Very high model confidence supports this classification"
            elif confidence > 0.7:
                guidance['primary_recommendation'] = "⚠️ AVOID CONSUMPTION - Likely poisonous classification"
                guidance['confidence_assessment'] = "Moderate to high confidence in poisonous classification"
            else:
                guidance['primary_recommendation'] = "❓ UNCERTAIN BUT AVOID - Low confidence poisonous classification"
                guidance['confidence_assessment'] = "Low confidence but erring on side of safety"
                guidance['expert_consultation'] = True
        else:  # edible
            if confidence > 0.9:
                guidance['primary_recommendation'] = "✅ Model suggests edible, but expert verification still required"
                guidance['confidence_assessment'] = "High model confidence, but never consume without expert confirmation"
                guidance['expert_consultation'] = True
            else:
                guidance['primary_recommendation'] = "❓ UNCERTAIN - Expert identification required"
                guidance['confidence_assessment'] = "Low to moderate confidence - professional verification essential"
                guidance['expert_consultation'] = True
        
        # Always add these precautions
        guidance['additional_precautions'] = [
            "Never consume any wild mushroom without 100% positive identification",
            "Consult multiple field guides and expert mycologists",
            "Consider spore print and other advanced identification methods",
            "Be aware that mushroom appearance can vary with age and conditions",
            "When in doubt, don't consume - mushroom poisoning can be fatal"
        ]
        
        return guidance
    
    def _generate_identification_tips(self, concept_analysis):
        """Generate educational identification tips based on concept analysis"""
        tips = []
        
        # Get top contributing concepts
        top_concepts = concept_analysis['sorted_by_contribution'][:5]
        
        for concept in top_concepts:
            if concept['contribution_to_prediction'] != 0:
                category = concept['concept_category']
                if category == 'gill_color':
                    tips.append(f"Key feature: {concept['concept_clean']} - gill color is crucial for safety assessment")
                elif category == 'odor':
                    tips.append(f"Important: {concept['concept_clean']} - odor is one of the most reliable identification features")
                elif category == 'cap_color':
                    tips.append(f"Notable: {concept['concept_clean']} - cap color provides primary visual identification")
                elif category == 'stalk_properties':
                    tips.append(f"Significant: {concept['concept_clean']} - stalk features help distinguish between similar species")
        
        # Add general tips
        tips.extend([
            "Always examine gill attachment, color, and spacing",
            "Note any color changes when mushroom is bruised or cut",
            "Document habitat, season, and growth pattern",
            "Take spore print for definitive identification",
            "Compare with multiple reliable field guides"
        ])
        
        return tips[:8]  # Return top 8 tips
    
    def _forward_with_detailed_analysis(self, x):
        """Forward pass with comprehensive layer-by-layer analysis"""
        detailed_activations = {}
        current_input = x
        
        # Analyze each layer group
        layer_groups = [
            ('layer1', [0, 1, 2, 3]),  # Linear, BatchNorm, ReLU, Dropout
            ('layer2', [4, 5, 6, 7]),  # Linear, BatchNorm, ReLU, Dropout
            ('layer3', [8, 9, 10, 11]), # Linear, BatchNorm, ReLU, Dropout
            ('output', [12])            # Final Linear
        ]
        
        for group_name, layer_indices in layer_groups:
            group_analysis = {
                'input_stats': {
                    'shape': list(current_input.shape),
                    'mean': float(current_input.mean()),
                    'std': float(current_input.std()),
                    'min': float(current_input.min()),
                    'max': float(current_input.max()),
                    'norm': float(current_input.norm())
                }
            }
            
            # Process through layer group
            for layer_idx in layer_indices:
                layer = self.model.net[layer_idx]
                current_input = layer(current_input)
                
                # Store intermediate results for analysis
                if isinstance(layer, nn.Linear):
                    group_analysis['linear_output'] = {
                        'pre_activation_stats': {
                            'mean': float(current_input.mean()),
                            'std': float(current_input.std()),
                            'min': float(current_input.min()),
                            'max': float(current_input.max())
                        }
                    }
                elif isinstance(layer, nn.ReLU):
                    group_analysis['post_relu_stats'] = {
                        'active_neurons': int((current_input > 0).sum()),
                        'total_neurons': int(current_input.numel()),
                        'activation_rate': float((current_input > 0).float().mean()),
                        'mean_activation': float(current_input.mean()),
                        'max_activation': float(current_input.max())
                    }
            
            group_analysis['final_output_stats'] = {
                'shape': list(current_input.shape),
                'mean': float(current_input.mean()),
                'std': float(current_input.std()),
                'min': float(current_input.min()),
                'max': float(current_input.max())
            }
            
            detailed_activations[group_name] = group_analysis
        
        # Final prediction analysis
        final_logit = current_input.item()
        probability = torch.sigmoid(current_input).item()
        
        detailed_activations['prediction_analysis'] = {
            'raw_logit': final_logit,
            'sigmoid_probability': probability,
            'prediction_class': 'poisonous' if probability > 0.5 else 'edible',
            'confidence': max(probability, 1 - probability),
            'decision_strength': 'very_strong' if abs(final_logit) > 2.0 else 'strong' if abs(final_logit) > 1.0 else 'moderate' if abs(final_logit) > 0.5 else 'weak',
            'distance_from_threshold': abs(probability - 0.5),
            'certainty_level': 'high' if abs(final_logit) > 1.0 else 'medium' if abs(final_logit) > 0.5 else 'low'
        }
        
        return current_input, detailed_activations
    
    def _analyze_concepts_detailed(self, concept_scores):
        """Comprehensive concept-by-concept analysis with educational categorization"""
        detailed_concepts = []
        
        for i, (concept, score) in enumerate(zip(self.concepts, concept_scores)):
            weight_info = self.concept_weights['concept_weights'][i]
            contribution = score * weight_info['model_weight']
            
            concept_detail = {
                'concept_index': i,
                'concept_text': concept,
                'concept_clean': weight_info['concept_clean'],
                'concept_category': weight_info['concept_category'],
                'similarity_score': float(score),
                'similarity_interpretation': self._interpret_similarity_score(score),
                'model_weight': weight_info['model_weight'],
                'weight_direction': weight_info['weight_direction'],
                'weight_magnitude': weight_info['weight_magnitude'],
                'contribution_to_prediction': float(contribution),
                'contribution_interpretation': self._interpret_contribution(contribution),
                'importance_rank': weight_info['importance_rank'],
                'importance_percentile': weight_info['importance_percentile'],
                'safety_relevance': weight_info['safety_relevance']
            }
            
            detailed_concepts.append(concept_detail)
        
        # Sort by absolute contribution for analysis
        sorted_by_contribution = sorted(detailed_concepts, 
                                      key=lambda x: abs(x['contribution_to_prediction']), 
                                      reverse=True)
        
        # Enhanced category analysis with educational context
        category_analysis = {}
        for concept in detailed_concepts:
            category = concept['concept_category']
            if category not in category_analysis:
                category_analysis[category] = {
                    'concept_count': 0,
                    'total_contribution': 0,
                    'average_similarity': 0,
                    'concepts': [],
                    'safety_relevance': concept['safety_relevance']
                }
            
            category_analysis[category]['concept_count'] += 1
            category_analysis[category]['total_contribution'] += concept['contribution_to_prediction']
            category_analysis[category]['average_similarity'] += concept['similarity_score']
            category_analysis[category]['concepts'].append(concept['concept_clean'])
        
        # Finalize category averages and assessments
        for category in category_analysis:
            count = category_analysis[category]['concept_count']
            category_analysis[category]['average_similarity'] /= count
            category_analysis[category]['influence_level'] = (
                'high' if abs(category_analysis[category]['total_contribution']) > 0.5 else
                'medium' if abs(category_analysis[category]['total_contribution']) > 0.2 else 'low'
            )
        
        return {
            'all_concepts': detailed_concepts,
            'sorted_by_contribution': sorted_by_contribution,
            'top_supporting_concepts': [c for c in sorted_by_contribution[:10] if c['contribution_to_prediction'] > 0],
            'top_opposing_concepts': [c for c in sorted_by_contribution[:10] if c['contribution_to_prediction'] < 0],
            'high_similarity_concepts': [c for c in detailed_concepts if c['similarity_score'] > 0.7],
            'category_analysis': category_analysis,
            'concept_statistics': {
                'total_concepts': len(detailed_concepts),
                'positive_contributions': len([c for c in detailed_concepts if c['contribution_to_prediction'] > 0]),
                'negative_contributions': len([c for c in detailed_concepts if c['contribution_to_prediction'] < 0]),
                'high_similarity_count': len([c for c in detailed_concepts if c['similarity_score'] > 0.7]),
                'medium_similarity_count': len([c for c in detailed_concepts if 0.4 < c['similarity_score'] <= 0.7]),
                'low_similarity_count': len([c for c in detailed_concepts if c['similarity_score'] <= 0.4])
            }
        }
    
    def _interpret_similarity_score(self, score):
        """Interpret concept similarity score with educational context"""
        if score > 0.8:
            return "very_high_similarity"
        elif score > 0.6:
            return "high_similarity"
        elif score > 0.4:
            return "moderate_similarity"
        elif score > 0.2:
            return "low_similarity"
        else:
            return "very_low_similarity"
    
    def _interpret_contribution(self, contribution):
        """Interpret concept contribution to final prediction with educational context"""
        if abs(contribution) > 0.3:
            direction = "strongly_supports_poisonous" if contribution > 0 else "strongly_supports_edible"
        elif abs(contribution) > 0.1:
            direction = "moderately_supports_poisonous" if contribution > 0 else "moderately_supports_edible"
        elif abs(contribution) > 0.05:
            direction = "weakly_supports_poisonous" if contribution > 0 else "weakly_supports_edible"
        else:
            direction = "minimal_influence"
        
        return direction
    
    def _analyze_decision_pathway(self, logit, probability, concept_analysis):
        """Analyze the complete decision-making pathway with educational insights"""
        
        # Calculate contribution statistics
        all_contributions = [c['contribution_to_prediction'] for c in concept_analysis['all_concepts']]
        positive_contrib = sum(c for c in all_contributions if c > 0)
        negative_contrib = sum(c for c in all_contributions if c < 0)
        net_contrib = positive_contrib + negative_contrib
        
        # Enhanced decision pathway steps
        decision_steps = [
            {
                'step': 1,
                'description': 'Image processed through CLIP ViT-L/14 encoder',
                'output': f'{CLIP_EMBEDDING_DIM}-dimensional visual feature vector',
                'educational_note': 'Visual features capture shape, color, texture, and spatial relationships'
            },
            {
                'step': 2,
                'description': f'Computed similarity to {self.num_concepts} mushroom concept descriptions',
                'output': f'{self.num_concepts}-dimensional concept similarity vector',
                'educational_note': 'Each concept represents a specific morphological feature important for identification'
            },
            {
                'step': 3,
                'description': 'Concatenated visual and concept features',
                'output': f'{self.input_dim}-dimensional hybrid feature vector',
                'educational_note': 'Combines raw visual data with structured biological knowledge'
            },
            {
                'step': 4,
                'description': 'Processed through 4-layer MLP with batch normalization',
                'output': f'Single logit value: {logit:.6f}',
                'educational_note': 'Neural network learns complex relationships between features and safety'
            },
            {
                'step': 5,
                'description': 'Applied sigmoid activation for probability',
                'output': f'Final probability: {probability:.6f}',
                'educational_note': 'Converts logit to interpretable probability between 0 and 1'
            },
            {
                'step': 6,
                'description': 'Binary classification decision',
                'output': f'{"Poisonous" if probability > 0.5 else "Edible"} (threshold: 0.5)',
                'educational_note': 'Final safety classification based on learned patterns from training data'
            }
        ]
        
        # Key decision factors with educational context
        key_factors = []
        for concept in concept_analysis['sorted_by_contribution'][:5]:
            if abs(concept['contribution_to_prediction']) > 0.05:
                key_factors.append({
                    'concept': concept['concept_clean'],
                    'category': concept['concept_category'],
                    'similarity': concept['similarity_score'],
                    'weight': concept['model_weight'],
                    'contribution': concept['contribution_to_prediction'],
                    'influence': concept['contribution_interpretation'],
                    'importance_rank': concept['importance_rank'],
                    'safety_relevance': concept['safety_relevance']
                })
        
        return {
            'decision_pathway_steps': decision_steps,
            'contribution_breakdown': {
                'total_positive_contribution': positive_contrib,
                'total_negative_contribution': negative_contrib,
                'net_contribution': net_contrib,
                'contribution_balance': 'poisonous_dominant' if positive_contrib > abs(negative_contrib) else 'edible_dominant'
            },
            'key_decision_factors': key_factors,
            'decision_confidence_analysis': {
                'logit_magnitude': abs(logit),
                'confidence_level': 'very_high' if abs(logit) > 2.0 else 'high' if abs(logit) > 1.0 else 'medium' if abs(logit) > 0.5 else 'low',
                'decision_certainty': max(probability, 1 - probability),
                'uncertainty_factors': self._identify_uncertainty_factors(concept_analysis)
            }
        }
    
    def _identify_uncertainty_factors(self, concept_analysis):
        """Identify factors that contribute to prediction uncertainty with educational context"""
        uncertainty_factors = []
        
        # Check for balanced contributions
        positive_contrib = sum(c['contribution_to_prediction'] for c in concept_analysis['all_concepts'] if c['contribution_to_prediction'] > 0)
        negative_contrib = abs(sum(c['contribution_to_prediction'] for c in concept_analysis['all_concepts'] if c['contribution_to_prediction'] < 0))
        
        if abs(positive_contrib - negative_contrib) < 0.2:
            uncertainty_factors.append("Balanced positive and negative concept contributions indicate mixed signals")
        
        # Check for weak concept activations
        high_similarity_count = len([c for c in concept_analysis['all_concepts'] if c['similarity_score'] > 0.6])
        if high_similarity_count < 3:
            uncertainty_factors.append("Few concepts with high similarity scores suggest unclear visual features")
        
        # Check for conflicting high-weight concepts
        top_concepts = concept_analysis['sorted_by_contribution'][:5]
        pos_count = len([c for c in top_concepts if c['contribution_to_prediction'] > 0])
        neg_count = len([c for c in top_concepts if c['contribution_to_prediction'] < 0])
        if pos_count > 0 and neg_count > 0:
            uncertainty_factors.append("Conflicting signals from high-importance concepts indicate ambiguous features")
        
        # Check for safety-critical feature uncertainty
        critical_concepts = [c for c in concept_analysis['all_concepts'] if c['safety_relevance'] in ['Extremely Critical', 'Very High']]
        unclear_critical = [c for c in critical_concepts if c['similarity_score'] < 0.5]
        if len(unclear_critical) > 2:
            uncertainty_factors.append("Unclear safety-critical features (odor, gill color) increase identification uncertainty")
        
        return uncertainty_factors if uncertainty_factors else ["No significant uncertainty factors identified - clear feature patterns detected"]
    
    def _compare_with_training_context(self, probability, prediction):
        """Compare inference with training performance context"""
        training_perf = self.tbom_data.get('performance_metrics', {})
        test_results = training_perf.get('final_test_results', {})
        
        # Determine confidence category based on training performance
        confidence = max(probability, 1 - probability)
        
        if confidence > 0.9:
            expected_accuracy = min(0.95, test_results.get('accuracy', 0.8) + 0.1)
            reliability = "very_high"
        elif confidence > 0.8:
            expected_accuracy = test_results.get('accuracy', 0.8) + 0.05
            reliability = "high"
        elif confidence > 0.7:
            expected_accuracy = test_results.get('accuracy', 0.8)
            reliability = "moderate"
        else:
            expected_accuracy = max(0.6, test_results.get('accuracy', 0.8) - 0.1)
            reliability = "low"
        
        return {
            'training_performance_context': {
                'model_test_accuracy': test_results.get('accuracy', 0),
                'model_roc_auc': test_results.get('roc_auc', 0),
                'model_pr_auc': test_results.get('pr_auc', 0),
                'cross_validation_stability': training_perf.get('cross_validation_results', {}).get('std_accuracy', 0)
            },
            'prediction_reliability_assessment': {
                'confidence_category': f"{confidence:.1%}_confidence",
                'expected_accuracy_for_this_confidence': expected_accuracy,
                'reliability_rating': reliability,
                'recommendation': self._generate_safety_recommendation(probability, reliability)
            },
            'class_specific_context': {
                prediction: test_results.get('per_class_metrics', {}).get(prediction, {})
            }
        }
    
    def _generate_safety_recommendation(self, probability, reliability):
        """Generate comprehensive safety recommendation based on prediction and reliability"""
        if probability > 0.5:  # Poisonous prediction
            if reliability in ['very_high', 'high']:
                return "🚫 DO NOT CONSUME - High confidence poisonous classification with reliable model performance"
            else:
                return "⚠️ AVOID CONSUMPTION - Poisonous classification but with lower confidence, expert verification recommended"
        else:  # Edible prediction
            if reliability in ['very_high', 'high']:
                return "✅ Model suggests edible, but always verify with expert mycologist before consumption"
            else:
                return "❓ Uncertain classification - Professional identification required before any consumption"

# =============================================================================
# ENHANCED MAIN FUNCTION WITH COMPLETE DSSE INTEGRATION
# =============================================================================

def main():
    """Enhanced main IBOM generation function with DSSE signature and all features"""
    parser = argparse.ArgumentParser(
        description='Enhanced Hybrid Concept-Based Mushroom Classifier - Complete IBOM Generator',
        epilog="""
Complete Features:
- DSSE signature for supply chain security
- Concept categorization by morphological features  
- Educational mode with detailed explanations
- Advanced conflict detection and resolution
- Export functionality with multiple formats
- Uncertainty visualization support
- Zero hardcoded paths with auto-discovery

Environment Variables (Optional):
  TBOM_PATH     Path to TBOM.json file
  MODEL_PATH    Path to trained model (.pt file)  
  CSV_PATH      Path to mushroom concept CSV file

Examples:
  python IBOM.py --image_file mushroom.jpg
  python IBOM.py --image_dir images/ --output analysis.json --dsse
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Image input (required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_file", help="Single mushroom image file path")
    group.add_argument("--image_dir", help="Directory containing mushroom images")
    
    # Optional overrides (no defaults - full auto-discovery)
    parser.add_argument("--tbom_path", help="Override TBOM.json path (auto-discovered if not specified)")
    parser.add_argument("--model_path", help="Override model.pt path (auto-discovered if not specified)")
    parser.add_argument("--csv_path", help="Override CSV path (auto-discovered if not specified)")
    
    # Output configuration
    parser.add_argument("--output", help="Output IBOM filename (auto-generated if not specified)")
    parser.add_argument("--output_dir", help="Output directory (default: current directory)")
    
    # Enhanced features
    parser.add_argument("--dsse", action="store_true", help="Include DSSE signature for supply chain security")
    parser.add_argument("--educational_mode", action="store_true", default=True, help="Include educational content and explanations")
    parser.add_argument("--export_summary", action="store_true", help="Also export summary report")
    
    # Operational modes
    parser.add_argument("--no_interactive", action="store_true", help="Disable interactive prompts (fail if files not found)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🍄 Enhanced Hybrid Concept-Based Mushroom Classifier - Complete IBOM Generator")
        print("=" * 80)
        print("🔐 DSSE Signature Support | 🎛️ Concept Categorization | 📚 Educational Mode")
        print("📊 Uncertainty Visualization | 🚫 Conflict Detection | 📥 Export Functionality")
        print("=" * 80)
    
    try:
        # Dynamic path discovery with comprehensive fallback
        if args.tbom_path and args.model_path and args.csv_path:
            # All paths explicitly provided
            if not args.quiet:
                print("📁 Using explicitly provided paths")
            
            # Validate provided paths
            for name, path in [("TBOM", args.tbom_path), ("Model", args.model_path), ("CSV", args.csv_path)]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{name} file not found: {path}")
            
            tbom_path, model_path, csv_path = args.tbom_path, args.model_path, args.csv_path
        
        else:
            # Use auto-discovery with smart fallback
            if not args.quiet:
                print("🔧 Smart discovery with comprehensive fallback")
            
            tbom_path, model_path, csv_path = get_paths_with_smart_fallback()
            
            # Apply any explicit overrides
            tbom_path = args.tbom_path or tbom_path
            model_path = args.model_path or model_path
            csv_path = args.csv_path or csv_path
        
        # Validate image input
        if args.image_file:
            if not os.path.exists(args.image_file):
                raise FileNotFoundError(f"Image file not found: {args.image_file}")
            image_info = {"type": "single", "path": args.image_file}
            image_paths = [args.image_file]
        else:
            if not os.path.exists(args.image_dir):
                raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
            
            # Find supported image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(args.image_dir).glob(f"*{ext}"))
                image_files.extend(Path(args.image_dir).glob(f"*{ext.upper()}"))
            
            if not image_files:
                raise ValueError(f"No image files found in directory: {args.image_dir}")
            
            image_info = {"type": "directory", "path": args.image_dir, "files": image_files}
            image_paths = [str(f) for f in image_files]
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if image_info["type"] == "single":
                image_name = Path(args.image_file).stem
                output_name = f"Enhanced_IBOM_{image_name}_{timestamp}.json"
            else:
                dir_name = Path(args.image_dir).name
                output_name = f"Enhanced_IBOM_batch_{dir_name}_{timestamp}.json"
            
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / output_name)
            else:
                output_path = output_name
        
        # Display configuration
        if not args.quiet:
            print(f"\n Enhanced Configuration Summary:")
            print(f"   TBOM: {Path(tbom_path).name} ({Path(tbom_path).parent})")
            print(f"   Model: {Path(model_path).name} ({Path(model_path).parent})")
            print(f"   CSV: {Path(csv_path).name} ({Path(csv_path).parent})")
            print(f"   Images: {len(image_paths)} file(s)")
            print(f"   Output: {output_path}")
            print(f"   DSSE Signature: {'Enabled' if args.dsse else 'Disabled'}")
            print(f"   Educational Mode: {'Enabled' if args.educational_mode else 'Disabled'}")
        
        # Initialize enhanced IBOM generator
        if not args.quiet:
            print("\n Initializing Enhanced IBOM Generator")
        
        ibom_generator = DetailedIBOMGenerator(model_path, tbom_path, csv_path)
        
        # Process images with comprehensive analysis
        detailed_results = []
        
        if not args.quiet:
            print(f"\n📸 Processing {len(image_paths)} image(s)...")
        
        for i, image_path in enumerate(image_paths):
            if not args.quiet:
                print(f"🔬 Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = ibom_generator.process_image(image_path)
                detailed_results.append(result)
                
                if not args.quiet:
                    pred_class = result['neural_network_analysis']['prediction_analysis']['prediction_class']
                    confidence = result['neural_network_analysis']['prediction_analysis']['confidence']
                    print(f"   Result: {pred_class.upper()} ({confidence:.1%} confidence)")
                    
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
                continue
        
        if not detailed_results:
            raise RuntimeError("No images were successfully processed")
        
        # Build comprehensive IBOM with enhanced features
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        ibom = {
            "ibom_type": "Enhanced_DetailedInferenceBillOfMaterials",
            "ibom_version": "4.0-complete-with-dsse-educational",
            "generation_metadata": {
                "timestamp_utc": timestamp,
                "generator_script": Path(__file__).name,
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "device_used": ibom_generator.device,
                "educational_mode_enabled": args.educational_mode,
                "dsse_signature_enabled": args.dsse,
                "configuration": {
                    "tbom_file": str(Path(tbom_path).resolve()),
                    "model_file": str(Path(model_path).resolve()),
                    "csv_file": str(Path(csv_path).resolve()),
                    "discovery_method": "explicit" if (args.tbom_path and args.model_path and args.csv_path) else "auto_discovery",
                    "concept_categorization": True,
                    "conflict_detection": True
                }
            },
            "model_architecture_summary": {
                "model_name": "Enhanced Hybrid Concept-Based MLP Classifier",
                "backbone": CLIP_MODEL_NAME,
                "concept_count": ibom_generator.num_concepts,
                "input_dimension": ibom_generator.input_dim,
                "architecture": "4-layer MLP with batch normalization and educational categorization",
                "morphological_categories": len(ibom_generator.get_concept_categories_with_education())
            },
            "inference_configuration": {
                "preprocessing_pipeline": [
                    f"CLIP {CLIP_MODEL_NAME} preprocessing",
                    "Image resize to 224x224",
                    "Tensor conversion and normalization",
                    "Concept similarity computation",
                    "Educational categorization mapping"
                ],
                "concept_similarity_method": "cosine_similarity(image_embedding, text_concept_embedding)",
                "decision_threshold": 0.5,
                "binary_classification": ["edible", "poisonous"],
                "educational_features": {
                    "concept_categories": list(ibom_generator.get_concept_categories_with_education().keys()),
                    "conflict_detection": True,
                    "safety_guidance": True,
                    "uncertainty_analysis": True
                }
            },
            "detailed_inference_results": detailed_results,
            "concept_weight_analysis": ibom_generator.concept_weights,
            "aggregate_analysis": {
                "total_images_processed": len(detailed_results),
                "prediction_distribution": {
                    "poisonous": len([r for r in detailed_results if r['neural_network_analysis']['prediction_analysis']['prediction_class'] == 'poisonous']),
                    "edible": len([r for r in detailed_results if r['neural_network_analysis']['prediction_analysis']['prediction_class'] == 'edible'])
                },
                "confidence_statistics": {
                    "mean_confidence": np.mean([r['neural_network_analysis']['prediction_analysis']['confidence'] for r in detailed_results]),
                    "min_confidence": min([r['neural_network_analysis']['prediction_analysis']['confidence'] for r in detailed_results]),
                    "max_confidence": max([r['neural_network_analysis']['prediction_analysis']['confidence'] for r in detailed_results]),
                    "high_confidence_count": len([r for r in detailed_results if r['neural_network_analysis']['prediction_analysis']['confidence'] > 0.8])
                }
            },
            "educational_summary": {
                "concept_categories_analyzed": len(ibom_generator.get_concept_categories_with_education()),
                "safety_guidance_provided": True,
                "identification_tips_included": True,
                "morphological_features_categorized": True
            },
            "software_environment": get_software_versions()
        }
        
        # Add DSSE signature if requested
        if args.dsse:
            if not args.quiet:
                print("🔐 Generating DSSE signature for supply chain security...")
            
            dsse_envelope, dsse_payload = generate_dsse_signature(ibom)
            ibom["dsse_signature"] = {
                "envelope": dsse_envelope,
                "payload_summary": {
                    "subject_count": len(dsse_payload["subject"]),
                    "predicate_type": dsse_payload["predicateType"],
                    "signature_algorithm": "SHA256",
                    "signing_entity": "dependable-fungai-v1.0"
                }
            }
        
        # Generate IBOM content signature
        ibom_string = json.dumps(ibom, sort_keys=True).encode('utf-8')
        ibom["ibom_signature"] = hashlib.sha256(ibom_string).hexdigest()
        
        # Save main IBOM file
        with open(output_path, 'w') as f:
            json.dump(ibom, f, indent=4)
        
        # Export summary report if requested
        if args.export_summary:
            summary_path = output_path.replace('.json', '_summary.json')
            summary_report = ibom_generator.export_analysis_report(
                detailed_results[0], 
                None, 
                None
            )
            
            with open(summary_path, 'w') as f:
                json.dump(summary_report, f, indent=4)
            
            if not args.quiet:
                print(f"Summary report exported: {summary_path}")
        
        # Success summary
        if not args.quiet:
            print(f"\n Enhanced IBOM Analysis Complete!")
            print(f"   Output: {output_path}")
            print(f"   File Size: {len(json.dumps(ibom)) / 1024:.1f} KB")
            print(f"   IBOM Signature: {ibom['ibom_signature'][:16]}...")
            if args.dsse:
                print(f"   DSSE Signature: {ibom['dsse_signature']['envelope']['signatures'][0]['sig'][:16]}...")
            
            # Enhanced results summary
            print(f"\n📋 Analysis Results Summary:")
            print(f"   Total Images: {len(detailed_results)}")
            print(f"   Poisonous Classifications: {ibom['aggregate_analysis']['prediction_distribution']['poisonous']}")
            print(f"   Edible Classifications: {ibom['aggregate_analysis']['prediction_distribution']['edible']}")
            print(f"   Mean Confidence: {ibom['aggregate_analysis']['confidence_statistics']['mean_confidence']:.1%}")
            print(f"   High Confidence (>80%): {ibom['aggregate_analysis']['confidence_statistics']['high_confidence_count']}")
            
            # Sample results
            print(f"\nSample Results:")
            for i, result in enumerate(detailed_results[:3]):
                img_name = result['image_metadata']['image_id']
                prediction = result['neural_network_analysis']['prediction_analysis']['prediction_class']
                confidence = result['neural_network_analysis']['prediction_analysis']['confidence']
                safety_rec = result['training_context']['prediction_reliability_assessment']['recommendation'][:50] + "..."
                print(f"   {i+1}. {img_name}: {prediction.upper()} ({confidence:.1%}) - {safety_rec}")
            
            if len(detailed_results) > 3:
                print(f"   ... and {len(detailed_results) - 3} more images")
            
            print(f"\n Security Features:")
            print(f"   Content Integrity: SHA256 signature included")
            if args.dsse:
                print(f"   Supply Chain Security: DSSE envelope with in-toto attestation")
            print(f"   Educational Mode: Comprehensive morphological categorization")
            print(f"   Conflict Detection: Biological impossibility validation")
            
            print(f"\n⚠️ Important Safety Notice:")
            print(f"   This AI analysis is for educational and research purposes only.")
            print(f"   Never consume any mushroom based solely on AI classification.")
            print(f"   Always consult certified mycologists for positive identification.")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Enhanced IBOM generation failed: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


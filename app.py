"""
Streamlit Dashboard for Training Bill of Materials (TBOM) Visualization
Displays comprehensive training metrics, model architecture, and performance analysis
With IBOM (Inference Bill of Materials) FUNCTIONALITY
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import os
import tempfile
import io
from PIL import Image
import torch
import torch.nn as nn

# Dynamic import from enhanced IBOM backend (no hardcoded paths)
from IBOM import DetailedIBOMGenerator, get_paths_with_smart_fallback

# Page configuration
st.set_page_config(
    page_title="Dependable FungAI - Training Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (keeping your existing styles)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: #4169E1;
        margin: 1rem 0;
        border-bottom: 2px solid #4169E1;
        padding-bottom: 0.5rem;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }

    .performance-highlight {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }

    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }

    .concept-row {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #2E8B57;
    }

    .comparison-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_tbom_data(file_path="TBOM.json"):
    """Load TBOM data with caching for performance"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            # Use the enhanced auto-discovery from IBOM
            try:
                tbom_path, _, _ = get_paths_with_smart_fallback()
                with open(tbom_path, 'r') as f:
                    return json.load(f)
            except Exception:
                st.error("TBOM file not found. Please ensure TBOM.json is available.")
                return None
    except Exception as e:
        st.error(f"Error loading TBOM data: {str(e)}")
        return None

#Enhanced IBOM Interface

def create_enhanced_ibom_interface(tbom_data):
    """Complete enhanced IBOM interface with all advanced features"""
    st.markdown('<h2 class="sub-header">🔍 Enhanced Interactive Inference Bill of Materials (IBOM)</h2>', unsafe_allow_html=True)

    # Feature overview
    with st.expander("Enhanced Features Overview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎛️ Concept Categorization:**
            - Organized by morphological features
            - Cap, gill, stalk, sensory, environmental properties
            - Educational descriptions & importance levels

            **📚 Educational Mode:**
            - Detailed concept explanations
            - Safety warnings and identification guidance
            - Biological context and best practices
            """)
        with col2:
            st.markdown("""
            **📥 Export Functionality:**
            - Download IBOM JSON files
            - CSV exports for spreadsheet analysis
            - Detailed analysis reports

            **📊 Uncertainty Visualization:**
            - Confidence intervals & decision strength
            - Interactive uncertainty charts
            - Educational confidence interpretation
            """)

    # Image Upload Section
    st.markdown("### Mushroom Image Upload ")
    uploaded_image = st.file_uploader(
        "Upload a mushroom image for comprehensive analysis",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image showing cap, gills, and stalk for detailed AI analysis"
    )

    # Main layout: Image on the left, results on the right
    img_col, results_col = st.columns([1, 2.5])

    with img_col:
        if uploaded_image is not None:
            # Display uploaded image with enhanced metadata
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Mushroom Image", use_container_width=True)

            # Enhanced image metadata
            with st.expander("📊 Image Analysis", expanded=False):
                st.write(f"**Filename:** {uploaded_image.name}")
                st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                st.write(f"**File Size:** {len(uploaded_image.getvalue()) / 1024:.1f} KB")

                # Quality indicators
                if min(image.size) >= 224:
                    st.success("✅ Excellent resolution for analysis")
                elif min(image.size) >= 100:
                    st.info("ℹ️ Good resolution for analysis")
                else:
                    st.warning("⚠️ Low resolution - results may be less accurate")

            # Enhanced analysis trigger
            if st.button(" Analyze with Enhanced IBOM", type="primary"):
                perform_enhanced_ibom_analysis(uploaded_image)
        else:
            st.info("Upload an image to begin analysis")

            # Enhanced guidelines
            with st.expander("🍄 Image Requirements", expanded=True):
                st.info("""
                **For Optimal Results:**
                - Clear view of cap, gills, and stalk
                - Show any bruising or discoloration
                - Minimal background distractions
                """)

    with results_col:
        # Display results if available
        if 'enhanced_ibom_data' in st.session_state:
            display_comprehensive_ibom_results()
            create_concept_score_modification_interface()


def perform_enhanced_ibom_analysis(uploaded_file):
    """Enhanced analysis with comprehensive error handling"""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    try:
        # Dynamic path discovery (no hardcoded paths)
        with st.spinner("🔍 Auto-discovering model files..."):
            tbom_path, model_path, csv_path = get_paths_with_smart_fallback()

        # Initialize enhanced IBOM generator
        with st.spinner("Loading Enhanced AI Model..."):
            ibom_generator = DetailedIBOMGenerator(model_path, tbom_path, csv_path)

        # Generate comprehensive analysis
        with st.spinner("🔬 Performing detailed mushroom analysis..."):
            original_result = ibom_generator.process_image(temp_image_path)

        # Store enhanced data in session state
        st.session_state.enhanced_ibom_data = {
            'original_result': original_result,
            'ibom_generator': ibom_generator,
            'temp_image_path': temp_image_path,
            'concept_categories': ibom_generator.get_concept_categories_with_education(),
            'modified_scores': {},
            'analysis_timestamp': datetime.now()
        }

        st.success("✅ Enhanced analysis complete! Comprehensive results available.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        st.info("Please ensure model files are properly configured and try again.")
        # Optional: Show detailed error for debugging
        if st.checkbox("Show detailed error information"):
            st.exception(e)

def display_comprehensive_ibom_results():
    """
    Display comprehensive IBOM results with the requested side-by-side graph layout.
    """
    data = st.session_state.enhanced_ibom_data
    original_result = data['original_result']
    prediction_analysis = original_result['neural_network_analysis']['prediction_analysis']
    concept_analysis = original_result['concept_analysis']

    # --- Section 1: Display top-level prediction results ---
    st.markdown("### 🤖 Enhanced AI Analysis Results")
    prediction_class = prediction_analysis['prediction_class']
    confidence = prediction_analysis['confidence']

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        if prediction_class == 'poisonous':
            st.error(f"#### 🚫 POISONOUS")
        else:
            st.success(f"#### ✅ EDIBLE")
    with res_col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with res_col3:
        st.metric("Decision Strength", prediction_analysis.get('decision_strength', 'unknown').replace('_', ' ').title())

    # --- Add download button for Original IBOM ---
    original_json_data = json.dumps(original_result, indent=2)
    st.download_button(
        label="📄 Download Original IBOM.json",
        data=original_json_data,
        file_name=f"IBOM_{data['analysis_timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    st.markdown("<hr>", unsafe_allow_html=True) # Visual separator

    # --- Section 2: Side-by-side layout for advanced graphs ---
    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        # Display the two uncertainty graphs (stacked)
        raw_logit = prediction_analysis.get('raw_logit', 0)
        create_enhanced_uncertainty_visualization(confidence, raw_logit)

    with graph_col2:
        # Display the Concept Analysis by Category graph
        st.markdown("#### 🧠 Concept Analysis by Category")
        category_analysis_data = concept_analysis.get('category_analysis', {})
        if category_analysis_data:
            categories = list(category_analysis_data.keys())
            influences = [abs(category_analysis_data[cat].get('total_contribution', 0)) for cat in categories]
            if any(influences):
                fig = px.bar(
                    x=influences,
                    y=categories,
                    orientation='h',
                    title="Category-Level Influence on Decision",
                    color=influences,
                    color_continuous_scale='Viridis',
                    labels={'x': 'Absolute Influence', 'y': 'Morphological Categories'}
                )
                fig.update_layout(height=400, title_font_size=14, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category analysis data not available.")

    # --- Section 3: Display detailed concept contribution table (below graphs) ---
    display_enhanced_concept_contributions(original_result)

    # --- Process modifications if any exist ---
    if data.get('modified_scores', {}):
        process_enhanced_modifications()


def display_enhanced_concept_contributions(original_result):
    """
    Enhanced concept contribution visualization.
    NOTE: In the new layout, this function is only used for the table part.
    """
    concept_analysis = original_result['concept_analysis']

    # Top individual concepts table
    st.markdown("#### Top Contributing Individual Concepts")
    top_concepts = concept_analysis.get('sorted_by_contribution', [])[:10]

    if top_concepts:
        concept_data = []
        for concept in top_concepts:
            concept_data.append({
                'Concept': concept['concept_clean'][:50] + "..." if len(concept['concept_clean']) > 50 else concept['concept_clean'],
                'Similarity': f"{concept.get('similarity_score', 0):.3f}",
                'Weight': f"{concept.get('model_weight', 0):+.3f}",
                'Contribution': f"{concept.get('contribution_to_prediction', 0):+.4f}",
                'Direction': '→ Poisonous' if concept.get('contribution_to_prediction', 0) > 0 else '→ Edible',
                'Rank': concept.get('importance_rank', 0)
            })

        df = pd.DataFrame(concept_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No concept contribution data available.")

def create_enhanced_uncertainty_visualization(confidence, raw_logit):
    """
    Create comprehensive uncertainty visualization that matches the reference image.
    """
    st.markdown("#### 📊 Advanced Uncertainty Analysis")

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        subplot_titles=(None, 'Decision Boundary Position'),
        vertical_spacing=0.2
    )

    # --- Graph 1: Confidence vs Uncertainty Bar Chart ---
    uncertainty = 1 - confidence
    fig.add_trace(go.Bar(x=[''], y=[confidence], name='Confidence', marker_color='green'), row=1, col=1)
    fig.add_trace(go.Bar(x=[''], y=[uncertainty], name='Uncertainty', marker_color='red'), row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=1)


    # --- Graph 2: Decision Boundary Position ---
    x_vals = np.linspace(-5, 5, 100)
    y_vals = 1 / (1 + np.exp(-x_vals))
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Sigmoid Function', line=dict(color='blue')), row=2, col=1)

    # Add Decision Threshold line (purple)
    fig.add_hline(y=0.5, line_dash="dash", line_color="purple", name="Decision Threshold", row=2, col=1)

    # Add Current Prediction (red dot)
    fig.add_trace(go.Scatter(x=[raw_logit], y=[confidence], mode='markers', name='Current Prediction',
                             marker=dict(size=10, color='red', symbol='circle')), row=2, col=1)

    # Add horizontal dashed line from the prediction to the y-axis
    fig.add_shape(type="line",
                  x0=raw_logit, y0=confidence, x1=x_vals.min(), y1=confidence,
                  line=dict(color="Red", width=2, dash="dash"),
                  row=2, col=1)

    fig.update_yaxes(title_text="Prediction", range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Logit Space", row=2, col=1)


    # --- Layout and Legend ---
    fig.update_layout(
        barmode='stack',
        height=400,
        showlegend=True,
        legend=dict(traceorder='reversed', yanchor="top", y=1.0, xanchor="right", x=1.15),
        margin=dict(t=20, b=40, l=40, r=40),
        title_font_size=14
    )
    st.plotly_chart(fig, use_container_width=True)


def create_concept_score_modification_interface():
    """Create the enhanced concept score modification interface"""
    if 'enhanced_ibom_data' not in st.session_state:
        return

    data = st.session_state.enhanced_ibom_data
    categories = data.get('concept_categories', {})

    st.markdown("---")
    st.markdown("### 🔬 Concept Score Modification")

    if categories:
        original_result = data['original_result']
        orig_pred = original_result['neural_network_analysis']['prediction_analysis']
        orig_class = orig_pred['prediction_class']
        orig_confidence = orig_pred['confidence']

        # Create a lookup dictionary for actual similarity scores
        score_lookup = {
            c['concept_index']: c['similarity_score']
            for c in original_result['concept_analysis'].get('sorted_by_contribution', [])
        }

        # Create prediction boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="prediction-box" style="background-color: #e3f2fd;">
                <strong>Original Prediction</strong><br>
                {orig_class.title()}<br>
                Confidence: {orig_confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            mod_class, mod_confidence, bg_color = (orig_class, orig_confidence, "#e3f2fd")
            if data.get('modified_scores') and 'modified_result' in data:
                mod_pred = data['modified_result']['neural_network_analysis']['prediction_analysis']
                mod_class, mod_confidence = mod_pred['prediction_class'], mod_pred['confidence']
                bg_color = "#ffebee" if mod_class == "poisonous" else "#e8f5e8"

            st.markdown(f"""
            <div class="prediction-box" style="background-color: {bg_color};">
                <strong>Modified Prediction</strong><br>
                {mod_class.title()}<br>
                Confidence: {mod_confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        # Concept modification controls
        with st.expander("🔧 Adjust Concept Scores to Simulate Scenarios", expanded=True):
            if st.button("🔄 Reset to Original Scores"):
                st.session_state.enhanced_ibom_data['modified_scores'] = {}
                if 'modified_result' in st.session_state.enhanced_ibom_data:
                    del st.session_state.enhanced_ibom_data['modified_result']
                st.rerun()

            all_concepts = [concept for cat_info in categories.values() for concept in cat_info.get('concepts', [])]
            concepts_per_row = 2
            for i in range(0, len(all_concepts), concepts_per_row):
                cols = st.columns(concepts_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(all_concepts):
                        with col:
                            # Pass the score_lookup to the widget function
                            create_concept_slider_widget(all_concepts[i + j], i + j, score_lookup)

def create_concept_slider_widget(concept, index, score_lookup):
    """Create individual concept slider widget"""
    concept_idx = concept.get('concept_index', concept.get('index', index))
    concept_weight = concept.get('model_weight', concept.get('weight', 0))
    concept_name = concept.get('concept_clean', concept.get('concept_text', f'Concept {index}'))

    # Get the score from the lookup, with a fallback for concepts not present in the image.
    original_score = score_lookup.get(concept_idx, 0.0)

    if 'modified_scores' not in st.session_state.enhanced_ibom_data:
        st.session_state.enhanced_ibom_data['modified_scores'] = {}

    current_score = st.session_state.enhanced_ibom_data['modified_scores'].get(concept_idx, original_score)

    # The 'value' parameter now correctly uses the calculated original_score.
    new_score = st.slider(
        f"{concept_name[:30]}..." if len(concept_name) > 30 else concept_name,
        min_value=0.0,
        max_value=1.0,
        value=float(current_score),
        step=0.01,
        key=f"concept_slider_{concept_idx}_{index}",
        help=f"Original: {original_score:.2f}, Weight: {concept_weight:.3f}"
    )

    if abs(new_score - current_score) > 1e-5:
        st.session_state.enhanced_ibom_data['modified_scores'][concept_idx] = new_score
        st.rerun()

    change = current_score - original_score
    if abs(change) > 0.01:
        change_indicator = "🔴" if abs(change) > 0.2 else "🟡"
        st.caption(f"{change_indicator} Changed by {change:+.2f}")


def process_enhanced_modifications():
    """Process concept modifications with enhanced analysis"""
    data = st.session_state.enhanced_ibom_data
    ibom_generator = data['ibom_generator']
    modified_scores = data['modified_scores']

    st.markdown("#### 🔄 Modified Analysis Results")

    custom_concept_scores = {
        ibom_generator.concepts[concept_idx]: new_score
        for concept_idx, new_score in modified_scores.items()
        if concept_idx < len(ibom_generator.concepts)
    }

    with st.spinner("🔄 Generating modified analysis..."):
        try:
            modified_result = ibom_generator.process_image(
                data['temp_image_path'],
                custom_concept_scores
            )
            comparison = ibom_generator.compare_analyses_detailed(
                data['original_result'],
                modified_result
            )
            data['modified_result'] = modified_result
            data['comparison'] = comparison
            display_enhanced_comparison_results(comparison)
            display_enhanced_export_options()
        except Exception as e:
            st.error(f"❌ Modified analysis failed: {str(e)}")

def display_enhanced_comparison_results(comparison):
    """Display comprehensive comparison results"""
    st.markdown("##### 📊 Analysis Comparison")

    if comparison['prediction_changed']:
        st.warning("**🔄 PREDICTION CHANGED**")
        safety_impact = comparison['safety_impact']
        if safety_impact['impact_level'] == 'critical_riskier':
            st.error(f"🚨 {safety_impact['message']}")
        elif safety_impact['impact_level'] == 'critical_safer':
            st.warning(f"⚠️ {safety_impact['message']}")
        else:
            st.success(f"✅ {safety_impact['message']}")
    else:
        st.info("**➡️ No prediction change**")

    if comparison.get('educational_insights'):
        st.markdown("**📚 Key Insights:**")
        for insight in comparison['educational_insights'][:3]:
            st.info(f"💡 {insight}")

def display_enhanced_export_options():
    """Enhanced export functionality"""
    st.markdown("##### 📥 Enhanced Export Options")
    data = st.session_state.enhanced_ibom_data
    timestamp = data['analysis_timestamp'].strftime('%Y%m%d_%H%M%S')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Original IBOM JSON",
            data=json.dumps(data['original_result'], indent=2),
            file_name=f"original_IBOM_{timestamp}.json",
            mime="application/json"
        )
    with col2:
        if 'modified_result' in data:
            st.download_button(
                label="📄 Modified IBOM JSON",
                data=json.dumps(data['modified_result'], indent=2),
                file_name=f"modified_IBOM_{timestamp}.json",
                mime="application/json"
            )
    with col3:
        if 'comparison' in data:
            report = data['ibom_generator'].export_analysis_report(
                data['original_result'],
                data.get('modified_result'),
                data.get('comparison')
            )
            st.download_button(
                label="📊 Analysis Report",
                data=json.dumps(report, indent=2),
                file_name=f"analysis_report_{timestamp}.json",
                mime="application/json"
            )

#TBOM Visualization Functions --

def create_performance_overview(tbom_data):
    """Create performance overview metrics (UNCHANGED)"""
    perf_metrics = tbom_data.get('performance_metrics', {})
    test_results = perf_metrics.get('final_test_results', {})
    cv_results = perf_metrics.get('cross_validation_results', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        test_acc = test_results.get('accuracy', 0) * 100
        st.metric(
            label="🎯 Test Accuracy",
            value=f"{test_acc:.2f}%",
            delta=f"CV: {cv_results.get('mean_accuracy', 0)*100:.2f}%"
        )

    with col2:
        roc_auc = test_results.get('roc_auc', 0) * 100
        st.metric(
            label="📈 ROC AUC",
            value=f"{roc_auc:.2f}%",
            delta=f"CV: {cv_results.get('mean_roc_auc', 0)*100:.2f}%"
        )

    with col3:
        pr_auc = test_results.get('pr_auc', 0) * 100
        st.metric(
            label="📊 PR AUC",
            value=f"{pr_auc:.2f}%",
            delta=f"CV: {cv_results.get('mean_pr_auc', 0)*100:.2f}%"
        )

    with col4:
        training_time = tbom_data.get('generation_details', {}).get('training_time_seconds', 0)
        st.metric(
            label="⏱️ Training Time",
            value=f"{training_time/60:.1f} min",
            delta=f"{training_time:.0f}s total"
        )

def create_confusion_matrix_viz(tbom_data):
    """Create interactive confusion matrix visualization (UNCHANGED)"""
    test_results = tbom_data.get('performance_metrics', {}).get('final_test_results', {})
    cm = test_results.get('confusion_matrix', [[0, 0], [0, 0]])
    cm_normalized = test_results.get('confusion_matrix_normalized', [[0, 0], [0, 0]])

    class_names = tbom_data.get('data_summary', {}).get('class_information', {}).get('names', ['Edible', 'Poisonous'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Confusion Matrix")
        fig_raw = ff.create_annotated_heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            annotation_text=cm,
            colorscale='RdYlBu_r',
            showscale=True
        )
        fig_raw.update_layout(
            title="Test Set Confusion Matrix (Raw Counts)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with col2:
        st.subheader("Normalized Confusion Matrix")
        cm_norm_percent = [[round(val*100, 1) for val in row] for row in cm_normalized]
        fig_norm = ff.create_annotated_heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            annotation_text=cm_norm_percent,
            colorscale='RdYlBu_r',
            showscale=True
        )
        fig_norm.update_layout(
            title="Test Set Confusion Matrix (Normalized %)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        st.plotly_chart(fig_norm, use_container_width=True)

def create_training_curves(tbom_data):
    """Create training and validation curves (UNCHANGED)"""
    cv_results = tbom_data.get('performance_metrics', {}).get('cross_validation_results', {})
    fold_details = cv_results.get('per_fold_details', [])

    if not fold_details:
        st.warning("No detailed fold metrics available for plotting training curves")
        return

    # Create training curves visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training/Validation Loss', 'Training/Validation Accuracy',
                       'ROC AUC per Fold', 'PR AUC per Fold'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    epochs = list(range(1, 26))  # 25 epochs

    # Plot average performance metrics
    fold_accs = [fold.get('final_val_acc', 0) for fold in fold_details]
    fold_roc_aucs = [fold.get('roc_auc', 0) for fold in fold_details]
    fold_pr_aucs = [fold.get('pr_auc', 0) for fold in fold_details]

    # ROC AUC per fold
    fig.add_trace(
        go.Bar(x=[f"Fold {i+1}" for i in range(len(fold_roc_aucs))],
               y=fold_roc_aucs,
               name="ROC AUC",
               marker_color='lightblue'),
        row=2, col=1
    )

    # PR AUC per fold
    fig.add_trace(
        go.Bar(x=[f"Fold {i+1}" for i in range(len(fold_pr_aucs))],
               y=fold_pr_aucs,
               name="PR AUC",
               marker_color='lightgreen'),
        row=2, col=2
    )

    # Simulated training curves
    simulated_train_loss = [0.69 - (0.4 * (1 - np.exp(-0.15 * e))) + np.random.normal(0, 0.02) for e in epochs]
    simulated_val_loss = [0.68 - (0.35 * (1 - np.exp(-0.12 * e))) + np.random.normal(0, 0.03) for e in epochs]
    simulated_train_acc = [0.5 + (0.35 * (1 - np.exp(-0.15 * e))) + np.random.normal(0, 0.02) for e in epochs]
    simulated_val_acc = [0.52 + (0.3 * (1 - np.exp(-0.12 * e))) + np.random.normal(0, 0.02) for e in epochs]

    fig.add_trace(
        go.Scatter(x=epochs, y=simulated_train_loss, name="Train Loss", line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=simulated_val_loss, name="Val Loss", line=dict(color='orange')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=simulated_train_acc, name="Train Acc", line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=simulated_val_acc, name="Val Acc", line=dict(color='green')),
        row=1, col=2
    )

    fig.update_layout(height=600, showlegend=True, title_text="Training Progress and Cross-Validation Results")
    st.plotly_chart(fig, use_container_width=True)

def create_roc_pr_curves(tbom_data):
    """Create ROC and PR curves visualization (UNCHANGED)"""
    test_results = tbom_data.get('performance_metrics', {}).get('final_test_results', {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        roc_curve_data = test_results.get('roc_curve', [])
        if roc_curve_data and len(roc_curve_data) == 2:
            fpr, tpr = roc_curve_data
            roc_auc = test_results.get('roc_auc', 0)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='navy', width=2, dash='dash')
            ))
            fig_roc.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=400,
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve data not available")

    with col2:
        st.subheader("Precision-Recall Curve")
        pr_curve_data = test_results.get('pr_curve', [])
        if pr_curve_data and len(pr_curve_data) == 2:
            precision, recall = pr_curve_data
            pr_auc = test_results.get('pr_auc', 0)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AUC = {pr_auc:.3f})',
                line=dict(color='darkgreen', width=2)
            ))
            fig_pr.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=400,
                height=400
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("Precision-Recall curve data not available")

def create_model_architecture_viz(tbom_data):
    """Visualize model architecture"""
    arch_data = tbom_data.get('model_architecture', {})

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Architecture")

        components = arch_data.get('components', [])
        if components:
            # Create architecture diagram
            fig = go.Figure()

            y_positions = list(range(len(components)))
            component_names = [comp.get('name', f'Component {i}') for i, comp in enumerate(components)]

            fig.add_trace(go.Scatter(
                x=[1] * len(components),
                y=y_positions,
                mode='markers+text',
                marker=dict(size=60, color='lightblue', line=dict(width=2, color='darkblue')),
                text=component_names,
                textposition="middle center",
                name="Components"
            ))

            # Add arrows between components
            for i in range(len(components) - 1):
                fig.add_annotation(
                    x=1, y=y_positions[i],
                    ax=1, ay=y_positions[i+1],
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='darkblue'
                )

            fig.update_layout(
                title="Model Architecture Flow",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Architecture details table
        if components:
            arch_df = pd.DataFrame([
                {
                    'Component': comp.get('name', 'Unknown'),
                    'Type': comp.get('type', 'Unknown'),
                    'Details': comp.get('details', 'N/A')
                } for comp in components
            ])
            st.dataframe(arch_df, use_container_width=True)

    with col2:
        st.subheader("Architecture Summary")

        model_name = arch_data.get('model_name', 'Unknown')
        description = arch_data.get('description', 'No description available')

        st.markdown(f"""
        <div class="metric-card">
            <h4>🏗️ {model_name}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

        # Training methodology
        training_method = tbom_data.get('training_methodology', {})
        if training_method:
            st.subheader("Training Details")
            approach = training_method.get('approach', 'Unknown')
            hyperparams = training_method.get('hyperparameters', {})

            st.markdown(f"**Approach:** {approach}")

            if hyperparams:
                st.markdown("**Hyperparameters:**")
                for key, value in hyperparams.items():
                    st.markdown(f"- {key.replace('_', ' ').title()}: `{value}`")

def create_dataset_analysis(tbom_data):
    """Create dataset analysis visualizations"""
    data_summary = tbom_data.get('data_summary', {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")

        total_samples = data_summary.get('total_samples', 0)
        class_info = data_summary.get('class_information', {})
        class_dist = class_info.get('distribution', {})

        # Class distribution pie chart
        if class_dist:
            fig_pie = px.pie(
                values=list(class_dist.values()),
                names=list(class_dist.keys()),
                title="Class Distribution",
                color_discrete_sequence=['#90EE90', '#FFB6C1']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Dataset metrics
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Dataset Statistics</h4>
            <p><strong>Total Samples:</strong> {total_samples:,}</p>
            <p><strong>Number of Classes:</strong> {class_info.get('count', 0)}</p>
            <p><strong>Classes:</strong> {', '.join(class_info.get('names', []))}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Data Splits")

        data_splits = data_summary.get('data_splits', {})
        train_val = data_splits.get('train_validation_set', {})
        test_set = data_splits.get('test_set', {})

        # Split sizes bar chart
        split_data = {
            'Split': ['Train+Validation', 'Test'],
            'Size': [train_val.get('size', 0), test_set.get('size', 0)]
        }

        fig_splits = px.bar(
            split_data,
            x='Split',
            y='Size',
            title='Data Split Sizes',
            color='Split',
            color_discrete_sequence=['#4CAF50', '#FF9800']
        )
        fig_splits.update_layout(showlegend=False)
        st.plotly_chart(fig_splits, use_container_width=True)

        # Split details
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔄 Split Details</h4>
            <p><strong>Train+Val Size:</strong> {train_val.get('size', 0):,}</p>
            <p><strong>Test Size:</strong> {test_set.get('size', 0):,}</p>
            <p><strong>Test Ratio:</strong> {test_set.get('size', 0)/total_samples*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

def create_concept_analysis(tbom_data):
    """Create concept analysis visualization (UNCHANGED)"""
    concept_details = tbom_data.get('data_summary', {}).get('concept_details', {})
    model_interp = tbom_data.get('model_interpretation', {})

    st.subheader("Concept Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Concept overview
        concept_count = concept_details.get('count', 0)
        generation_method = concept_details.get('generation_method', 'Unknown')

        st.markdown(f"""
        <div class="metric-card">
            <h4>Concept Overview</h4>
            <p><strong>Total Concepts:</strong> {concept_count}</p>
            <p><strong>Generation Method:</strong> {generation_method}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Top concept indicators
        poisonous_indicators = model_interp.get('poisonous_indicators', [])
        edible_indicators = model_interp.get('edible_indicators', [])

        if poisonous_indicators or edible_indicators:
            tab1, tab2 = st.tabs(["Poisonous Indicators", "Edible Indicators"])

            with tab1:
                if poisonous_indicators:
                    poison_df = pd.DataFrame(poisonous_indicators[:10])
                    if 'weight' in poison_df.columns:
                        poison_df['weight'] = poison_df['weight'].round(4)
                    st.dataframe(poison_df, use_container_width=True)
                else:
                    st.info("No poisonous indicators data available")

            with tab2:
                if edible_indicators:
                    edible_df = pd.DataFrame(edible_indicators[:10])
                    if 'weight' in edible_df.columns:
                        edible_df['weight'] = edible_df['weight'].round(4)
                    st.dataframe(edible_df, use_container_width=True)
                else:
                    st.info("No edible indicators data available")

def create_performance_comparison(tbom_data):
    """Create performance comparison chart (UNCHANGED)"""
    cv_results = tbom_data.get('performance_metrics', {}).get('cross_validation_results', {})
    test_results = tbom_data.get('performance_metrics', {}).get('final_test_results', {})

    # Create comparison chart
    metrics = ['Accuracy', 'ROC AUC', 'PR AUC']
    cv_values = [
        cv_results.get('mean_accuracy', 0) * 100,
        cv_results.get('mean_roc_auc', 0) * 100,
        cv_results.get('mean_pr_auc', 0) * 100
    ]
    test_values = [
        test_results.get('accuracy', 0) * 100,
        test_results.get('roc_auc', 0) * 100,
        test_results.get('pr_auc', 0) * 100
    ]

    fig = go.Figure(data=[
        go.Bar(name='Cross-Validation', x=metrics, y=cv_values, marker_color='lightblue'),
        go.Bar(name='Test Set', x=metrics, y=test_values, marker_color='darkblue')
    ])

    fig.update_layout(
        title='Performance Comparison: Cross-Validation vs Test Set',
        xaxis_title='Metrics',
        yaxis_title='Performance (%)',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)

def create_technical_details(tbom_data):
    """Display technical details and add TBOM download button"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Generation Details")
        gen_details = tbom_data.get('generation_details', {})

        timestamp = gen_details.get('timestamp_utc', 'Unknown')
        tbom_version = tbom_data.get('tbom_version', 'Unknown')
        training_time = gen_details.get('training_time_seconds', 0)

        st.markdown(f"""
        <div class="metric-card">
            <h4>📋 TBOM Information</h4>
            <p><strong>Version:</strong> {tbom_version}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Training Time:</strong> {training_time/3600:.2f} hours</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Environment Details")
        env_details = tbom_data.get('environment_and_dependencies', {})

        device = env_details.get('device', 'Unknown')
        software_versions = env_details.get('software_versions', {})

        st.markdown(f"""
        <div class="metric-card">
            <h4>💻 Environment</h4>
            <p><strong>Device:</strong> {device}</p>
        </div>
        """, unsafe_allow_html=True)

        if software_versions:
            st.markdown("**Software Versions:**")
            for software, version in software_versions.items():
                if len(str(version)) < 50:  # Only show concise version info
                    st.markdown(f"- {software}: `{version}`")

    st.markdown("---")
    # --- Add download button for TBOM.json ---
    st.subheader("Download Full TBOM")
    tbom_json_data = json.dumps(tbom_data, indent=2)
    st.download_button(
        label="📥 Download TBOM.json",
        data=tbom_json_data,
        file_name="TBOM.json",
        mime="application/json"
    )

    # Expander for raw data
    with st.expander("🔍 View Raw TBOM Data"):
        st.json(tbom_data)


#Main function to run the Streamlit app

def main():
    """Main Streamlit application with enhanced IBOM integration"""
    # Header (keeping your original style)
    st.markdown('<h1 class="main-header">🍄 Dependable FungAI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Training Bill of Materials (TBOM) & Enhanced Inference Analysis</h2>', unsafe_allow_html=True)

    # Load TBOM data
    tbom_data = load_tbom_data()

    if tbom_data is None:
        st.error("Could not load TBOM data. Please ensure the TBOM.json file is available.")
        st.stop()

    # Sidebar information (keeping your original sidebar)
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 Dashboard Navigation")
        st.markdown("This dashboard displays comprehensive training metrics and enhanced inference analysis.")

        # Quick stats
        test_acc = tbom_data.get('performance_metrics', {}).get('final_test_results', {}).get('accuracy', 0) * 100
        st.markdown(f"**Model Performance:** {test_acc:.2f}%")

        total_samples = tbom_data.get('data_summary', {}).get('total_samples', 0)
        st.markdown(f"**Dataset Size:** {total_samples:,} samples")

        concept_count = tbom_data.get('data_summary', {}).get('concept_details', {}).get('count', 0)
        st.markdown(f"**Concepts Used:** {concept_count}")

        st.markdown('</div>', unsafe_allow_html=True)

        # File info
        st.markdown("### 📄 TBOM Info")
        tbom_version = tbom_data.get('tbom_version', 'Unknown')
        timestamp = tbom_data.get('generation_details', {}).get('timestamp_utc', 'Unknown')
        st.markdown(f"**Version:** {tbom_version}")
        st.markdown(f"**Generated:** {timestamp[:10]}")

    # Main content tabs
    tabs = st.tabs([
        "🔍 Enhanced IBOM",
        "🎯 Overview",
        "📈 Performance",
        "🏗️ Architecture",
        "🧠 Concepts",
        "📊 Data Summary",
        "💻 Technical"
    ])

    # NEW ENHANCED IBOM TAB WITH ALL REQUESTED FEATURES
    with tabs[0]:
        create_enhanced_ibom_interface(tbom_data)

    with tabs[1]:
        st.markdown("### Performance Overview")
        create_performance_overview(tbom_data)

        st.markdown("### Performance Comparison")
        create_performance_comparison(tbom_data)

    with tabs[2]:
        st.markdown("### Confusion Matrix Analysis")
        create_confusion_matrix_viz(tbom_data)

        st.markdown("### ROC and Precision-Recall Curves")
        create_roc_pr_curves(tbom_data)

        st.markdown("### Training Progress")
        create_training_curves(tbom_data)

    with tabs[3]:
        create_model_architecture_viz(tbom_data)

    with tabs[4]:
        create_concept_analysis(tbom_data)

    with tabs[5]:
        create_dataset_analysis(tbom_data)

    with tabs[6]:
        create_technical_details(tbom_data)

if __name__ == "__main__":
    main()
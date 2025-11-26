"""
Visualize the Hierarchical Attention Architecture.

This script generates a visual diagram of the 2-level hierarchical attention
architecture used for intermittent demand forecasting.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create a comprehensive architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4F8',
        'embedding': '#B8E6F0',
        'pwl': '#FFE5CC',
        'dense': '#FFD9B3',
        'shift_scale': '#D4E6F1',
        'attention_feature': '#FAD7A0',
        'attention_component': '#F8B88B',
        'zero_prob': '#D5DBDB',
        'output': '#A9DFBF'
    }
    
    # Title
    ax.text(8, 11.5, 'Hierarchical Attention Architecture', 
            fontsize=20, weight='bold', ha='center')
    ax.text(8, 11.0, '2-Level Attention: Feature-Level + Component-Level', 
            fontsize=14, ha='center', style='italic')
    
    # ===== INPUT LAYER =====
    y_input = 10.0
    
    # Main input
    main_input = FancyBboxPatch((0.5, y_input), 2, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['input'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(main_input)
    ax.text(1.5, y_input + 0.25, 'Main Input\n(10 features)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # SKU input
    sku_input = FancyBboxPatch((3, y_input), 1.5, 0.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(sku_input)
    ax.text(3.75, y_input + 0.25, 'SKU ID\n(6099 SKUs)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # SKU Embedding
    y_embed = 9.0
    embedding = FancyBboxPatch((3, y_embed), 1.5, 0.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['embedding'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(embedding)
    ax.text(3.75, y_embed + 0.25, 'SKU Embedding\n(8 dim)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrow from SKU input to embedding
    arrow = FancyArrowPatch((3.75, y_input), (3.75, y_embed + 0.5),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # ===== COMPONENT BRANCHES =====
    y_component = 8.0
    components = [
        {'name': 'Trend', 'x': 0.5, 'features': '[0] Time', 'use_pwl': True},
        {'name': 'Seasonal', 'x': 4.5, 'features': '[2,3,4,5] Seasonal', 'use_pwl': False},
        {'name': 'Holiday', 'x': 8.5, 'features': '[1] Holiday Dist', 'use_pwl': True},
        {'name': 'Regressor', 'x': 12.5, 'features': '[6,7,8,9] Regressors', 'use_pwl': False},
    ]
    
    component_outputs = []
    
    for comp in components:
        x = comp['x']
        
        # Feature extraction
        feature_box = FancyBboxPatch((x, y_component), 1.8, 0.4, 
                                     boxstyle="round,pad=0.03", 
                                     facecolor=colors['input'], 
                                     edgecolor='gray', linewidth=1)
        ax.add_patch(feature_box)
        ax.text(x + 0.9, y_component + 0.2, comp['features'], 
                ha='center', va='center', fontsize=7)
        
        # Arrow from main input
        arrow = FancyArrowPatch((1.5, y_input), (x + 0.9, y_component + 0.4),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray', linestyle='--')
        ax.add_patch(arrow)
        
        y_curr = y_component - 0.8
        
        # PWL Calibration (for Trend and Holiday)
        if comp['use_pwl']:
            pwl_box = FancyBboxPatch((x, y_curr), 1.8, 0.5, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor=colors['pwl'], 
                                     edgecolor='orange', linewidth=2)
            ax.add_patch(pwl_box)
            ax.text(x + 0.9, y_curr + 0.25, f'{comp["name"]} PWL\n(32 features)', 
                    ha='center', va='center', fontsize=8, weight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((x + 0.9, y_component), (x + 0.9, y_curr + 0.5),
                                   arrowstyle='->', mutation_scale=15, 
                                   linewidth=1.5, color='black')
            ax.add_patch(arrow)
            y_curr -= 0.8
        
        # Dense layer
        dense_box = FancyBboxPatch((x, y_curr), 1.8, 0.5, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['dense'], 
                                   edgecolor='darkorange', linewidth=2)
        ax.add_patch(dense_box)
        ax.text(x + 0.9, y_curr + 0.25, f'Dense + Mish\n(32 units)', 
                ha='center', va='center', fontsize=8, weight='bold')
        
        # Arrow
        if comp['use_pwl']:
            arrow = FancyArrowPatch((x + 0.9, y_curr + 1.3), (x + 0.9, y_curr + 0.5),
                                   arrowstyle='->', mutation_scale=15, 
                                   linewidth=1.5, color='black')
        else:
            arrow = FancyArrowPatch((x + 0.9, y_component), (x + 0.9, y_curr + 0.5),
                                   arrowstyle='->', mutation_scale=15, 
                                   linewidth=1.5, color='black')
        ax.add_patch(arrow)
        
        y_curr -= 0.8
        
        # Shift-and-Scale (ID-specific)
        shift_scale_box = FancyBboxPatch((x, y_curr), 1.8, 0.5, 
                                         boxstyle="round,pad=0.05", 
                                         facecolor=colors['shift_scale'], 
                                         edgecolor='blue', linewidth=2)
        ax.add_patch(shift_scale_box)
        ax.text(x + 0.9, y_curr + 0.25, f'Shift & Scale\n(ID-specific)', 
                ha='center', va='center', fontsize=8, weight='bold')
        
        # Arrow from dense
        arrow = FancyArrowPatch((x + 0.9, y_curr + 1.3), (x + 0.9, y_curr + 0.5),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)
        
        # Arrow from embedding to shift-scale
        arrow = FancyArrowPatch((3.75, y_embed), (x + 0.9, y_curr + 0.5),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='blue', linestyle=':')
        ax.add_patch(arrow)
        
        component_outputs.append((x + 0.9, y_curr))
    
    # ===== LEVEL 1: FEATURE-LEVEL ATTENTION =====
    y_feature_attn = 3.5
    
    ax.text(8, y_feature_attn + 1.2, '━━━ Level 1: Feature-Level Attention ━━━', 
            fontsize=12, weight='bold', ha='center', 
            bbox=dict(boxstyle='round', facecolor=colors['attention_feature'], alpha=0.5))
    
    for i, comp in enumerate(components):
        x = comp['x']
        
        # Feature attention box
        attn_box = FancyBboxPatch((x, y_feature_attn), 1.8, 0.5, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['attention_feature'], 
                                  edgecolor='darkgoldenrod', linewidth=2.5)
        ax.add_patch(attn_box)
        ax.text(x + 0.9, y_feature_attn + 0.35, f'{comp["name"]}', 
                ha='center', va='center', fontsize=9, weight='bold')
        ax.text(x + 0.9, y_feature_attn + 0.12, 'Feature Attention', 
                ha='center', va='center', fontsize=7)
        
        # Arrow from component output
        arrow = FancyArrowPatch((component_outputs[i][0], component_outputs[i][1]), 
                               (x + 0.9, y_feature_attn + 0.5),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2, color='darkgoldenrod')
        ax.add_patch(arrow)
        
        # Annotation
        if i == 0:
            ax.text(x + 0.9, y_feature_attn - 0.3, 'Which\nchangepoints?', 
                    ha='center', va='top', fontsize=7, style='italic', color='darkred')
        elif i == 2:
            ax.text(x + 0.9, y_feature_attn - 0.3, 'Which\ndistances?', 
                    ha='center', va='top', fontsize=7, style='italic', color='darkred')
    
    # ===== LEVEL 2: COMPONENT-LEVEL ATTENTION =====
    y_component_attn = 1.8
    x_component_attn = 7
    
    ax.text(8, y_component_attn + 1.0, '━━━ Level 2: Component-Level Attention ━━━', 
            fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=colors['attention_component'], alpha=0.5))
    
    # Component attention box
    comp_attn_box = FancyBboxPatch((x_component_attn, y_component_attn), 2.5, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['attention_component'], 
                                   edgecolor='darkred', linewidth=3)
    ax.add_patch(comp_attn_box)
    ax.text(x_component_attn + 1.25, y_component_attn + 0.35, 'Component Attention', 
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_component_attn + 1.25, y_component_attn + 0.1, '(4 weights: Trend, Seasonal, Holiday, Regressor)', 
            ha='center', va='center', fontsize=7)
    
    # Arrows from feature attention to component attention
    for i, comp in enumerate(components):
        x = comp['x']
        arrow = FancyArrowPatch((x + 0.9, y_feature_attn), 
                               (x_component_attn + 1.25, y_component_attn + 0.6),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2, color='darkred')
        ax.add_patch(arrow)
    
    # Arrow from embedding to component attention
    arrow = FancyArrowPatch((3.75, y_embed), (x_component_attn + 1.25, y_component_attn + 0.6),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=2, color='blue', linestyle=':')
    ax.add_patch(arrow)
    
    # Annotation
    ax.text(x_component_attn + 1.25, y_component_attn - 0.3, 
            'Which components matter\nfor this SKU?', 
            ha='center', va='top', fontsize=7, style='italic', color='darkred')
    
    # ===== ZERO PROBABILITY NETWORK =====
    y_zero = 0.5
    x_zero = 6
    
    zero_box = FancyBboxPatch((x_zero, y_zero), 4.5, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['zero_prob'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(zero_box)
    ax.text(x_zero + 2.25, y_zero + 0.4, 'Zero Probability Network', 
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_zero + 2.25, y_zero + 0.15, 'Dense(64) → Dense(64) → P(Zero)', 
            ha='center', va='center', fontsize=8)
    
    # Arrow from component attention to zero prob
    arrow = FancyArrowPatch((x_component_attn + 1.25, y_component_attn), 
                           (x_zero + 2.25, y_zero + 0.6),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ===== OUTPUTS =====
    y_output = -0.5
    
    # Base forecast
    base_box = FancyBboxPatch((2, y_output), 2, 0.4, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['output'], 
                              edgecolor='green', linewidth=2)
    ax.add_patch(base_box)
    ax.text(3, y_output + 0.2, 'Base Forecast', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Final forecast
    final_box = FancyBboxPatch((5, y_output), 2.5, 0.4, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['output'], 
                               edgecolor='green', linewidth=2)
    ax.add_patch(final_box)
    ax.text(6.25, y_output + 0.2, 'Final Forecast', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Zero probability output
    zero_out_box = FancyBboxPatch((8, y_output), 2, 0.4, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['output'], 
                                  edgecolor='green', linewidth=2)
    ax.add_patch(zero_out_box)
    ax.text(9, y_output + 0.2, 'P(Zero)', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows to outputs
    arrow = FancyArrowPatch((x_component_attn + 1.25, y_component_attn), 
                           (3, y_output + 0.4),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='green')
    ax.add_patch(arrow)
    
    arrow = FancyArrowPatch((x_zero + 2.25, y_zero), 
                           (6.25, y_output + 0.4),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='green')
    ax.add_patch(arrow)
    
    arrow = FancyArrowPatch((x_zero + 2.25, y_zero), 
                           (9, y_output + 0.4),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color='green')
    ax.add_patch(arrow)
    
    # ===== LEGEND =====
    legend_x = 11.5
    legend_y = 8.5
    
    ax.text(legend_x + 1, legend_y + 0.5, 'Legend', 
            fontsize=11, weight='bold', ha='center')
    
    legend_items = [
        ('Input/Features', colors['input']),
        ('PWL Calibration', colors['pwl']),
        ('Dense Layer', colors['dense']),
        ('Shift & Scale', colors['shift_scale']),
        ('Feature Attention', colors['attention_feature']),
        ('Component Attention', colors['attention_component']),
        ('Zero Prob Network', colors['zero_prob']),
        ('Output', colors['output']),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y - 0.3 * i
        rect = patches.Rectangle((legend_x, y - 0.1), 0.3, 0.2, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.4, y, label, fontsize=8, va='center')
    
    # ===== KEY INSIGHTS =====
    insights_x = 11.5
    insights_y = 4.5
    
    ax.text(insights_x + 1, insights_y + 0.5, 'Key Insights', 
            fontsize=11, weight='bold', ha='center')
    
    insights = [
        '• Sparse Attention (Entmax)',
        '• 5 Attention Layers',
        '• 23,225 Parameters',
        '• SKU-Specific Weights',
        '• Interpretable',
    ]
    
    for i, insight in enumerate(insights):
        y = insights_y - 0.3 * i
        ax.text(insights_x + 0.1, y, insight, fontsize=8, va='center')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Generating hierarchical attention architecture diagram...")
    
    fig = create_architecture_diagram()
    
    # Save
    output_file = 'hierarchical_attention_architecture.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Architecture diagram saved: {output_file}")
    
    plt.close()
    print("\nDone!")

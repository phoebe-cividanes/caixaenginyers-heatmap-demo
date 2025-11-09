"""
Streamlit App for Caixa Enginyers Branch Location Optimization
Using the sophisticated Economic and Social Scoring System
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path

# ===========================================
# DATA LOADING
# ===========================================

@st.cache_data
def load_scored_data():
    """Load the pre-scored municipality data."""
    # Try to load the cleaned scored data
    data_path = Path(__file__).parent.parent / "data" / "municipalities_scored_clean.csv"
    
    if not data_path.exists():
        st.error(f"❌ Scored data not found at {data_path}")
        st.info("Run: `python scripts/apply_scoring_pipeline.py` to generate scores")
        st.stop()
    
    df = pd.read_csv(data_path)
    return df


@st.cache_data
def load_geospatial_data():
    """Load geospatial data for municipalities (lat/lon)."""
    # TODO: Load your GeoJSON or CSV with municipality coordinates
    # For now, we'll create synthetic coordinates based on province
    
    # This is a placeholder - replace with actual coordinates
    province_centers = {
        'Madrid': (40.4168, -3.7038),
        'Barcelona': (41.3851, 2.1734),
        'Valencia/València': (39.4699, -0.3763),
        'Sevilla': (37.3891, -5.9845),
        'Coruña, A': (43.3713, -8.3960),
        'Bizkaia': (43.2630, -2.9350),
        'Alicante/Alacant': (38.3452, -0.4810),
        'Murcia': (37.9922, -1.1307),
        'Palmas, Las': (28.1236, -15.4366),
        'Zaragoza': (41.6488, -0.8891),
    }
    
    return province_centers


def merge_geospatial(df, province_centers):
    """Add lat/lon to municipalities based on province centers (simplified)."""
    # This is a simplified approach - in production, you'd have exact coordinates
    df = df.copy()
    
    # Add slight random offset to province centers for visualization
    np.random.seed(42)
    df['lat'] = df['provincia'].map(lambda p: province_centers.get(p, (40.0, 0.0))[0]) + np.random.uniform(-0.5, 0.5, len(df))
    df['lon'] = df['provincia'].map(lambda p: province_centers.get(p, (40.0, 0.0))[1]) + np.random.uniform(-0.5, 0.5, len(df))
    
    return df


# ===========================================
# SCORING FUNCTIONS
# ===========================================

def get_score_column(alpha):
    """Get the appropriate score column name for given alpha."""
    alpha_int = int(alpha * 100)
    return f'total_score_alpha_{alpha_int}'


def filter_data(df, filters):
    """Apply filters to the dataframe."""
    filtered = df.copy()
    
    # Population filter
    if filters['min_population'] > 0:
        filtered = filtered[filtered['poblacion_total'] >= filters['min_population']]
    if filters['max_population'] < 1000000:
        filtered = filtered[filtered['poblacion_total'] <= filters['max_population']]
    
    # Province filter
    if filters['provinces']:
        filtered = filtered[filtered['provincia'].isin(filters['provinces'])]
    
    # Bank saturation filter
    if filters['max_bank_saturation'] < 1.0:
        filtered = filtered[filtered['normalized_bank_count'] <= filters['max_bank_saturation']]
    
    return filtered


# ===========================================
# VISUALIZATION FUNCTIONS
# ===========================================

def create_color_scale(scores):
    """Create color scale from blue (low) to red (high)."""
    # Normalize scores to 0-1 for color mapping
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return [[100, 100, 100, 200]] * len(scores)
    
    normalized = (scores - min_score) / (max_score - min_score)
    
    colors = []
    for norm_score in normalized:
        # Blue (low) -> Yellow (medium) -> Red (high)
        if norm_score < 0.5:
            # Blue to Yellow
            r = int(255 * (norm_score * 2))
            g = int(255 * (norm_score * 2))
            b = int(255 * (1 - norm_score * 2))
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (norm_score - 0.5) * 2))
            b = 0
        colors.append([r, g, b, 200])
    
    return colors


# ===========================================
# STREAMLIT UI
# ===========================================

st.set_page_config(
    page_title="Caixa Enginyers - Branch Location Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏦 Caixa Enginyers - Branch Location Optimizer")
st.markdown("**Interactive heatmap for optimal bank branch placement in Spain**")
st.caption("Powered by sophisticated Economic and Social Scoring algorithms with sigmoid market opportunity modeling")

# Load data
with st.spinner("Loading scored municipality data..."):
    df = load_scored_data()
    province_centers = load_geospatial_data()
    df = merge_geospatial(df, province_centers)

st.success(f"✅ Loaded {len(df)} municipalities with scoring data")

# ===========================================
# SIDEBAR - MAIN CONTROLS
# ===========================================

with st.sidebar:
    st.header("🎚️ Strategy Control")
    
    # THE KEY CONTROL: Alpha slider
    st.markdown("### **Trade-off: Economic ↔ Social**")
    alpha = st.slider(
        "Strategic Focus",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0 = Pure Social Impact | 0.5 = Balanced | 1.0 = Pure Economic ROI"
    )
    
    # Visual indicator
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Social Weight", f"{(1-alpha)*100:.0f}%")
    with col2:
        st.metric("Economic Weight", f"{alpha*100:.0f}%")
    
    # Strategy label
    if alpha <= 0.2:
        strategy_label = "🤝 **Pure Social Impact**"
        strategy_color = "blue"
    elif alpha <= 0.4:
        strategy_label = "🌱 **Social-Leaning**"
        strategy_color = "lightblue"
    elif alpha <= 0.6:
        strategy_label = "⚖️ **Balanced** (Recommended)"
        strategy_color = "green"
    elif alpha <= 0.8:
        strategy_label = "💼 **Economic-Leaning**"
        strategy_color = "orange"
    else:
        strategy_label = "💰 **Pure Economic ROI**"
        strategy_color = "red"
    
    st.markdown(f"**Current Strategy:** {strategy_label}")
    
    st.divider()
    
    # ===========================================
    # FILTERS
    # ===========================================
    
    st.header("🔍 Filters")
    
    with st.expander("Population Range", expanded=False):
        min_pop = st.number_input(
            "Minimum Population",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000
        )
        max_pop = st.number_input(
            "Maximum Population",
            min_value=1000,
            max_value=1000000,
            value=1000000,
            step=10000
        )
    
    with st.expander("Geographic Filters", expanded=False):
        all_provinces = sorted(df['provincia'].dropna().unique())
        selected_provinces = st.multiselect(
            "Select Provinces (empty = all)",
            options=all_provinces,
            default=[]
        )
    
    with st.expander("Market Conditions", expanded=False):
        max_bank_sat = st.slider(
            "Max Bank Saturation",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Filter out municipalities with too many banks"
        )
    
    # Apply filters
    filters = {
        'min_population': min_pop,
        'max_population': max_pop,
        'provinces': selected_provinces,
        'max_bank_saturation': max_bank_sat
    }
    
    df_filtered = filter_data(df, filters)
    
    st.info(f"📊 {len(df_filtered)} municipalities match filters")
    
    st.divider()
    
    # ===========================================
    # VISUALIZATION SETTINGS
    # ===========================================
    
    st.header("🎨 Visualization")
    
    top_n = st.slider(
        "Show Top N Locations",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    radius = st.slider(
        "Point Size",
        min_value=50,
        max_value=500,
        value=150,
        step=25
    )

# ===========================================
# MAIN CONTENT
# ===========================================

# Get the appropriate score column
score_col = get_score_column(alpha)

# Check if normalized version exists, otherwise use raw
score_col_norm = score_col + '_normalized' if score_col + '_normalized' in df_filtered.columns else score_col

# Calculate scores for filtered data
df_filtered = df_filtered.copy()
df_filtered['current_score'] = df_filtered[score_col_norm]

# Sort by score
df_sorted = df_filtered.sort_values('current_score', ascending=False)

# ===========================================
# METRICS ROW
# ===========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_economic = df_filtered['economic_score_normalized'].mean()
    st.metric(
        "Avg Economic Score",
        f"{avg_economic:.1f}",
        help="Average economic score across filtered municipalities"
    )

with col2:
    avg_social = df_filtered['social_score_normalized'].mean()
    st.metric(
        "Avg Social Score",
        f"{avg_social:.1f}",
        help="Average social score across filtered municipalities"
    )

with col3:
    top_municipality = df_sorted.iloc[0]['municipio'] if len(df_sorted) > 0 else "N/A"
    st.metric(
        "Top Location",
        top_municipality,
        help="Highest scoring municipality with current settings"
    )

with col4:
    financial_deserts = len(df_filtered[df_filtered['normalized_bank_count'] < 0.2])
    st.metric(
        "Financial Deserts",
        financial_deserts,
        help="Municipalities with <20% bank saturation"
    )

st.divider()

# ===========================================
# TOP LOCATIONS TABLE
# ===========================================

st.subheader(f"🏆 Top {top_n} Recommended Locations")

# Select columns to display
display_cols = [
    'municipio', 'provincia', 'poblacion_total', 'num_bancos',
    'economic_score_normalized', 'social_score_normalized', 'current_score'
]

df_top = df_sorted[display_cols].head(top_n).reset_index(drop=True)
df_top.columns = [
    'Municipality', 'Province', 'Population', 'Banks',
    'Economic Score', 'Social Score', 'Total Score'
]

# Style the dataframe
styled_df = df_top.style.format({
    'Population': '{:,.0f}',
    'Banks': '{:.0f}',
    'Economic Score': '{:.1f}',
    'Social Score': '{:.1f}',
    'Total Score': '{:.1f}'
}).background_gradient(subset=['Total Score'], cmap='RdYlGn')

st.dataframe(styled_df, use_container_width=True, height=400)

# Download button
csv = df_top.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Top Locations (CSV)",
    data=csv,
    file_name=f"top_locations_alpha_{int(alpha*100)}.csv",
    mime="text/csv"
)

st.divider()

# ===========================================
# INTERACTIVE MAP
# ===========================================

st.subheader("🗺️ Interactive Heatmap")

# Prepare data for visualization
df_map = df_sorted.head(min(1000, len(df_sorted))).copy()  # Limit for performance
df_map['color'] = create_color_scale(df_map['current_score'])
df_map['size'] = radius

# Top locations for highlighting
df_top_map = df_sorted.head(top_n).copy()
df_top_map['size'] = radius * 1.5

# Calculate map center
center_lat = df_map['lat'].mean()
center_lon = df_map['lon'].mean()

# Create layers
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[lon, lat]',
    get_color='color',
    get_radius='size',
    pickable=True,
    opacity=0.6,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
)

top_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_top_map,
    get_position='[lon, lat]',
    get_color=[255, 215, 0, 255],  # Gold
    get_radius='size',
    pickable=True,
    opacity=1.0,
    stroked=True,
    filled=True,
    line_width_min_pixels=2,
)

# Tooltip
tooltip = {
    "html": """
    <b>{municipio}</b><br/>
    <b>Province:</b> {provincia}<br/>
    <b>Population:</b> {poblacion_total:,.0f}<br/>
    <b>Banks:</b> {num_bancos:.0f}<br/>
    <b>Economic Score:</b> {economic_score_normalized:.1f}<br/>
    <b>Social Score:</b> {social_score_normalized:.1f}<br/>
    <b>Total Score:</b> {current_score:.1f}
    """,
    "style": {
        "backgroundColor": "rgba(30,30,30,0.95)",
        "color": "white",
        "fontSize": "12px",
        "padding": "10px"
    }
}

# Create deck
deck = pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5.5,
        pitch=0
    ),
    map_provider="carto",
    map_style="light",
    layers=[scatter_layer, top_layer],
    tooltip=tooltip
)

st.pydeck_chart(deck, use_container_width=True)

# Legend
st.markdown("""
**Map Legend:**
- 🔵 Blue → Low Score | 🟡 Yellow → Medium Score | 🔴 Red → High Score
- ⭐ Gold circles = Top {} locations
""".format(top_n))

st.divider()

# ===========================================
# INSIGHTS & INTERPRETATION
# ===========================================

with st.expander("📊 How to Interpret the Results", expanded=False):
    st.markdown(f"""
    ### Current Strategy: {strategy_label}
    
    **Economic Score (α={alpha:.2f})** considers:
    - Potential revenue (income × density × economic activity)
    - Scalable costs (office size, staffing per customer)
    - Infrastructure penalty in underbanked areas
    - **Sigmoid market opportunity** (peaks at 40% bank saturation)
    
    **Social Score (1-α={1-alpha:.2f})** considers:
    - Community sustainability (elderly × youth balance)
    - Financial need (inverse of bank saturation)
    - Future viability (requires youth population)
    
    ### Key Innovations:
    1. **Sigmoid Function**: Recognizes that moderate competition (30-50%) is optimal
    2. **Scalable Costs**: Office size and staffing scale with expected demand
    3. **Infrastructure Penalty**: Higher costs in financial deserts
    4. **Sustainable Social**: Requires both current need (elderly) AND future viability (youth)
    
    ### Recommended Actions:
    - **α=0.0-0.3**: Focus on underserved communities, financial inclusion
    - **α=0.4-0.6**: Balanced approach (recommended for Caixa Enginyers)
    - **α=0.7-1.0**: Maximize profitability, urban centers
    """)

with st.expander("🔧 Technical Details", expanded=False):
    st.markdown(f"""
    ### Scoring Methodology
    
    **Total Score Formula:**
    ```
    Total Score = α × Economic Score + (1-α) × Social Score
    ```
    
    **Economic Score Components:**
    - Revenue = Income × Density × Economic_Activity_Factor
    - Costs = (Fixed + Variable) × Operational × Infrastructure
    - Market_Opportunity = Sigmoid(bank_saturation, peak=0.4)
    - ROI = (Revenue / Costs) × Market_Opportunity
    
    **Social Score Components:**
    - Community_Sustainability = %_over_65 × %_under_30
    - Financial_Need = 1 - normalized_bank_count
    - Social_Score = Community × Financial_Need
    
    ### Data Quality:
    - ✅ {len(df)} municipalities processed
    - ✅ Outliers capped at 99th percentile
    - ✅ Percentages normalized to [0-1]
    - ✅ Duplicates removed
    - ✅ Missing values imputed
    """)

# Footer
st.divider()
st.caption("🏦 Caixa Enginyers Branch Location Optimization Tool | Built with Streamlit | Data: INE Spain")

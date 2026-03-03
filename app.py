import streamlit as st
import pandas as pd
from data_engine import DataEngine
from predictor import ConflictPredictor
import plotly.express as px
import pydeck as pdk
import datetime
import numpy as np
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Geopolitical Prediction Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {
        background: #0f172a; /* Slightly lighter slate */
        color: #f8fafc;
    }
    .stApp {
        background-color: #0f172a;
    }
    /* ENFORCE FULL OPACITY - NO FADING */
    * {
        transition: none !important;
        animation: none !important;
        opacity: 1.0 !important;
    }
    .sidebar .sidebar-content {
        background-color: #0d1117;
    }
    h1, h2, h3 {
        color: #22d3ee;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .risk-card {
        padding: 24px;
        border-radius: 12px;
        background: #1e293b;
        border: 2px solid #334155;
        border-left: 6px solid #22d3ee;
        margin-bottom: 20px;
        color: #ffffff;
    }
    .stButton>button {
        background: #0891b2;
        color: white;
        border: none;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Logic ---

def main():
    st.title("🛡️ GLOBAL CONFLICT PREDICTION ENGINE")
    st.subheader(f"Predicting Geopolitical Shifts | AI Analysis of Global Sentiment | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Initialize Engines
    GEMINI_API_KEY = "AIzaSyCUADOdQcpDl4omCXi8xHR_rDX_IV5h1_0"
    data_engine = DataEngine()
    predictor = ConflictPredictor(api_key=GEMINI_API_KEY)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ CONTROL CENTER")
        st.info("Analyzing GDELT 2.0 (Real-time news updates) & OpenSky (Global Flight Data).")
        if st.button("RUN GLOBAL SCAN", key="scan_btn"):
            st.session_state.data_fetched = True
        
        st.markdown("---")
        st.write("### Prediction Settings")
        min_risk = st.slider("Min Risk Score", 0, 20, 5)

        # Region Selector for Intelligence Report
        st.markdown("---")
        st.markdown("### 🌍 REGIONAL INTELLIGENCE")
        region_list = [
            "-- Select a Region --",
            "Ukraine", "Russia", "Israel", "Iran", "China", "Taiwan", 
            "India", "Pakistan", "North Korea", "South Korea", "Japan",
            "United States of America", "Brazil", "Turkey", "Syria", 
            "Iraq", "Afghanistan", "Saudi Arabia", "Yemen", "Ethiopia",
            "Nigeria", "South Africa", "France", "United Kingdom",
            "Germany", "Poland", "Philippines", "Myanmar", "Mexico",
            "Colombia", "Venezuela", "Egypt", "Libya", "Sudan",
            "Somalia", "Democratic Republic of the Congo", "Mali"
        ]
        selected_region = st.selectbox("Select a country/region:", region_list, key="region_select")
        
    # Data Fetching
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False

    if st.session_state.data_fetched:
        status_placeholder = st.empty()
        ticker_placeholder = st.empty()
        import time
        
        scanned_actors = [
            "USA vs RUS", "ISR vs IRN", "CHN vs PHL", "UKR vs RUS", "IND vs PAK", 
            "TWN vs CHN", "SAU vs HOU", "PRK vs KOR", "ARM vs AZE", "ETH vs EGY"
        ]
        
        # Step 1: Initialize
        status_placeholder.info("📡 INITIALIZING SECURE CONNECTION TO INTELLIGENCE FEEDS...")
        time.sleep(2)
        
        # Step 2: Actual Blocking Fetch (with visual spinner)
        with st.spinner("🛰️ DOWNLOADING REAL-TIME SATELLITE & NEWS DATA..."):
            # These can take up to 20s total if both timeout
            gdelt_df = data_engine.fetch_latest_gdelt_events()
            flights_df = data_engine.fetch_flight_data()
        
        # Step 3: Fast "Synthesis" Ticker for Wow Factor
        for i in range(5):
            current_actor = scanned_actors[i % len(scanned_actors)]
            status_placeholder.success(f"✅ DATA ACQUIRED. FINALIZING PREDICTIONS... {5-i}s")
            ticker_placeholder.code(f">>> SYNTHESIZING SIGNAL: {current_actor}\n>>> GEMINI AI ENRICHMENT ACTIVE\n>>> GENERATING GEOSPATIAL MAP...")
            time.sleep(1)

        status_placeholder.empty() 
        ticker_placeholder.empty()
        
        if gdelt_df is not None and not gdelt_df.empty:
            hotspots = predictor.analyze_risk(gdelt_df)
            predictions = predictor.generate_predictions(hotspots)
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### 🌍 GLOBAL 3D RISK MONITOR")
                
                # Prepare data for PyDeck
                hotspots['radius'] = hotspots['RiskScore'] * 20000
                
                # Prepare Data for Plane Icons (Ensure float types)
                icon_data = []
                if flights_df is not None and not flights_df.empty:
                    for _, row in flights_df.head(100).iterrows():
                        try:
                            lat = float(row['latitude'])
                            lon = float(row['longitude'])
                            if pd.notna(lat) and pd.notna(lon):
                                icon_data.append({
                                    "pos": [lon, lat],
                                    "heading": float(row['true_track']) if pd.notna(row['true_track']) else 0
                                })
                        except:
                            continue

                # --- HUMINT TICKER ---
                # Pre-calculate labels and summary
                hotspots['Label'] = hotspots['Actor1'] + " vs " + hotspots['Actor2']
                
                hotspots_summary = ", ".join(hotspots['Label'].head(5).tolist())
                with st.spinner("Decoding Humorous Signals..."):
                    headline = predictor.generate_humorous_headline(hotspots_summary)
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, rgba(34,211,238,0) 0%, rgba(34,211,238,0.1) 15%, rgba(34,211,238,0.1) 85%, rgba(34,211,238,0) 100%); 
                            padding: 20px; text-align: center; border-top: 1px solid rgba(34,211,238,0.3); border-bottom: 1px solid rgba(34,211,238,0.3); margin-bottom: 25px;">
                    <span style="color: #22d3ee; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; font-size: 0.8em;">📡 LIVE INTEL TICKER: &nbsp;</span>
                    <span style="font-family: 'Space Mono', monospace; font-size: 1.2em; color: #f8fafc; font-weight: 500;">"{headline}"</span>
                </div>
                """, unsafe_allow_html=True)

                # PyDeck Layers
                layers = [
                    # 1. Light Terrain Base (Using Google Maps Terrain Hybrid)
                    pdk.Layer(
                        "TileLayer",
                        data="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", # Terrain Hybrid (Lighter)
                        tileSize=256,
                        pickable=False,
                    ),
                    # 2. Interactive Land (Click Detection)
                    pdk.Layer(
                        "GeoJsonLayer",
                        data="https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson",
                        id="countries-layer",
                        get_fill_color=[255, 255, 255, 1], # Nearly invisible
                        get_line_color=[34, 211, 238, 50],
                        pickable=True,
                        auto_highlight=True,
                        highlight_color=[0, 242, 255, 100],
                    ),
                    # 3. Conflict Pillars
                    pdk.Layer(
                        "ColumnLayer",
                        hotspots,
                        get_position=["Long", "Lat"],
                        get_elevation="RiskScore * 50000",
                        radius=80000,
                        get_fill_color="[239, 68, 68, 200]",
                        pickable=False,
                    ),
                    # 4. Pillar Labels (ENHANCED)
                    pdk.Layer(
                        "TextLayer",
                        hotspots.head(15), 
                        get_position=["Long", "Lat"],
                        get_text="Label",
                        get_color=[255, 255, 255, 255],
                        get_size=20,
                        get_alignment_baseline="'bottom'",
                        get_pixel_offset=[0, -60],
                        font_family="'Space Mono', monospace",
                        font_weight="bold",
                        outline_width=2,
                        outline_color=[0, 0, 0, 150]
                    )
                ]

                # Tactical Globe View
                view_state = pdk.ViewState(
                    latitude=30,
                    longitude=20,
                    zoom=1.2,
                    pitch=30,
                )

                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    views=[pdk.View(type="_GlobeView", controller=True)],
                    tooltip={"text": "Region: {name}"},
                    map_style=None
                )

                # Capture clicks via Streamlit
                selection_event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object")

                # --- REGIONAL INTELLIGENCE SECTION ---
                # Method 1: Try to get region from map click
                clicked_region = None
                if selection_event and hasattr(selection_event, 'selection') and selection_event.selection:
                    sel = selection_event.selection
                    objects = sel.get('objects', {}) if isinstance(sel, dict) else getattr(sel, 'objects', {})
                    # Try multiple possible keys (PyDeck uses layer type or ID)
                    for key in ['countries-layer', 'GeoJsonLayer', 0]:
                        layer_objs = objects.get(key, []) if isinstance(objects, dict) else []
                        if layer_objs:
                            clicked_region = layer_objs[0].get('properties', {}).get('name', None)
                            if clicked_region:
                                break
                
                # Method 2: Use sidebar dropdown (guaranteed to work)
                display_region = clicked_region
                if not display_region and selected_region != "-- Select a Region --":
                    display_region = selected_region

                # Show Intelligence Report
                if display_region:
                    st.markdown(f"---")
                    st.markdown(f"## 👁️ REGIONAL INTELLIGENCE: {display_region}")
                    with st.spinner(f"Agentic Gemini AI analyzing {display_region}..."):
                        intelligence = predictor.get_regional_intelligence(display_region)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info("📜 **HISTORY OF CONFLICT**")
                            st.markdown(intelligence['history'])
                        with c2:
                            st.warning("🔮 **FUTURE POSSIBILITIES**")
                            st.markdown(intelligence['future'])

            with col2:
                st.write("### ⚠️ TACTICAL RISK PREDICTIONS")
                for idx, pred in enumerate(predictions):
                    with st.container():
                        risk_color = "#ff5252" if "CRITICAL" in pred['risk_level'].upper() else "#fbbf24"
                        st.markdown(f"""
                        <div class="risk-card" style="border-left-color: {risk_color}; margin-bottom: 20px;">
                            <h4 style='color: {risk_color}; margin-bottom: 5px;'>{pred['risk_level']} RISK</h4>
                            <p style='font-size: 1.3em; font-weight: 800; color: #22d3ee; margin-bottom: 10px;'>{pred['actors']}</p>
                            <p><strong style='color: #94a3b8;'>🛡️ Tactical Prediction:</strong> {pred['reason']}</p>
                            <p><strong style='color: #94a3b8;'>🛰️ X-SENTIMENT (SOCIAL):</strong> <i style='color: #38bdf8;'>{pred.get('social_sentiment', 'SIGNAL NOISY')}</i></p>
                            <p><strong style='color: #94a3b8;'>⚔️ Weapons/Assets:</strong> <span style='color: #38bdf8;'>{pred['weapons']}</span></p>
                            <div style='text-align: right; font-size: 0.8em; margin-top: 10px;'>
                                <a href='{pred["source"]}' style='color: #64748b; text-decoration: none;'>[SOURCE INTEL]</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # --- Visual Analytics Section ---
            st.markdown("---")
            st.write("### 📊 RISK DISTRIBUTION & ANALYTICS")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                fig = px.histogram(hotspots, x="RiskScore", title="Conflict Risk Distribution", color_discrete_sequence=['#00e5ff'])
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                country_risks = hotspots.groupby('Actor1')['RiskScore'].sum().sort_values(ascending=False).head(10)
                fig2 = px.bar(country_risks, title="Aggressive Actor Indicators", color_discrete_sequence=['#ff5252'])
                st.plotly_chart(fig2, use_container_width=True)
            
            with c3:
                if flights_df is not None:
                    origin_counts = flights_df['origin_country'].value_counts().head(10)
                    fig3 = px.pie(values=origin_counts.values, names=origin_counts.index, title="Active Air Traffic Origins", hole=0.4)
                    st.plotly_chart(fig3, use_container_width=True)

        else:
            st.error("Failed to fetch intelligence data. Check network connectivity.")
    else:
        # Landing Page
        st.markdown("""
        <div style="text-align: center; padding: 60px;">
            <h1 style="font-size: 3em; color: #00e5ff;">SYSTEM STANDBY</h1>
            <p style="font-size: 1.5em; color: #888;">Press 'RUN GLOBAL SCAN' to initialize the prediction engine.</p>
            <p style="color: #64748b;">Scanning news wires, satellite flight feeds, and historical patterns across 10,000+ actor pairs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show Regional Intelligence even without a scan
        if selected_region != "-- Select a Region --":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            predictor = ConflictPredictor(api_key=api_key)
            
            st.markdown(f"---")
            st.markdown(f"## 👁️ REGIONAL INTELLIGENCE: {selected_region}")
            with st.spinner(f"Analyzing {selected_region}..."):
                intelligence = predictor.get_regional_intelligence(selected_region)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("📜 **HISTORY OF CONFLICT**")
                    st.markdown(intelligence['history'])
                with c2:
                    st.warning("🔮 **FUTURE POSSIBILITIES**")
                    st.markdown(intelligence['future'])

    # (Music Player Removed)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from datetime import datetime
from one_min_promax_ai import run_signal_engine  # Make sure your AI model is saved as this file

# === Streamlit Config ===
st.set_page_config(
    page_title="🚀 Binary Signal Pro Max AI (1M)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === TITLE ===
st.title("🚀 Pro Max AI — Binary 1-Minute Signal Engine")
st.markdown("Smart AI signal predictions for the **next 1-minute candle** based on ensemble voting (XGBoost + CatBoost).")
st.caption("✅ Optimized for high-frequency binary trading.")

# === REFRESH BUTTON ===
if st.button("🔁 Refresh Pro Max AI"):
    with st.spinner("Running 1-minute Pro Max AI model..."):
        st.session_state['df_pro_max'] = run_signal_engine()
        st.session_state['last_refreshed'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# === DISPLAY RESULTS ===
df = st.session_state.get('df_pro_max', pd.DataFrame())

if not df.empty:
    st.success(f"✅ {len(df)} signals generated.")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ No signals yet. Click 'Refresh Pro Max AI' to generate.")

# === LAST REFRESHED ===
st.markdown(f"🕒 **Last Refreshed:** {st.session_state.get('last_refreshed', 'Not yet refreshed')}")

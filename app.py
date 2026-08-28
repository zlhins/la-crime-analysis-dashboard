"""
LA Crime Intelligence Dashboard
================================
Versi: Ultra-Optimized Memory (Anti-OOM)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import gc  # Garbage Collector untuk mengosongkan RAM

# =============================================================================
# 1. KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="LA Crime Intelligence",
    page_icon=":material/local_police:", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. TEMA WARNA & CUSTOM CSS
# =============================================================================
PRIMARY = "#0F172A"     
ACCENT = "#4F46E5"      
DANGER = "#E11D48"      
SUCCESS = "#059669"     
WARNING = "#D97706"     
MUTED = "#64748B"       
BORDER = "#E2E8F0"      
CARD_BG = "#FFFFFF"
PAGE_BG = "#F8FAFC"

SEQ_MIX = ["#4F46E5", "#0F172A", "#64748B", "#CBD5E1", "#E2E8F0"]
DAY_ORDER_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ID = {
    "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu", "Thursday": "Kamis",
    "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,400,0,0');
    
    .stApp {{ background-color: {PAGE_BG}; font-family: 'Inter', sans-serif; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}
    h1, h2, h3, h4 {{ color: {PRIMARY}; font-weight: 600; letter-spacing: -0.02em; }}

    .dash-header {{ border-bottom: 2px solid {PRIMARY}; padding-bottom: 12px; margin-bottom: 24px; display: flex; align-items: center; gap: 12px; }}
    .dash-header-icon {{ font-size: 36px; color: {PRIMARY}; }}
    .dash-header h1 {{ color: {PRIMARY}; margin: 0; font-size: 26px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; line-height: 1; }}
    .dash-header p {{ color: {MUTED}; margin: 4px 0 0 0; font-size: 14px; }}

    .kpi-card {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-top: 3px solid var(--kpi-color, {ACCENT}); padding: 16px 20px; height: 100%; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
    .kpi-label {{ font-size: 12px; color: {MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 6px; }}
    .kpi-icon {{ font-size: 16px; color: var(--kpi-color, {ACCENT}); }}
    .kpi-value {{ font-size: 24px; color: {PRIMARY}; font-weight: 700; margin-top: 8px; line-height: 1.1; }}
    .kpi-sub {{ font-size: 12px; color: {MUTED}; margin-top: 4px; }}

    .section-title {{ font-size: 15px; font-weight: 700; color: {PRIMARY}; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.02em; display: flex; align-items: center; gap: 6px; }}
    .section-caption {{ font-size: 13px; color: {MUTED}; margin-bottom: 16px; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ background-color: transparent; border: none; padding: 12px 4px; color: {MUTED}; font-weight: 500; font-size: 14px; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. FUNGSI LOAD DATA (DIET MEMORI EKSTREM)
# =============================================================================
DATA_PATH = "data/la_crime_clean.parquet" # Pastikan nama file sesuai dengan yang ada di folder data kamu

@st.cache_data(show_spinner="Memuat dataset & optimasi memori...")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    # Membaca file
    df = pd.read_parquet(path)

    # Membuang memori yang tidak perlu dengan cara downcasting (memperkecil tipe data)
    if not pd.api.types.is_datetime64_any_dtype(df["occurrence_date"]):
        df["occurrence_date"] = pd.to_datetime(df["occurrence_date"], errors="coerce")
    
    if "year" not in df.columns:
        df["year"] = df["occurrence_date"].dt.year.fillna(0).astype("int16")
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["occurrence_date"].dt.day_name()
    if "hour" not in df.columns:
        df["occurrence_time"] = df["occurrence_time"].astype(str)
        df["hour"] = df["occurrence_time"].str.extract(r'^(\d{1,2})')[0].astype(float).fillna(-1).astype("int8")

    # Mapping status satu kali di awal agar tidak memakan RAM saat filtering
    status_map = {
        "Invest Cont": "Investigasi Berjalan", "Adult Arrest": "Terselesaikan (Dewasa)",
        "Adult Other": "Resolusi Lain (Dewasa)", "Juv Arrest": "Terselesaikan (Anak)",
        "Juv Other": "Resolusi Lain (Anak)", "UNK": "Tidak Teridentifikasi",
    }
    df["status"] = df["status"].map(status_map).fillna(df["status"])
    df["is_arrest"] = df["status"].isin(["Terselesaikan (Dewasa)", "Terselesaikan (Anak)"])
    
    df["victim_age"] = pd.to_numeric(df["victim_age"], errors="coerce").fillna(0).astype("int8")
    df["age_group"] = pd.cut(
        df["victim_age"],
        bins=[0, 10, 18, 25, 35, 45, 55, 65, 120],
        labels=["0-10", "11-18", "19-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    
    # KUNCI UTAMA ANTI-OOM: Mengubah teks (string) menjadi Categorical
    cat_cols = ["area", "crime_category", "crime", "victim_gender", "victim_ethnicity", "premise", "weapon", "status", "day_of_week", "age_group"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

@st.cache_data
def to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"File data tidak ditemukan di `{DATA_PATH}`.")
    st.stop()

# =============================================================================
# 4. SIDEBAR — FILTER MENGGUNAKAN BOOLEAN MASKING
# =============================================================================
FILTER_KEYS = ["flt_year", "flt_area", "flt_category", "flt_days", "flt_hour", "flt_crimes"]

def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)

with st.sidebar:
    st.markdown("### :material/tune: PARAMETER")
    st.markdown("---")

    st.markdown("**:material/calendar_month: TAHUN**")
    available_years = sorted([y for y in df["year"].unique() if y > 0], reverse=True)
    selected_year = st.multiselect("Tahun", options=available_years, default=available_years, key="flt_year", label_visibility="collapsed")

    st.markdown("**:material/location_on: AREA**")
    available_areas = sorted(df["area"].dropna().unique().tolist())
    selected_area = st.multiselect("Area", options=available_areas, default=[], key="flt_area", label_visibility="collapsed", placeholder="Semua area")

    st.markdown("**:material/category: KATEGORI**")
    category_options = sorted(df["crime_category"].dropna().unique().tolist())
    selected_category = st.multiselect("Kategori", options=category_options, default=category_options, key="flt_category", label_visibility="collapsed")

    st.markdown("---")
    with st.expander("FILTER LANJUTAN"):
        selected_days = st.multiselect("HARI", options=DAY_ORDER_EN, default=[], format_func=lambda d: DAY_ID[d], key="flt_days", placeholder="Semua hari")
        hour_range = st.slider("RENTANG JAM", 0, 23, (0, 23), key="flt_hour")
        top_crime_options = sorted(df["crime"].value_counts().head(40).index.tolist())
        selected_crimes = st.multiselect("MODUS (40 TERATAS)", options=top_crime_options, default=[], key="flt_crimes", placeholder="Semua modus")

    st.markdown("---")
    st.button("Reset Filter", use_container_width=True, on_click=reset_filters)

# ----------------- PENGAMAN MEMORI -----------------
mask = pd.Series(True, index=df.index)

if selected_year:
    mask &= df["year"].isin(selected_year)
if selected_area:
    mask &= df["area"].isin(selected_area)
if selected_category:
    mask &= df["crime_category"].isin(selected_category)
if selected_days:
    mask &= df["day_of_week"].isin(selected_days)
mask &= df["hour"].between(hour_range[0], hour_range[1])
if selected_crimes:
    mask &= df["crime"].isin(selected_crimes)

total_crimes = mask.sum()

st.sidebar.caption(
    f"SAMPEL DATA: **{total_crimes:,}** / **{len(df):,}** "
    f"({(total_crimes / len(df) * 100 if len(df) else 0):.1f}%)"
)

# =============================================================================
# 5. HEADER
# =============================================================================
period_min = df["occurrence_date"].min()
period_max = df["occurrence_date"].max()
st.markdown(f"""
<div class="dash-header">
    <span class="material-symbols-outlined dash-header-icon">policy</span>
    <div>
        <h1>LA CRIME INTELLIGENCE</h1>
        <p>Analisis komprehensif metrik kejahatan, distribusi geospasial, dan demografi ({period_min:%b %Y} – {period_max:%b %Y})</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 6. HELPER: KARTU KPI
# =============================================================================
def kpi_card(col, icon: str, label: str, value: str, sub: str = "", color: str = PRIMARY):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(f"""
    <div class="kpi-card" style="--kpi-color:{color}">
        <div class="kpi-label"><span class="material-symbols-outlined kpi-icon">{icon}</span> {label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def section_intro(icon: str, title: str, caption: str = ""):
    st.markdown(f'<div class="section-title"><span class="material-symbols-outlined" style="font-size:18px;">{icon}</span> {title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

# =============================================================================
# 7. KONTEN UTAMA - KALKULASI KPI (MENGGUNAKAN df.loc)
# =============================================================================
if total_crimes == 0:
    st.warning("Data tidak tersedia untuk parameter filter yang dipilih.")
    st.stop()

serious_pct = (df.loc[mask, "crime_category"] == "Serious Crime").mean() * 100
arrest_pct = df.loc[mask, "is_arrest"].mean() * 100

crime_counts = df.loc[mask, "crime"].value_counts()
top_crime = crime_counts.index[0] if len(crime_counts) > 0 else "N/A"
top_crime_n = crime_counts.iloc[0] if len(crime_counts) > 0 else 0

area_counts = df.loc[mask, "area"].value_counts()
top_area = area_counts.index[0] if len(area_counts) > 0 else "N/A"
top_area_n = area_counts.iloc[0] if len(area_counts) > 0 else 0

avg_age = df.loc[mask, "victim_age"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_card(k1, "monitoring", "Total Insiden", f"{total_crimes:,}", color=PRIMARY)
kpi_card(k2, "warning", "Insiden Serius", f"{serious_pct:.1f}%", "Rasio terhadap total", color=DANGER)
kpi_card(k3, "gavel", "Penangkapan", f"{arrest_pct:.1f}%", "Kasus terselesaikan", color=SUCCESS)
kpi_card(k4, "fingerprint", "Modus Dominan", str(top_crime).title()[:22] + ("…" if len(str(top_crime)) > 22 else ""), f"{top_crime_n:,} laporan", color=WARNING)
kpi_card(k5, "map", "Area Rawan", str(top_area), f"{top_area_n:,} laporan", color=ACCENT)
kpi_card(k6, "person", "Usia Rata-rata", f"{avg_age:.0f} Thn", "Demografi korban", color=MUTED)

st.write("")

# =============================================================================
# 8. TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/explore: PETA & OVERVIEW",
    ":material/schedule: ANALISIS WAKTU",
    ":material/group: DEMOGRAFI",
    ":material/shield: ATRIBUT KASUS",
    ":material/table_view: DATA TABEL",
])

# ------------------------- TAB 1: OVERVIEW & MAP ----------------------------
with tab1:
    st.write("")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        section_intro("timeline", "Tren Tahunan")
        trend_data = df.loc[mask, "year"].value_counts().reset_index()
        trend_data.columns = ["year", "count"]
        trend_data = trend_data.sort_values("year")
        fig_trend = px.area(trend_data, x="year", y="count", markers=True, color_discrete_sequence=[PRIMARY])
        fig_trend.update_traces(line=dict(width=2), fillcolor="rgba(15, 23, 42, 0.05)")
        fig_trend.update_layout(xaxis_title="", yaxis_title="", template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(dtick=1), height=300)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        section_intro("bar_chart", "Distribusi Area (Top 10)")
        area_data = area_counts.head(10).reset_index()
        area_data.columns = ["area", "count"]
        fig_area = px.bar(area_data, x="count", y="area", orientation="h", color="count", color_continuous_scale="gray", text="count")
        fig_area.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
        fig_area.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="", template="plotly_white", coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig_area, use_container_width=True)

    st.write("")
    section_intro("public", "Kepadatan Insiden Geospasial", "Heatmap distribusi. Diambil maks 1.000 sampel acak agar performa browser optimal.")
    map_mask = mask & df["latitude"].notna()
    
    if map_mask.sum() > 0:
        if map_mask.sum() > 1000:
            sample_idx = df[map_mask].sample(1000, random_state=42).index
            map_data = df.loc[sample_idx, ["latitude", "longitude"]]
        else:
            map_data = df.loc[map_mask, ["latitude", "longitude"]]

        with st.spinner("Merender peta..."):
            m = folium.Map(location=[34.0522, -118.2437], zoom_start=10, tiles="OpenStreetMap")
            heat_data = map_data.values.tolist()
            HeatMap(heat_data, radius=13, blur=12, gradient={0.4: PRIMARY, 0.65: WARNING, 1: DANGER}).add_to(m)
            components.html(m._repr_html_(), height=450)
    else:
        st.info("Data koordinat tidak tersedia.")

# ------------------------- TAB 2: TEMPORAL ----------------------------------
with tab2:
    st.write("")
    col_time1, col_time2 = st.columns(2)

    with col_time1:
        section_intro("schedule", "Distribusi Jam")
        hour_data = df.loc[mask & (df["hour"] >= 0), "hour"].value_counts().sort_index().reset_index()
        hour_data.columns = ["hour", "count"]
        fig_hour = px.bar(hour_data, x="hour", y="count", color="count", color_continuous_scale="Blues")
        fig_hour.update_layout(xaxis_title="Jam (00–23)", yaxis_title="", template="plotly_white", coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
        fig_hour.update_xaxes(dtick=2)
        st.plotly_chart(fig_hour, use_container_width=True)

    with col_time2:
        section_intro("event", "Distribusi Hari")
        day_data = df.loc[mask, "day_of_week"].value_counts().reindex(DAY_ORDER_EN).reset_index()
        day_data.columns = ["day", "count"]
        day_data["day_id"] = day_data["day"].map(DAY_ID)
        fig_day = px.bar(day_data, x="day_id", y="count", color_discrete_sequence=[MUTED])
        fig_day.update_layout(xaxis_title="", yaxis_title="", template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig_day, use_container_width=True)

    st.write("")
    section_intro("grid_on", "Intensitas Waktu", "Matriks jam kejadian versus hari.")
    pivot_subset = df.loc[mask & (df["hour"] >= 0), ["day_of_week", "hour", "report_number"]]
    pivot = pivot_subset.pivot_table(index="day_of_week", columns="hour", values="report_number", aggfunc="count", fill_value=0).reindex(DAY_ORDER_EN)
    pivot.index = [DAY_ID[d] if d in DAY_ID else d for d in pivot.index]
    
    fig_heat = px.imshow(pivot, color_continuous_scale="gray", aspect="auto", labels=dict(x="Jam", y="Hari", color="Insiden"))
    fig_heat.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------- TAB 3: DEMOGRAFI ---------------------------------
with tab3:
    st.write("")
    col_demo1, col_demo2 = st.columns([1, 2])

    with col_demo1:
        section_intro("wc", "Gender")
        gender_mask = mask & df["victim_gender"].isin(["Male", "Female"])
        gender_counts = df.loc[gender_mask, "victim_gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        gender_counts["gender"] = gender_counts["gender"].map({"Male": "Pria", "Female": "Wanita"})

        if not gender_counts.empty and gender_counts['count'].sum() > 0:
            fig_gender = px.pie(gender_counts, names="gender", values="count", hole=0.6, color_discrete_sequence=[PRIMARY, BORDER])
            fig_gender.update_traces(textinfo="percent", hoverinfo="label+value")
            fig_gender.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("Data gender tidak tersedia.")

    with col_demo2:
        section_intro("bar_chart_4_bars", "Kelompok Usia")
        age_grp = df.loc[mask, "age_group"].value_counts().reindex(["0-10", "11-18", "19-25", "26-35", "36-45", "46-55", "56-65", "65+"]).reset_index()
        age_grp.columns = ["group", "count"]
        fig_age = px.bar(age_grp, x="group", y="count", color_discrete_sequence=[ACCENT])
        fig_age.update_layout(xaxis_title="Rentang Usia", yaxis_title="", template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig_age, use_container_width=True)

    st.write("")
    section_intro("groups", "Profil Etnisitas (Top 10)")
    eth_mask = mask & (df["victim_ethnicity"] != "Unknown")
    eth_data = df.loc[eth_mask, "victim_ethnicity"].value_counts().head(10).reset_index()
    eth_data.columns = ["ethnicity", "count"]
    
    fig_eth = px.bar(eth_data, x="count", y="ethnicity", orientation="h", color="count", color_continuous_scale="gray")
    fig_eth.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="", template="plotly_white", coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_eth, use_container_width=True)

# ------------------------- TAB 4: ATRIBUT KASUS -----------------------------
with tab4:
    st.write("")
    hide_unknown_weapon = st.checkbox("Sembunyikan entitas 'Unknown'", value=True)

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        section_intro("hardware", "Instrumen Senjata (Top 10)")
        if hide_unknown_weapon:
            valid_weapons = [w for w in df["weapon"].cat.categories if "UNKNOWN" not in str(w).upper()]
            weapon_mask = mask & df["weapon"].isin(valid_weapons)
        else:
            weapon_mask = mask
            
        weapon_data = df.loc[weapon_mask, "weapon"].value_counts().head(10).reset_index()
        weapon_data.columns = ["weapon", "count"]
        
        fig_weapon = px.bar(weapon_data, x="count", y="weapon", orientation="h", color_discrete_sequence=[PRIMARY])
        fig_weapon.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="", template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=320)
        st.plotly_chart(fig_weapon, use_container_width=True)

    with col_w2:
        section_intro("task_alt", "Status Resolusi")
        status_data = df.loc[mask, "status"].value_counts().reset_index()
        status_data.columns = ["status", "count"]
        fig_status = px.pie(status_data, names="status", values="count", hole=0.6, color_discrete_sequence=SEQ_MIX)
        fig_status.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=320, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        st.plotly_chart(fig_status, use_container_width=True)

    st.write("")
    section_intro("store", "Konteks Ruang (Premise)")
    premise_data = df.loc[mask, "premise"].value_counts().head(10).reset_index()
    premise_data.columns = ["premise", "count"]
    fig_premise = px.bar(premise_data, x="count", y="premise", orientation="h", color_discrete_sequence=[MUTED])
    fig_premise.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="", template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_premise, use_container_width=True)

# ------------------------- TAB 5: DATA TABEL --------------------------------
with tab5:
    st.write("")
    section_intro("dataset", "Eksplorasi Data Tabular", "Menampilkan maksimal 2.000 baris terbaru. Gunakan tombol unduh untuk mendapatkan data lengkap.")

    keyword = st.text_input("Pencarian Modus:", placeholder="Ketik kata kunci (misal: theft, assault)...")
    
    display_cols = [
        "report_number", "occurrence_date", "area", "crime_category", "crime",
        "victim_age", "victim_gender", "victim_ethnicity", "premise", "weapon", "status",
    ]
    
    if keyword:
        kw_mask = mask & df["crime"].astype(str).str.contains(keyword, case=False, na=False)
        table_df = df.loc[kw_mask, display_cols]
    else:
        table_df = df.loc[mask, display_cols]

    st.dataframe(
        table_df.head(2000), 
        use_container_width=True, hide_index=True, height=400,
    )

    # Batasi unduhan CSV max 50.000 baris demi menjaga RAM server
    csv_bytes = to_csv_bytes(table_df.head(50000))
    st.download_button(
        "Unduh CSV (Maks 50k baris)", data=csv_bytes,
        file_name="la_crime_export.csv", mime="text/csv",
        type="primary", icon=":material/download:"
    )

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 12px;'>"
    "Data bersumber dari Los Angeles Police Department (LAPD) Open Data. "
    "Fluktuasi tren harus diinterpretasikan dengan mempertimbangkan perubahan metodologi pelaporan."
    "</div>", 
    unsafe_allow_html=True
)

# Bersihkan sisa memori di akhir siklus render
gc.collect()
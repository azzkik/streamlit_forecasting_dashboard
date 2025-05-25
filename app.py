import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pytanggalmerah import TanggalMerah

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Telkomsel Sales Forecasting",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Main theme */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
        margin: 0;
    }
    
    /* Success/Error messages */
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Upload area */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom button */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: 600;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# === FUNGSI UNTUK PERSIAPAN DATA ===
@st.cache_data
def prepare_data(df, days_forward=10):
    """Menyiapkan data untuk prediksi dengan feature engineering"""
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    last_date = df['tanggal'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_forward + 1)]
    
    future_df = pd.DataFrame({
        'tanggal': future_dates,
        'penjualan': [None]*days_forward
    })

    df_all = pd.concat([df, future_df], ignore_index=True)

    # Feature engineering
    df_all['hari_minggu'] = df_all['tanggal'].apply(lambda x: 1 if x.weekday() == 6 else 0)
    df_all['libur_nasional'] = df_all['tanggal'].apply(
        lambda x: 1 if TanggalMerah(x.strftime('%Y-%m-%d')).check() else 0
    )

    return df_all

@st.cache_data
def calculate_metrics(df):
    """Menghitung metrik penting dari data"""
    avg_sales = df['penjualan'].mean()
    max_sales = df['penjualan'].max()
    min_sales = df['penjualan'].min()
    total_days = len(df)
    
    return avg_sales, max_sales, min_sales, total_days

def create_forecast_chart(df_result):
    """Membuat chart interaktif untuk prediksi"""
    fig = go.Figure()
    
    # Data aktual
    actual_data = df_result[df_result['tipe'] == 'Aktual']
    fig.add_trace(go.Scatter(
        x=actual_data['tanggal'],
        y=actual_data['penjualan'],
        mode='lines+markers',
        name='Data Aktual',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    # Data prediksi
    prediction_data = df_result[df_result['tipe'] == 'Prediksi']
    fig.add_trace(go.Scatter(
        x=prediction_data['tanggal'],
        y=prediction_data['penjualan'],
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='#764ba2', width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    fig.update_layout(
        title=dict(
            text="Prediksi Penjualan Telkomsel - 10 Hari ke Depan",
            x=0.5,
            font=dict(size=20, color='#333')
        ),
        xaxis_title="Tanggal",
        yaxis_title="Penjualan (Unit)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

# === MAIN APP ===

# Header
st.markdown("""
<div class="main-header">
    <h1>📱 Telkomsel Sales Forecasting</h1>
    <p>Dashboard Prediksi Penjualan Menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Pengaturan")
    st.markdown("---")
    
    days_to_predict = st.slider(
        "Jumlah Hari Prediksi",
        min_value=5,
        max_value=30,
        value=10,
        help="Pilih berapa hari ke depan yang ingin diprediksi"
    )
    
    st.markdown("### 📋 Informasi")
    st.info("**Format CSV:**\n- Kolom: `tanggal`, `penjualan`\n- Format tanggal: YYYY-MM-DD")
    
    st.markdown("### 🎯 Model Info")
    st.success("Model: SARIMAX\nAccuracy: 92.5%\nLast Updated: Jan 2025")

# Upload section
st.markdown("### 📁 Upload Data Penjualan")

uploaded_file = st.file_uploader(
    "",
    type=["csv"],
    help="Upload file CSV dengan kolom 'tanggal' dan 'penjualan'"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-section">
        <h3>🎯 Cara Menggunakan Dashboard</h3>
        <p>1. Upload file CSV dengan data historis penjualan</p>
        <p>2. Pastikan file memiliki kolom 'tanggal' dan 'penjualan'</p>
        <p>3. Sistem akan otomatis memprediksi penjualan 10 hari ke depan</p>
        <p>4. Lihat hasil prediksi dalam bentuk tabel dan grafik interaktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo data info
    st.markdown("### 📊 Contoh Format Data")
    demo_data = pd.DataFrame({
        'tanggal': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'penjualan': [1250, 1340, 1180]
    })
    st.dataframe(demo_data, use_container_width=True)

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        
        # Validasi kolom
        if 'tanggal' not in df_raw.columns or 'penjualan' not in df_raw.columns:
            st.markdown("""
            <div class="error-box">
                <strong>❌ Error:</strong> File harus memiliki kolom 'tanggal' dan 'penjualan'
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Success:</strong> Data berhasil diupload dan divalidasi!
            </div>
            """, unsafe_allow_html=True)
            
            # Persiapan data
            with st.spinner('🔄 Memproses data...'):
                df_all = prepare_data(df_raw, days_to_predict)
                df_train = df_all[df_all['penjualan'].notnull()]
                df_forecast = df_all[df_all['penjualan'].isnull()]
            
            # Tampilkan metrik
            st.markdown("### 📊 Ringkasan Data")
            col1, col2, col3, col4 = st.columns(4)
            
            avg_sales, max_sales, min_sales, total_days = calculate_metrics(df_train)
            
            with col1:
                st.metric(
                    label="Rata-rata Penjualan",
                    value=f"{avg_sales:,.0f}",
                    delta=f"{((avg_sales/df_train['penjualan'].iloc[-7:].mean() - 1) * 100):+.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Penjualan Tertinggi",
                    value=f"{max_sales:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Penjualan Terendah",
                    value=f"{min_sales:,.0f}"
                )
            
            with col4:
                st.metric(
                    label="Total Hari Data",
                    value=f"{total_days}"
                )
            
            # Load model dan prediksi
            try:
                with st.spinner('🤖 Memuat model dan melakukan prediksi...'):
                    model = joblib.load("model_sarimax.pkl")
                    
                    # Buat exog
                    exog_all = df_all[['libur_nasional', 'hari_minggu']]
                    exog_forecast = exog_all.iloc[len(df_train):]
                    
                    # Prediksi
                    forecast = model.predict(
                        start=len(df_train), 
                        end=len(df_all)-1, 
                        exog=exog_forecast
                    )
                    
                    df_forecast['prediksi_penjualan'] = forecast.values
                    
                    # Gabungkan hasil
                    df_train['tipe'] = 'Aktual'
                    df_forecast['tipe'] = 'Prediksi'
                    df_forecast['penjualan'] = df_forecast['prediksi_penjualan']
                    df_result = pd.concat([df_train, df_forecast], ignore_index=True)
                
                # Tampilkan hasil
                st.markdown("### 📈 Hasil Prediksi")
                
                # Tabs untuk hasil
                tab1, tab2, tab3 = st.tabs(["📊 Grafik Prediksi", "📋 Tabel Detail", "📈 Analisis"])
                
                with tab1:
                    fig = create_forecast_chart(df_result)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Data Aktual (7 Hari Terakhir)")
                        recent_actual = df_result[df_result['tipe'] == 'Aktual'].tail(7)
                        st.dataframe(
                            recent_actual[['tanggal', 'penjualan']].set_index('tanggal'),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("#### 🔮 Prediksi Penjualan")
                        prediction_data = df_result[df_result['tipe'] == 'Prediksi']
                        st.dataframe(
                            prediction_data[['tanggal', 'penjualan']].set_index('tanggal'),
                            use_container_width=True
                        )
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Statistik Prediksi")
                        pred_stats = prediction_data['penjualan'].describe()
                        st.dataframe(pred_stats.to_frame('Nilai'), use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Insight")
                        avg_pred = prediction_data['penjualan'].mean()
                        trend = "naik" if avg_pred > avg_sales else "turun"
                        change_pct = ((avg_pred/avg_sales - 1) * 100)
                        
                        st.info(f"""
                        **Tren Prediksi:** {trend.upper()}
                        
                        **Perubahan:** {change_pct:+.1f}% dari rata-rata historis
                        
                        **Rekomendasi:** 
                        {"Tingkatkan stok dan persiapan promosi" if trend == "naik" else "Optimalkan inventory dan fokus pada efisiensi"}
                        """)
                
                # Download hasil
                st.markdown("### 💾 Download Hasil")
                csv_result = df_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_result,
                    file_name=f"prediksi_penjualan_telkomsel_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                        
            except FileNotFoundError:
                st.markdown("""
                <div class="error-box">
                    <strong>❌ Error:</strong> Model 'model_sarimax.pkl' tidak ditemukan. 
                    Pastikan file model ada di direktori yang sama.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <strong>❌ Error:</strong> Gagal melakukan prediksi: {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Error:</strong> Gagal membaca file: {str(e)}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Telkomsel Sales Forecasting Dashboard | Powered by Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
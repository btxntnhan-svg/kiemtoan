import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# 1. CẤU HÌNH TRANG WEB STREAMLIT
st.set_page_config(
    page_title="Hệ thống Phát hiện Giao dịch Bất thường",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS TÙY CHỈNH CHO GIAO DIỆN PREMIUM
st.markdown("""
    <style>
        /* Tùy chỉnh tiêu đề và text */
        .main-title {
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Tùy chỉnh các thẻ KPI */
        .kpi-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            flex: 1;
            background: linear-gradient(135deg, #151b2c 0%, #0b0f19 100%);
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, border-color 0.2s;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: #6366f1;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 0.25rem;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #9ca3af;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Nhãn trạng thái rủi ro */
        .badge {
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-urgent { background-color: #ef4444; color: #ffffff; }
        .badge-high { background-color: #f97316; color: #ffffff; }
        .badge-medium { background-color: #eab308; color: #000000; }
        .badge-low { background-color: #3b82f6; color: #ffffff; }
        .badge-normal { background-color: #10b981; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# 3. HÀM HUẤN LUYỆN VÀ TIỀN XỬ LÝ
@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer):
    """Đọc dữ liệu CSV"""
    df = pd.read_csv(file_path_or_buffer, parse_dates=["transaction_date"], dayfirst=False)
    return df

def preprocess_and_train(df, contamination, n_estimators, random_state):
    """Tiền xử lý và huấn luyện Isolation Forest"""
    df_copy = df.copy()
    
    # 1. Trích xuất giờ giao dịch
    if "transaction_date" in df_copy.columns:
        df_copy["transaction_date"] = pd.to_datetime(df_copy["transaction_date"], errors="coerce")
        df_copy["gio_giao_dich"] = df_copy["transaction_date"].dt.hour
    else:
        st.error("Không tìm thấy cột 'transaction_date' trong dữ liệu!")
        st.stop()
        
    # 2. Trạng thái nhân viên
    if "is_employee" in df_copy.columns:
        df_copy["co nhan vien"] = df_copy["is_employee"].astype(int)
    else:
        df_copy["co nhan vien"] = 0
        
    # 3. Phân loại ngoài giờ (< 6h hoặc > 18h)
    df_copy["ngoai_gio"] = (df_copy["gio_giao_dich"] < 6) | (df_copy["gio_giao_dich"] > 18)
    
    # 4. Trích xuất thuộc tính đưa vào mô hình
    x_features = df_copy[["amount", "gio_giao_dich", "co nhan vien"]].copy()
    
    # 5. Chuẩn hóa thang đo (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_features)
    
    # 6. Huấn luyện Isolation Forest
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    
    # 7. Tính điểm score & dự đoán bất thường
    df_copy["anomaly_score"] = iso.decision_function(X_scaled) # Điểm càng thấp càng bất thường
    df_copy["is_anomaly"] = iso.predict(X_scaled) == -1
    
    # 8. Phân mức độ rủi ro bất thường dựa trên phân vị của nhóm bất thường
    df_anom = df_copy[df_copy["is_anomaly"] == True]
    if len(df_anom) > 0:
        q25 = df_anom["anomaly_score"].quantile(0.25)
        q50 = df_anom["anomaly_score"].quantile(0.50)
        q75 = df_anom["anomaly_score"].quantile(0.75)
    else:
        q25 = q50 = q75 = 0.0
        
    def classify_urgency(row):
        if not row["is_anomaly"]:
            return "Bình thường"
        score = row["anomaly_score"]
        if score < q25:
            return "Khẩn cấp"
        elif score < q50:
            return "Cao"
        elif score < q75:
            return "Trung bình"
        else:
            return "Thấp"
            
    df_copy["risk_level"] = df_copy.apply(classify_urgency, axis=1)
    
    return df_copy, iso, scaler, q25, q50, q75

# 4. GIAO DIỆN CHÍNH
st.markdown('<div class="main-title">🛡️ HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Giải pháp ứng dụng trí tuệ nhân tạo (Isolation Forest) phát hiện các hành vi gian lận và giao dịch bất thường</div>', unsafe_allow_html=True)

# 5. SIDEBAR CẤU HÌNH
st.sidebar.markdown("### ⚙️ Cấu hình Mô hình & Dữ liệu")

# Tải file dữ liệu
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV giao dịch", type=["csv"])

demo_file_path = "transactions_Q1_demo.csv"
use_demo = False

if uploaded_file is None:
    if os.path.exists(demo_file_path):
        st.sidebar.info("💡 Chưa tải file lên. Sử dụng dữ liệu demo mặc định.")
        use_demo = True
    else:
        st.sidebar.warning("⚠️ Không tìm thấy file demo. Hãy tải file CSV lên.")
        st.info("Vui lòng tải tệp dữ liệu giao dịch định dạng CSV từ thanh điều hướng bên trái.")
        st.stop()

# Thiết lập siêu tham số mô hình
contamination = st.sidebar.slider(
    "Tỷ lệ nhiễm bẩn (Contamination)",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
    help="Tỷ lệ giao dịch bất thường dự kiến trong dữ liệu"
)

n_estimators = st.sidebar.slider(
    "Số lượng cây (n_estimators)",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
    help="Số lượng cây quyết định trong mô hình Isolation Forest"
)

random_state = st.sidebar.number_input(
    "Mã hạt giống (Random State)",
    value=42,
    step=1,
    help="Đảm bảo kết quả huấn luyện mô hình giống nhau giữa các lần chạy"
)

# Nút huấn luyện lại mô hình
retrain_clicked = st.sidebar.button("⚡ Huấn luyện lại mô hình", use_container_width=True)

# 6. QUẢN LÝ SESSION STATE ĐỂ TỐI ƯU TỐC ĐỘ PHẢN HỒI (Caching Model & Data)
if "model_trained" not in st.session_state or retrain_clicked:
    st.session_state["model_trained"] = False

# Đọc dữ liệu dựa trên nguồn upload hoặc demo
data_source = uploaded_file if uploaded_file is not None else demo_file_path
df_raw = load_data(data_source)

# Huấn luyện mô hình nếu chưa có hoặc có sự thay đổi tham số/ấn nút retrain
if not st.session_state["model_trained"] or \
   st.session_state.get("last_contamination") != contamination or \
   st.session_state.get("last_n_estimators") != n_estimators or \
   st.session_state.get("last_random_state") != random_state or \
   st.session_state.get("last_data_source") != (uploaded_file.name if uploaded_file else "demo"):

    with st.spinner("🧠 Hệ thống đang xử lý dữ liệu và huấn luyện mô hình Isolation Forest..."):
        df_proc, model, scaler, q25, q50, q75 = preprocess_and_train(
            df_raw, contamination, n_estimators, random_state
        )
        
        # Lưu vào Session State
        st.session_state["df_processed"] = df_proc
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["q25"] = q25
        st.session_state["q50"] = q50
        st.session_state["q75"] = q75
        
        # Lưu các tham số đã chạy để check thay đổi
        st.session_state["last_contamination"] = contamination
        st.session_state["last_n_estimators"] = n_estimators
        st.session_state["last_random_state"] = random_state
        st.session_state["last_data_source"] = uploaded_file.name if uploaded_file else "demo"
        st.session_state["model_trained"] = True
        
        st.toast("Huấn luyện mô hình thành công!", icon="🚀")

# Lấy dữ liệu và mô hình từ session state
df = st.session_state["df_processed"]
q25 = st.session_state["q25"]
q50 = st.session_state["q50"]
q75 = st.session_state["q75"]

# 7. HIỂN THỊ CÁC CHỈ SỐ KPI CHÍNH
total_txns = len(df)
anomalies_df = df[df["is_anomaly"] == True]
total_anomalies = len(anomalies_df)
anomaly_rate = (total_anomalies / total_txns) * 100

total_out_of_hours = df["ngoai_gio"].sum()
out_of_hours_rate = (total_out_of_hours / total_txns) * 100

total_anomaly_amount = anomalies_df["amount"].sum()
urgent_count = df[df["risk_level"] == "Khẩn cấp"].shape[0]

# Định dạng tiền tệ VND
def format_money(val):
    if val >= 1e9:
        return f"{val/1e9:,.2f} tỷ VND"
    elif val >= 1e6:
        return f"{val/1e6:,.2f} triệu VND"
    else:
        return f"{val:,.0f} VND"

# Render KPI bằng HTML để đẹp mắt
st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Tổng giao dịch</div>
            <div class="kpi-value">{total_txns:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Giao dịch bất thường</div>
            <div class="kpi-value" style="color: #ef4444;">{total_anomalies:,.0f} <span style="font-size: 1.1rem; color: #ef4444;">({anomaly_rate:.2f}%)</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Số giao dịch ngoài giờ</div>
            <div class="kpi-value" style="color: #f59e0b;">{total_out_of_hours:,.0f} <span style="font-size: 1.1rem; color: #f59e0b;">({out_of_hours_rate:.2f}%)</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Tổng tiền bất thường</div>
            <div class="kpi-value" style="color: #ef4444; font-size: 1.5rem; line-height: 2.1rem;">{format_money(total_anomaly_amount)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Số ca Rủi ro Khẩn cấp</div>
            <div class="kpi-value" style="color: #ec4899;">{urgent_count:,.0f}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 8. CÁC TAB CHỨC NĂNG CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Phân tích & Thống kê", 
    "🔮 Trực quan hóa 3D Không gian rủi ro", 
    "🔍 Bộ lọc & Tra cứu giao dịch",
    "🧪 Trình kiểm tra giao dịch đơn lẻ"
])

# ==================== TAB 1: PHÂN TÍCH & THỐNG KÊ ====================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Phân phối số lượng giao dịch theo giờ trong ngày")
        
        # Thống kê giao dịch theo giờ
        hour_counts_all = df['gio_giao_dich'].value_counts().sort_index()
        hour_counts_anom = anomalies_df['gio_giao_dich'].value_counts().sort_index()
        
        fig_hour = go.Figure()
        # Biểu đồ cột tổng số giao dịch
        fig_hour.add_trace(go.Bar(
            x=hour_counts_all.index,
            y=hour_counts_all.values,
            name='Tất cả giao dịch',
            marker_color='#3b82f6',
            opacity=0.75
        ))
        # Biểu đồ cột giao dịch bất thường
        fig_hour.add_trace(go.Bar(
            x=hour_counts_anom.index,
            y=hour_counts_anom.values,
            name='Giao dịch bất thường',
            marker_color='#ef4444',
            yaxis='y2'
        ))
        
        # Cấu hình 2 trục Y để trực quan rõ ràng hơn (vì số giao dịch bất thường nhỏ hơn nhiều)
        fig_hour.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Giờ trong ngày', tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(title='Tổng số giao dịch', titlefont=dict(color='#3b82f6'), tickfont=dict(color='#3b82f6')),
            yaxis2=dict(
                title='Số giao dịch bất thường',
                titlefont=dict(color='#ef4444'),
                tickfont=dict(color='#ef4444'),
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode='x unified'
        )
        
        # Vẽ các vùng ngoài giờ (trước 6h và sau 18h)
        fig_hour.add_vrect(x0=-0.5, x1=5.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Ngoài giờ", annotation_position="top left")
        fig_hour.add_vrect(x0=18.5, x1=23.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Ngoài giờ", annotation_position="top right")
        
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col2:
        st.subheader("💳 Thống kê giao dịch bất thường theo Kênh (Channel)")
        
        channel_stats = df.groupby(['channel', 'is_anomaly']).size().unstack(fill_value=0)
        channel_stats['Tỷ lệ bất thường (%)'] = (channel_stats[True] / (channel_stats[True] + channel_stats[False])) * 100
        channel_stats = channel_stats.reset_index()
        
        fig_channel = px.bar(
            channel_stats,
            x='channel',
            y=True,
            text=channel_stats[True].apply(lambda x: f"{x:,.0f} ca"),
            labels={True: 'Số lượng bất thường', 'channel': 'Kênh giao dịch'},
            title='Số ca bất thường theo kênh giao dịch',
            color='channel',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_channel.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig_channel.update_traces(textposition='outside')
        st.plotly_chart(fig_channel, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📍 Thống kê giao dịch bất thường theo Chi nhánh (Location)")
        loc_stats = anomalies_df['location'].value_counts().reset_index()
        fig_loc = px.bar(
            loc_stats,
            y='location',
            x='count',
            orientation='h',
            labels={'count': 'Số ca bất thường', 'location': 'Chi nhánh'},
            color='count',
            color_continuous_scale='Reds'
        )
        fig_loc.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_loc, use_container_width=True)
        
    with col4:
        st.subheader("💸 Phân phối số tiền giao dịch (Bình thường vs Bất thường)")
        fig_amount = go.Figure()
        
        # Nhóm bình thường
        fig_amount.add_trace(go.Box(
            y=df[df['is_anomaly'] == False]['amount'],
            name='Bình thường',
            marker_color='#10b981',
            boxpoints='outliers'
        ))
        # Nhóm bất thường
        fig_amount.add_trace(go.Box(
            y=anomalies_df['amount'],
            name='Bất thường',
            marker_color='#ef4444',
            boxpoints='all'
        ))
        
        fig_amount.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_type="log", # Dùng Log-scale do số tiền chênh lệch quá lớn
            yaxis_title="Số tiền giao dịch (Log Scale - VND)",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_amount, use_container_width=True)

# ==================== TAB 2: TRỰC QUAN HÓA 3D ====================
with tab2:
    st.subheader("🔮 Bản đồ Không gian phân tách giao dịch bất thường 3D")
    st.write("Biểu đồ 3D tương tác biểu diễn mối tương quan giữa **Số tiền giao dịch (Amount)**, **Giờ giao dịch (Hour)**, và **Đối tượng Nhân viên (Is Employee)**. Các điểm có màu sắc khác nhau biểu thị mức độ rủi ro do Isolation Forest tính toán.")
    
    # Chuẩn bị dữ liệu vẽ 3D
    plot_df = df.copy()
    # Rút gọn nhãn hiển thị cho 3D để đỡ rối
    plot_df['Risk Status'] = plot_df['risk_level']
    plot_df['amount_million'] = plot_df['amount'] / 1e6
    
    # Tạo 3D Scatter plot sử dụng Plotly
    fig_3d = px.scatter_3d(
        plot_df,
        x='gio_giao_dich',
        y='amount_million',
        z='co nhan vien',
        color='Risk Status',
        color_discrete_map={
            "Bình thường": "#10b981",
            "Thấp": "#3b82f6",
            "Trung bình": "#eab308",
            "Cao": "#f97316",
            "Khẩn cấp": "#ef4444"
        },
        category_orders={"Risk Status": ["Bình thường", "Thấp", "Trung bình", "Cao", "Khẩn cấp"]},
        labels={
            'gio_giao_dich': 'Giờ giao dịch',
            'amount_million': 'Số tiền (Triệu VND)',
            'co nhan vien': 'Là Nhân viên (0=Không, 1=Có)',
            'Risk Status': 'Trạng thái rủi ro'
        },
        opacity=0.7,
        size=np.where(plot_df['is_anomaly'], 6, 2), # Cho điểm bất thường to hơn điểm thường
        hover_data={
            'transaction_id': True,
            'amount': ':,.0f',
            'gio_giao_dich': True,
            'co nhan vien': True,
            'Risk Status': True
        }
    )
    
    fig_3d.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="gray", showbackground=True, type="log"), # Log scale cho số tiền
            zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="gray", showbackground=True, tickvals=[0, 1])
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=650
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# ==================== TAB 3: BỘ LỌC & TRA CỨU GIAO DỊCH ====================
with tab3:
    st.subheader("🔍 Danh sách tra cứu chi tiết giao dịch")
    
    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        risk_filter = st.selectbox(
            "Lọc theo mức độ rủi ro",
            options=["Tất cả", "Khẩn cấp", "Cao", "Trung bình", "Thấp", "Bình thường"]
        )
    with col_f2:
        type_filter = st.selectbox(
            "Lọc loại giao dịch",
            options=["Tất cả"] + list(df['transaction_type'].dropna().unique())
        )
    with col_f3:
        search_query = st.text_input("Tìm kiếm mã giao dịch/Khách hàng", placeholder="Nhập mã...")
    with col_f4:
        amount_range = st.slider(
            "Lọc theo số tiền (Triệu VND)",
            min_value=0,
            max_value=int(df['amount'].max() / 1e6),
            value=(0, int(df['amount'].max() / 1e6))
        )
        
    # Áp dụng bộ lọc
    filtered_df = df.copy()
    
    if risk_filter != "Tất cả":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        
    if type_filter != "Tất cả":
        filtered_df = filtered_df[filtered_df['transaction_type'] == type_filter]
        
    if search_query:
        search_query = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df['transaction_id'].str.lower().str.contains(search_query) |
            filtered_df['customer_id_hash'].str.lower().str.contains(search_query) |
            filtered_df['account_no_hash'].str.lower().str.contains(search_query)
        ]
        
    # Lọc số tiền
    filtered_df = filtered_df[
        (filtered_df['amount'] >= amount_range[0] * 1e6) &
        (filtered_df['amount'] <= amount_range[1] * 1e6)
    ]
    
    st.write(f"Tìm thấy **{len(filtered_df):,}** giao dịch tương ứng với bộ lọc.")
    
    # Hiển thị bảng dữ liệu định dạng đẹp
    display_cols = [
        "transaction_id", "transaction_date", "customer_id_hash", "account_no_hash",
        "amount", "transaction_type", "channel", "counterparty_bank", "location",
        "is_employee", "gio_giao_dich", "risk_level"
    ]
    
    # Tạo style hiển thị rủi ro
    def color_risk(val):
        if val == "Khẩn cấp": return "color: #ef4444; font-weight: bold;"
        elif val == "Cao": return "color: #f97316; font-weight: bold;"
        elif val == "Trung bình": return "color: #eab308; font-weight: bold;"
        elif val == "Thấp": return "color: #3b82f6;"
        else: return "color: #10b981;"
        
    styled_df = filtered_df[display_cols].copy()
    styled_df['amount'] = styled_df['amount'].apply(lambda x: f"{x:,.0f} VND")
    
    st.dataframe(
        styled_df.style.map(color_risk, subset=['risk_level']),
        use_container_width=True,
        height=400
    )
    
    # Xuất báo cáo CSV/Excel
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        # Nút Download CSV
        csv_buffer = io.StringIO()
        filtered_df[display_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Tải xuống kết quả lọc (CSV)",
            data=csv_buffer.getvalue(),
            file_name="giao_dich_bat_thuong_loc.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_exp2:
        # Nút Download Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered_df[display_cols].to_excel(writer, index=False, sheet_name="GiaoDich")
        st.download_button(
            label="📥 Tải xuống kết quả lọc (Excel)",
            data=excel_buffer.getvalue(),
            file_name="giao_dich_bat_thuong_loc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ==================== TAB 4: TRÌNH KIỂM TRA ĐƠN LẺ ====================
with tab4:
    st.subheader("🧪 Kiểm tra thời gian thực một giao dịch đơn lẻ")
    st.write("Nhập thông số của một giao dịch giả lập bên dưới để xem mô hình Isolation Forest đánh giá mức độ rủi ro.")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### 📝 Nhập thông số giao dịch")
        test_amount = st.number_input("Số tiền giao dịch (VND)", min_value=1000, max_value=10000000000, value=5000000, step=500000)
        test_hour = st.slider("Giờ thực hiện giao dịch", min_value=0, max_value=23, value=14)
        test_employee = st.checkbox("Đối tượng thực hiện là nhân viên ngân hàng (Is Employee)")
        
        check_clicked = st.button("🔍 Tiến hành kiểm tra rủi ro", type="primary", use_container_width=True)
        
    with col_t2:
        st.markdown("#### 🎯 Kết quả đánh giá mô hình")
        if check_clicked:
            # 1. Trích xuất thuộc tính
            co_nv = 1 if test_employee else 0
            single_x = pd.DataFrame({
                "amount": [test_amount],
                "gio_giao_dich": [test_hour],
                "co nhan vien": [co_nv]
            })
            
            # 2. Scale
            scaler = st.session_state["scaler"]
            single_x_scaled = scaler.transform(single_x)
            
            # 3. Predict & Score
            model = st.session_state["model"]
            score = model.decision_function(single_x_scaled)[0]
            is_anomaly = model.predict(single_x_scaled)[0] == -1
            
            # 4. Xác định mức rủi ro theo các ngưỡng phân vị trong tập train
            if not is_anomaly:
                st.success("### ✅ Giao dịch Bình thường (An toàn)")
                st.markdown(f"""
                    - **Điểm số bất thường (Anomaly Score):** `{score:.4f}` (Điểm dương thể hiện giao dịch thuộc phân phối hành vi thông thường).
                    - **Đánh giá:** Giao dịch có mức độ tương thích cao với phân phối lịch sử. Không cần xử lý bổ sung.
                """)
            else:
                # Phân cấp mức rủi ro
                risk_level = "Thấp"
                color_hex = "#3b82f6"
                q25_v = st.session_state["q25"]
                q50_v = st.session_state["q50"]
                q75_v = st.session_state["q75"]
                
                if score < q25_v:
                    risk_level = "Khẩn cấp"
                    color_hex = "#ef4444"
                elif score < q50_v:
                    risk_level = "Cao"
                    color_hex = "#f97316"
                elif score < q75_v:
                    risk_level = "Trung bình"
                    color_hex = "#eab308"
                    
                st.markdown(f"""
                    <div style="background-color: {color_hex}22; padding: 1.5rem; border-left: 6px solid {color_hex}; border-radius: 8px; margin-bottom: 1rem;">
                        <h3 style="color: {color_hex}; margin: 0 0 0.5rem 0;">⚠️ Phát hiện Bất thường - Rủi ro: {risk_level}</h3>
                        <p style="margin: 0; font-size: 1.05rem;">
                            <b>Điểm số bất thường (Anomaly Score):</b> <code>{score:.4f}</code> (Điểm số âm thể hiện độ sai lệch lớn khỏi hành vi thông thường).
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Hiển thị giải thích nguyên nhân
                st.markdown("#### 🕵️ Phân tích lý do bất thường:")
                
                reasons = []
                
                # Check số tiền giao dịch
                mean_amount = df['amount'].mean()
                q99_amount = df['amount'].quantile(0.99)
                if test_amount > q99_amount:
                    reasons.append(f"🚩 **Số tiền giao dịch cực lớn**: {format_money(test_amount)} (Vượt ngưỡng 99% lịch sử: {format_money(q99_amount)}).")
                elif test_amount > mean_amount * 5:
                    reasons.append(f"🚩 **Số tiền giao dịch cao**: Gấp hơn {test_amount/mean_amount:.1f} lần so với giá trị trung bình ({format_money(mean_amount)}).")
                    
                # Check ngoài giờ
                if test_hour < 6 or test_hour > 18:
                    reasons.append(f"🚩 **Thực hiện ngoài giờ hành chính**: Giao dịch diễn ra lúc `{test_hour}:00` (Khung giờ ngoài hành chính quy định trước 6h và sau 18h).")
                    
                # Check nhân viên thực hiện số tiền lớn hoặc ngoài giờ
                if test_employee:
                    reasons.append("🚩 **Tài khoản nhân viên**: Giao dịch được thực hiện bởi nhân viên ngân hàng. Các giao dịch nhân viên có yêu cầu giám sát kiểm toán nghiêm ngặt hơn.")
                    if test_hour < 6 or test_hour > 18:
                        reasons.append("🚩 **Nhân viên hoạt động ngoài giờ**: Nhân viên thực hiện giao dịch vào giờ nghỉ là một dấu hiệu bất thường cao.")
                        
                if not reasons:
                    reasons.append("🚩 Sự kết hợp của các yếu tố (số tiền, giờ giao dịch, trạng thái nhân viên) tạo ra cấu hình đặc trưng khác biệt hoàn toàn với hành vi thông thường trong quá khứ.")
                    
                for r in reasons:
                    st.write(r)
                    
                # Đề xuất xử lý
                st.markdown("#### 🛡️ Khuyến nghị hành động:")
                if risk_level == "Khẩn cấp":
                    st.error("🚨 **KHẨN CẤP:** Đóng băng tạm thời tài khoản và thực hiện cuộc gọi xác minh trực tiếp với khách hàng ngay lập tức.")
                elif risk_level == "Cao":
                    st.warning("⚠️ **CẢNH BÁO CAO:** Yêu cầu xác thực OTP/biometric nâng cao hoặc chuyển cho bộ phận kiểm soát rủi ro thẩm định lại.")
                elif risk_level == "Trung bình":
                    st.info("ℹ️ **CẢNH BÁO TRUNG BÌNH:** Ghi nhật ký hệ thống và gửi thông báo kiểm tra giao dịch đến ứng dụng di động của khách hàng.")
                else:
                    st.info("ℹ️ **CẢNH BÁO THẤP:** Tiếp tục theo dõi các giao dịch kế tiếp của tài khoản này.")
        else:
            st.info("Nhấp nút 'Tiến hành kiểm tra rủi ro' ở cột bên trái để phân tích.")

import streamlit as st
import pandas as pd
from function import get_data, get_data_count

st.set_page_config(page_title="📊 API 데이터 모니터링", layout="wide")

st.title("📡 API 데이터 모니터링 대시보드")

# ✅ 상단에 총 데이터 건수 출력
total_count = get_data_count()
st.markdown(f"**총 데이터 건수:** `{total_count}`건")

# 📅 날짜 필터 UI
st.markdown("#### 기간 필터")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작일", value=None)
with col2:
    end_date = st.date_input("종료일", value=None)

# 🔍 데이터 불러오기
if start_date and end_date:
    df = get_data(start_date=start_date.isoformat(), end_date=end_date.isoformat())
else:
    df = get_data()

# 🧾 데이터 테이블 출력
st.dataframe(df)

# 📥 CSV 다운로드 버튼
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)
st.download_button(
    label="CSV 다운로드",
    data=csv,
    file_name="traffic_api_data.csv",
    mime='text/csv'
)

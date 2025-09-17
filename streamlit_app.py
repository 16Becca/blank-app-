import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# -----------------------------
# 공통 설정
# -----------------------------
st.set_page_config(page_title="실내 vs 실외 공기질 대시보드", layout="wide")

try:
    st.markdown("""
        <style>
        @font-face {
            font-family: 'Pretendard';
            src: url('/fonts/Pretendard-Bold.ttf');
        }
        html, body, [class*="css"]  {
            font-family: 'Pretendard', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
except:
    pass


# -----------------------------
# 유틸 함수
# -----------------------------
@st.cache_data
def preprocess_df(df: pd.DataFrame, date_col="date", value_col="value", group_col="group") -> pd.DataFrame:
    """데이터 전처리: 날짜 변환, 미래 데이터 제거"""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        today = pd.Timestamp(datetime.now().date())
        df = df[df[date_col] <= today]
    df = df.dropna().drop_duplicates()
    return df


@st.cache_data
def fetch_public_aqi_data(city="Seoul", country="KR", limit=100):
    """
    OpenAQ API 호출
    출처: https://docs.openaq.org/
    """
    url = f"https://api.openaq.org/v2/measurements?city={city}&country={country}&limit={limit}&sort=desc&order_by=datetime"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json().get("results", [])
        if not data:
            raise ValueError("No data")
        records = []
        for row in data:
            records.append({
                "date": row["date"]["utc"],
                "value": row["value"],
                "group": row["parameter"]
            })
        return pd.DataFrame(records)
    except Exception:
        return None


def generate_example_public_data():
    """공개 데이터 API 실패 시 예시 데이터"""
    dates = pd.date_range(end=datetime.today(), periods=30)
    return pd.DataFrame({
        "date": dates,
        "value": np.random.randint(20, 80, size=30),
        "group": "PM2.5"
    })


def generate_user_dataset_from_prompt():
    """보고서 내용을 반영한 가상의 실내/실외 데이터셋"""
    dates = pd.date_range(end=datetime.today(), periods=30)
    data = []
    for d in dates:
        # 실외
        data.append({"date": d, "value": np.random.randint(50, 100), "group": "실외_관측소"})
        # 실내: 교실, 가정
        data.append({"date": d, "value": np.random.randint(20, 60), "group": "실내_교실"})
        data.append({"date": d, "value": np.random.randint(15, 55), "group": "실내_가정"})
    return pd.DataFrame(data)


# -----------------------------
# 메인 화면
# -----------------------------
st.title("🌍 실내 vs 실외 공기질 데이터 대시보드")

tabs = st.tabs(["공개 데이터 기반", "사용자 입력 기반"])


# -----------------------------
# (1) 공개 데이터 기반 대시보드
# -----------------------------
with tabs[0]:
    st.header("공개 데이터 기반 대시보드 (OpenAQ)")

    df_public = fetch_public_aqi_data()
    if df_public is None or df_public.empty:
        st.warning("공개 API(OpenAQ) 호출에 실패하거나 데이터가 없습니다. 예시(대체) 데이터를 사용합니다.")
        df_public = generate_example_public_data()

    df_public = preprocess_df(df_public)

    # 시계열 꺾은선 그래프
    st.subheader("PM2.5 추이")
    fig_pub = px.line(df_public, x="date", y="value", color="group",
                      labels={"date": "날짜", "value": "농도(µg/m³)", "group": "항목"})
    fig_pub.update_layout(title="서울 PM2.5 (OpenAQ)", font_family="Pretendard")
    st.plotly_chart(fig_pub, use_container_width=True)

    # CSV 다운로드
    st.download_button(
        label="📥 데이터 다운로드 (CSV)",
        data=df_public.to_csv(index=False).encode("utf-8"),
        file_name="public_aqi.csv",
        mime="text/csv"
    )


# -----------------------------
# (2) 사용자 입력 기반 대시보드
# -----------------------------
with tabs[1]:
    st.header("사용자 입력 기반 대시보드 — 실내 vs 실외 공기질 비교")

    # 이미지 추가
    st.subheader("실내 오염원 예시 그림")
    st.image("0a05e60d-b696-425b-9011-6c8ca5c29310.png", use_column_width=True)

    # 데이터 생성 & 전처리
    df_user_raw = generate_user_dataset_from_prompt()
    df_user = preprocess_df(df_user_raw, date_col="date", value_col="value", group_col="group")

    # 실내/실외 카테고리 분류
    df_user["category"] = df_user["group"].apply(lambda g: "실내" if "실내" in g else "실외")

    # 상단: 시간에 따른 평균 비교
    st.subheader("시간에 따른 실내 vs 실외 평균 추이")
    df_line = df_user.groupby(["date", "category"])["value"].mean().reset_index()
    fig_line = px.line(df_line, x="date", y="value", color="category",
                       labels={"date": "날짜", "value": "PM2.5 (µg/m³)", "category": "구분"})
    fig_line.update_layout(title="평균 PM2.5 (실내 vs 실외)", font_family="Pretendard")
    st.plotly_chart(fig_line, use_container_width=True)

    # 중간: 도넛 차트 (최근 평균값)
    st.subheader("최근 30일 평균값 비교")
    avg_vals = df_user.groupby("category")["value"].mean().reset_index()
    fig_pie = px.pie(avg_vals, names="category", values="value", hole=0.5,
                     color="category", title="실내 vs 실외 평균 PM2.5")
    st.plotly_chart(fig_pie, use_container_width=True)

    # 하단: 세부 그룹 (교실/가정 vs 실외 관측소)
    st.subheader("세부 그룹별 공기질 추이")
    fig_detail = px.line(df_user, x="date", y="value", color="group",
                         labels={"date": "날짜", "value": "PM2.5 (µg/m³)", "group": "구분"})
    fig_detail.update_layout(title="교실·가정 vs 지역 관측소", font_family="Pretendard")
    st.plotly_chart(fig_detail, use_container_width=True)

    # CSV 다운로드
    st.download_button(
        label="📥 데이터 다운로드 (CSV)",
        data=df_user.to_csv(index=False).encode("utf-8"),
        file_name="user_aqi.csv",
        mime="text/csv"
    )

# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터 대시보드: OpenAQ (실시간/역사적 대기질), World Bank (연평균 PM2.5) 활용 예시
  - OpenAQ API: https://api.openaq.org/ (Docs: https://docs.openaq.org/resources/measurements) . Refer: turn0search0, turn0search6, turn0search9
  - World Bank PM2.5 indicator: EN.ATM.PM25.MC.M3 (Data page: https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3). Refer: turn0search1, turn0search4
  - WHO Ambient Air Quality Database: https://www.who.int/data/gho/data/themes/air-pollution/who-air-quality-database. Refer: turn0search2, turn0search5
- 사용자 입력(프롬프트 텍스트) 대시보드: 제공된 보고서 본문(입력 섹션)을 기반으로 교실별 예시 실내/실외 PM2.5 데이터 생성 및 시각화 (앱 실행 중 파일 업로드 불필요)
- 구현 규칙 준수:
  - 표준화 컬럼: date, value, group (옵션)
  - @st.cache_data 사용
  - 미래 데이터(로컬 자정 이후) 제거
  - API 실패 시 예시 데이터로 자동 대체 & 화면 안내
  - 전처리된 표 CSV 다운로드 제공
- 폰트: /fonts/Pretendard-Bold.ttf 적용 시도 (없으면 자동 생략)
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta, timezone
import pytz
from dateutil import parser

import plotly.express as px

# ---------------------------
# 기본 설정 (한국어)
# ---------------------------
st.set_page_config(page_title="실내 vs 실외 공기질 대시보드", layout="wide")

# Attempt to load local Pretendard font for Streamlit/Plotly/Matplotlib
def inject_pretendard_css():
    """
    시도: Pretendard-Bold.ttf를 /fonts 폴더에서 불러와 CSS로 적용
    (없으면 자연스럽게 무시됨)
    """
    try:
        import os
        font_path = "/fonts/Pretendard-Bold.ttf"
        if os.path.exists(font_path):
            # CSS 적용 (Streamlit 마진 내)
            st.markdown(
                f"""
                <style>
                @font-face {{
                    font-family: 'Pretendard';
                    src: url('file://{font_path}') format('truetype');
                    font-weight: 700;
                    font-style: normal;
                }}
                html, body, [class*="css"]  {{
                    font-family: 'Pretendard', sans-serif;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            # Plotly 글꼴 기본값 설정 (각 그래프에서 layout.font.family로 재설정 가능)
        else:
            # 파일 없으면 그냥 통과
            pass
    except Exception:
        pass

inject_pretendard_css()

# Helper: Asia/Seoul today (local)
SEOUL_TZ = pytz.timezone("Asia/Seoul")
def seoul_today_date() -> datetime.date:
    return datetime.now(SEOUL_TZ).date()

# ---------------------------
# 캐싱된 API 호출 함수 (재시도 포함)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_openaq_measurements(country: str = "KR", parameter: str = "pm25", limit: int = 1000, days: int = 7):
    """
    OpenAQ measurements API 호출 (간단 재시도 로직 포함).
    반환: pandas DataFrame (columns: date (datetime), value (float), group (location), latitude, longitude, unit)
    참고: API docs: https://docs.openaq.org/resources/measurements
    """
    base = "https://api.openaq.org/v2/measurements"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    params = {
        "country": country,
        "parameter": parameter,
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "limit": limit,
        "page": 1,
        "sort": "desc",
    }
    attempts = 3
    for i in range(attempts):
        try:
            resp = requests.get(base, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                if not data:
                    return pd.DataFrame()  # empty
                rows = []
                for r in data:
                    # r['date'] contains utc and local; use local if available
                    dt_local = r.get("date", {}).get("local") or r.get("date", {}).get("utc")
                    try:
                        dt = parser.isoparse(dt_local)
                    except Exception:
                        dt = None
                    rows.append({
                        "date": dt,
                        "value": r.get("value"),
                        "group": r.get("location"),
                        "parameter": r.get("parameter"),
                        "unit": r.get("unit"),
                        "latitude": r.get("coordinates", {}).get("latitude"),
                        "longitude": r.get("coordinates", {}).get("longitude"),
                        "city": r.get("city"),
                        "country": r.get("country"),
                    })
                df = pd.DataFrame(rows)
                return df
            else:
                # non-200
                continue
        except Exception:
            continue
    # 실패하면 빈 DF returned to signal fallback required
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_worldbank_pm25(country_codes: list[str] = ["KOR","CHN","IND"], years: list[int] = [2015,2018,2020,2022]):
    """
    World Bank WDI에서 EN.ATM.PM25.MC.M3 지표(연평균 PM2.5) 간단 조회 (REST API)
    참고: https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3
    This uses the World Bank API endpoint for indicators:
    http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date=YYYY:YYYY&format=json&per_page=100
    """
    indicator = "EN.ATM.PM25.MC.M3"
    rows = []
    for cc in country_codes:
        # Request for a range of years
        year_min = min(years)
        year_max = max(years)
        url = f"http://api.worldbank.org/v2/country/{cc}/indicator/{indicator}"
        params = {"date": f"{year_min}:{year_max}", "format":"json", "per_page":100}
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                j = resp.json()
                if isinstance(j, list) and len(j) >= 2:
                    for entry in j[1]:
                        year = int(entry.get("date"))
                        val = entry.get("value")
                        if val is not None:
                            rows.append({
                                "country": entry.get("country", {}).get("value"),
                                "countryiso3": cc,
                                "year": year,
                                "value": float(val),
                            })
        except Exception:
            continue
    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        return pd.DataFrame()

# ---------------------------
# 예시 데이터 (API 실패 시 대체)
# ---------------------------
def generate_example_openaq_example():
    """
    API 실패 또는 빈 응답 시 사용할 예시 데이터 (한국의 가상의 관측소 3개, 7일치 hourly)
    표준화: date, value, group, latitude, longitude, unit
    """
    now = datetime.now(SEOUL_TZ)
    end = now
    start = end - timedelta(days=6)
    periods = 24 * 7
    times = pd.date_range(start=start - timedelta(hours=start.hour, minutes=start.minute, seconds=start.second),
                          periods=periods, freq="H", tz=SEOUL_TZ)
    groups = ["교실_A(실내)", "교실_B(실내)", "관측소_서울시_중구(실외)"]
    records = []
    rng = np.random.default_rng(42)
    for g in groups:
        base = 10 if "실내" in g else 25
        for t in times:
            # introduce daily pattern + noise
            hour = t.hour
            diurnal = 5 * np.sin((hour / 24) * 2 * np.pi)
            val = max(0.5, base + diurnal + rng.normal(0, 5))
            records.append({
                "date": t.to_pydatetime(),
                "value": round(float(val),2),
                "group": g,
                "unit": "µg/m³",
                "latitude": 37.56 + rng.normal(0, 0.005),
                "longitude": 126.97 + rng.normal(0, 0.005),
                "city":"Seoul",
                "country":"KR"
            })
    df = pd.DataFrame(records)
    return df

def generate_user_dataset_from_prompt():
    """
    사용자 입력(프롬프트 본문)을 바탕으로 생성한 '학급별 실내 vs 실외 PM2.5' 예시 데이터.
    - 5교시(08:00-15:00) 동안 30분 간격 측정, 5일치(평일)
    - groups: '교실1', '교실2', '가정(실내)', '지역(실외)'
    표준화: date, value, group
    """
    # school days: 최근 주중 5일 (서울 기준)
    today = datetime.now(SEOUL_TZ).date()
    # find last Monday
    td = today
    # ensure weekday 0-4
    # choose last 7 days and then pick five most recent weekdays
    days = []
    for d in range(1, 15):
        candidate = today - timedelta(days=d)
        if candidate.weekday() < 5:  # Mon-Fri
            days.append(candidate)
        if len(days) >= 5:
            break
    days = sorted(days)
    records = []
    rng = np.random.default_rng(123)
    for day in days:
        for hh in range(8, 16):  # 08:00 - 15:00
            for minute in [0,30]:
                dt = datetime(day.year, day.month, day.day, hh, minute, tzinfo=SEOUL_TZ)
                # generate values
                outdoor = max(5, 20 + 10*np.sin((hh-8)/8*2*np.pi) + rng.normal(0,3))
                classroom1 = max(5, outdoor*0.6 + 5 + rng.normal(0,2))
                classroom2 = max(5, outdoor*0.8 + rng.normal(0,2))
                home = max(5, outdoor*0.9 + 3 + rng.normal(0,2))
                records += [
                    {"date": dt, "value": round(float(classroom1),2), "group":"교실 1 (실내)"},
                    {"date": dt, "value": round(float(classroom2),2), "group":"교실 2 (실내)"},
                    {"date": dt, "value": round(float(home),2), "group":"가정 (실내)"},
                    {"date": dt, "value": round(float(outdoor),2), "group":"지역 관측소 (실외)"},
                ]
    df = pd.DataFrame(records)
    return df

# ---------------------------
# 유틸: 미래 데이터 제거 / 표준화 / 전처리
# ---------------------------
def preprocess_df(df: pd.DataFrame, date_col: str = "date", value_col: str = "value", group_col: str = "group"):
    """
    - date 형변환
    - 결측치 처리 (value 결측은 제거)
    - 중복 제거
    - 미래 데이터(서울 자정 이후) 제거
    - 표준화 컬럼명: date, value, group (group optional)
    """
    df = df.copy()
    # rename if necessary
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if value_col != "value":
        df = df.rename(columns={value_col: "value"})
    if group_col != "group" and group_col in df.columns:
        df = df.rename(columns={group_col: "group"})
    # date parsing
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            # attempt manual parsing with dateutil
            df["date"] = df["date"].apply(lambda x: parser.parse(x) if pd.notnull(x) else pd.NaT)
    # drop rows with missing value/date
    df = df.dropna(subset=["date","value"])
    # ensure numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    # remove duplicates
    df = df.drop_duplicates()
    # remove future data: compare date.date() > seoul_today_date() -> drop
    today = seoul_today_date()
    # ensure timezone aware handling: convert to SEOUL_TZ if naive assume local
    def _date_to_seoul(d):
        if d.tzinfo is None:
            return SEOUL_TZ.localize(d)
        else:
            return d.astimezone(SEOUL_TZ)
    df["date"] = df["date"].apply(_date_to_seoul)
    df = df[df["date"].dt.date <= today]
    # sort
    df = df.sort_values("date").reset_index(drop=True)
    # keep only relevant columns and fill group if missing
    if "group" not in df.columns:
        df["group"] = "전체"
    # standard columns preserved: date, value, group
    return df[["date","value","group"] + [c for c in df.columns if c not in ["date","value","group"]]]

# ---------------------------
# UI: 사이드바 기본
# ---------------------------
st.sidebar.header("데시보드 설정")
# 날짜 범위 필터 기본값 (공개 데이터용)
default_days = st.sidebar.slider("공개 데이터 조회 기간(일)", min_value=1, max_value=30, value=7)

# ---------------------------
# 메인: 탭 구성 (공개 데이터 / 사용자 입력)
# ---------------------------
tabs = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력 기반 대시보드 (보고서 기반 예시)"])

# ---------------------------
# 탭 1: 공개 데이터 대시보드
# ---------------------------
with tabs[0]:
    st.header("공식 공개 데이터 대시보드 (OpenAQ + World Bank 예시)")
    st.markdown(
        """
        **설명:** OpenAQ의 관측치(실시간/역사적)와 World Bank의 연평균 PM2.5(연도별)를 사용해 실외 공기질을 시각화합니다.
        - API 호출 실패 시, 예시 데이터로 자동 대체됩니다(화면 안내 표시).
        - 출처는 코드 주석에 명시되어 있습니다.
        """
    )
    col1, col2 = st.columns([3,1])
    with col2:
        st.subheader("설정")
        country_code = st.selectbox("조회 국가 코드 선택", options=["KR","IN","CN","US","FI"], index=0, help="ISO 3166-1 alpha-2")
        parameter = st.selectbox("측정 파라미터", options=["pm25","pm10","no2","o3"], index=0)
        limit = st.slider("최대 레코드 수", min_value=100, max_value=5000, value=1000, step=100)
        st.caption("OpenAQ API 문서: https://docs.openaq.org/resources/measurements")

    # Fetch OpenAQ
    with st.spinner("OpenAQ 데이터 가져오는 중..."):
        df_openaq_raw = fetch_openaq_measurements(country=country_code, parameter=parameter, limit=limit, days=default_days)

    # If API returned empty -> show fallback with example and note
    if df_openaq_raw.empty:
        st.warning("공개 API(OpenAQ) 호출에 실패하거나 데이터가 없습니다. 예시(대체) 데이터를 사용합니다. (원본 API: https://api.openaq.org/)")
        df_op = generate_example_openaq_example()
    else:
        df_op = df_openaq_raw.copy()

    # Preprocess
    df_op = preprocess_df(df_op, date_col="date", value_col="value", group_col="group")

    st.subheader("원시/전처리된 표 (OpenAQ 기반)")
    st.write("아래 표는 전처리(결측/중복/미래 데이터 제거)된 결과입니다.")
    st.dataframe(df_op.head(200))

    # CSV 다운로드
    csv_buf = io.BytesIO()
    df_op.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button("전처리된 OpenAQ 데이터 CSV 다운로드", data=csv_buf, file_name="openaq_preprocessed.csv", mime="text/csv")

    # 시계열: 주요 관측소별 PM2.5 (혹은 선택한 parameter)
    st.subheader("시계열: 관측소별 값")
    # allow user to choose group(s)
    groups = df_op["group"].unique().tolist()
    sel_groups = st.multiselect("관측소(그룹) 선택", options=groups, default=groups[:5])
    if not sel_groups:
        st.info("관측소를 선택하세요.")
    else:
        df_ts = df_op[df_op["group"].isin(sel_groups)]
        if df_ts.empty:
            st.info("선택된 관측소에 데이터가 없습니다.")
        else:
            fig = px.line(df_ts, x="date", y="value", color="group", labels={"date":"일시","value":"농도 (µg/m³)","group":"관측소"})
            fig.update_layout(title=f"{parameter.upper()} 시계열 ({country_code})", legend_title="관측소", font_family="Pretendard")
            st.plotly_chart(fig, use_container_width=True)

    # 지도: 관측소 위치 (있을 때)
    st.subheader("지도: 관측소 위치 (있을 경우)")
    if "latitude" in df_op.columns and "longitude" in df_op.columns:
        if "latitude" in df_op.columns and df_op["latitude"].notna().any():
            # take latest value per group and plot
            latest = df_op.sort_values("date").groupby("group").last().dropna(subset=["latitude","longitude"]).reset_index()
            if latest.empty:
                st.info("지도용 좌표 데이터가 없습니다.")
            else:
                mdf = latest.rename(columns={"latitude":"lat","longitude":"lon","value":"농도"})
                st.map(mdf[["lat","lon"]])
                st.table(mdf[["group","농도"]].assign(설명="마지막 관측치"))
        else:
            st.info("좌표 데이터가 포함되어 있지 않아 지도를 표시할 수 없습니다.")
    else:
        st.info("좌표 데이터가 포함되어 있지 않아 지도를 표시할 수 없습니다.")

    # World Bank 연평균 PM2.5 (간단 조회)
    st.subheader("World Bank: 연평균 PM2.5 (예시)")
    try:
        wb_countries = st.multiselect("국가(ISO3) 선택 (World Bank)", options=["KOR","CHN","IND","FIN","ISL","USA"], default=["KOR","CHN","IND"])
        wb_years = st.multiselect("연도 범위 선택 (예시)", options=[2015,2016,2017,2018,2019,2020,2021,2022,2023], default=[2018,2020,2022])
        if wb_countries:
            df_wb = fetch_worldbank_pm25(country_codes=wb_countries, years=wb_years)
            if df_wb.empty:
                st.info("World Bank 데이터가 확보되지 않았습니다.")
            else:
                st.dataframe(df_wb.head(200))
                fig2 = px.line(df_wb, x="year", y="value", color="country", markers=True, labels={"year":"연도","value":"연평균 PM2.5 (µg/m³)","country":"국가"})
                fig2.update_layout(title="World Bank: 연평균 PM2.5", font_family="Pretendard")
                st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.info("World Bank API 호출 중 문제가 발생했습니다.")

    st.markdown("---")
    st.caption("데이터 출처 예: OpenAQ (https://api.openaq.org/), World Bank (https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3), WHO Air Quality Database (https://www.who.int/data/gho/data/themes/air-pollution/who-air-quality-database)")

# ---------------------------
# 탭 2: 사용자 입력 기반 대시보드 (프롬프트 텍스트 사용)
# ---------------------------
with tabs[1]:
    st.header("사용자 입력 기반 대시보드 — 보고서 본문(프롬프트) 기반 예시")
    st.markdown(
        """
        제공하신 보고서(문서 본문)를 바탕으로 **학교·가정의 실내 공기질(교실별 PM2.5)** 와 **지역(실외)** 을 비교하는 예시 데이터를 자동 생성하여 시각화합니다.
        - 앱 실행 중 파일 업로드나 추가 텍스트 입력을 요구하지 않습니다.
        - 사이드바 옵션은 데이터 특성에 맞춰 자동 구성됩니다.
        """
    )
    st.subheader("데이터 요약 (프롬프트 기반 생성)")
    st.write("원문에서 제안된 측정 항목: 교실별 PM2.5, 이산화탄소, 환기시간대별 변화 등. 여기서는 PM2.5(실내 vs 실외)를 시뮬레이션했습니다.")
    # Generate user dataset from prompt
    df_user_raw = generate_user_dataset_from_prompt()
    df_user = preprocess_df(df_user_raw, date_col="date", value_col="value", group_col="group")

    st.subheader("전처리된 사용자 데이터 (보고서 기반 예시)")
    st.dataframe(df_user.head(200))
    # CSV 다운로드
    buf2 = io.BytesIO()
    df_user.to_csv(buf2, index=False)
    buf2.seek(0)
    st.download_button("전처리된 사용자 데이터 CSV 다운로드", data=buf2, file_name="user_data_preprocessed.csv", mime="text/csv")

    # 자동 구성된 사이드바 옵션 (기간 필터, 스무딩, 단위)
    st.sidebar.subheader("사용자 데이터 옵션 (자동 구성)")
    # date range from data
    min_dt = df_user["date"].min()
    max_dt = df_user["date"].max()
    sel_range = st.sidebar.date_input("기간 필터 (사용자 데이터)", value=(min_dt.date(), max_dt.date()), min_value=min_dt.date(), max_value=max_dt.date())
    smoothing = st.sidebar.slider("이동평균 스무딩(간격, 관측치 단위)", min_value=1, max_value=9, value=3)
    show_points = st.sidebar.checkbox("측정점 표시", value=False)
    group_options = df_user["group"].unique().tolist()
    selected_groups = st.sidebar.multiselect("그룹 선택 (사용자 데이터)", options=group_options, default=group_options)

    # apply filters
    dr_start = datetime.combine(sel_range[0], datetime.min.time()).replace(tzinfo=SEOUL_TZ)
    dr_end = datetime.combine(sel_range[1], datetime.max.time()).replace(tzinfo=SEOUL_TZ)
    df_user_f = df_user[(df_user["date"] >= dr_start) & (df_user["date"] <= dr_end) & (df_user["group"].isin(selected_groups))].copy()

    if df_user_f.empty:
        st.info("선택된 조건에 해당하는 데이터가 없습니다.")
    else:
        # Time series with smoothing
        st.subheader("시계열(교실별 실내 vs 실외) — PM2.5")
        # apply moving average per group
        df_user_f["date_str"] = df_user_f["date"].dt.strftime("%Y-%m-%d %H:%M")
        df_plot = df_user_f.sort_values("date")
        if smoothing > 1:
            df_plot = df_plot.groupby("group").apply(lambda g: g.assign(value_s=g["value"].rolling(window=smoothing, min_periods=1).mean())).reset_index(drop=True)
            y_col = "value_s"
            ylabel = f"농도 (µg/m³) — {smoothing}시점 이동평균"
        else:
            y_col = "value"
            ylabel = "농도 (µg/m³)"
        fig3 = px.line(df_plot, x="date", y=y_col, color="group", labels={"date":"일시","value":"농도 (µg/m³)","value_s":"농도 (스무딩)","group":"그룹"})
        if show_points:
            fig3.update_traces(mode="lines+markers")
        fig3.update_layout(title="교실별(실내) vs 지역(실외) PM2.5 추이", font_family="Pretendard", yaxis_title=ylabel)
        st.plotly_chart(fig3, use_container_width=True)

        # 비교막대: 같은 시간대(예: 점심시간 12:00) 평균 비교
        st.subheader("비교: 특정 시간대 평균 (예: 12:00 ~ 13:00)")
        # aggregate hourly by group for selected date range
        df_user_f["hour"] = df_user_f["date"].dt.hour
        lunch_df = df_user_f[(df_user_f["hour"] >= 12) & (df_user_f["hour"] < 13)]
        if lunch_df.empty:
            st.info("선택된 기간에 12시대 데이터가 없습니다.")
        else:
            agg = lunch_df.groupby("group")["value"].mean().reset_index().rename(columns={"value":"평균 PM2.5 (µg/m³)"})
            fig4 = px.bar(agg, x="group", y="평균 PM2.5 (µg/m³)", labels={"group":"그룹"})
            fig4.update_layout(title="점심시간대(12시~13시) 평균 PM2.5 비교", font_family="Pretendard")
            st.plotly_chart(fig4, use_container_width=True)
            st.table(agg)

    st.markdown("---")
    st.subheader("실행 도움말 / 권장 활용법")
    st.markdown(

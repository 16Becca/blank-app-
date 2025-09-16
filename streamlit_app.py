# -*- coding: utf-8 -*-
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# --- 안전한 Matplotlib/폰트 설정 ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
import matplotlib.patches as patches

# --- Cartopy는 환경에 따라 설치 실패 가능 → 옵셔널 로딩 ---
USE_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    USE_CARTOPY = False

# -------------------------------------------------
# 전역 스타일/폰트
# -------------------------------------------------
def setup_font():
    """Pretendard 없으면 서버 기본 한글 폰트 후보로 폴백."""
    font_path = Path(__file__).parent / "fonts" / "Pretendard-Bold.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        rcParams["font.family"] = font_name
    else:
        # 서버/도커에서 흔한 한글 폰트 후보
        for cand in ["NanumGothic", "Noto Sans CJK KR", "AppleGothic"]:
            try:
                rcParams["font.family"] = cand
                break
            except Exception:
                pass
    rcParams["axes.unicode_minus"] = False
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.25

setup_font()
PE = [pe.withStroke(linewidth=2.5, foreground="white")]

st.set_page_config(page_title="뜨거워지는 바다: SST 대시보드", layout="wide", page_icon="🌊")

# -------------------------------------------------
# NOAA OISST v2 High-Res (0.25°) 일일 데이터 (연도별 파일)
# -------------------------------------------------
BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"

# -------------------------------------------------
# 데이터 로더: 'nearest + tolerance' + 연-경계/폴백 탐색 + 캐시
# -------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_sst(date: datetime.date, lat_range=(28, 42), lon_range=(120, 135)):
    """
    - 선택 날짜가 time index에 정확히 없을 때 발생하는
      "not all values found in index 'time' ..." 문제에 대응
    - 1) nearest + tolerance(3일) → 2) 7일 범위에서 과거로 폴백 탐색
    - 연도 경계 자동 처리
    - 반환: (DataArray, 실제사용날짜date) 또는 (None, None)
    """

    def _open_year(y: int):
        url = BASE_URL.format(year=y)
        # pydap 미설치 환경이 많으므로 기본엔진 → 실패 시 pydap
        try:
            ds = xr.open_dataset(url)  # netCDF4/OPeNDAP 자동
        except Exception:
            ds = xr.open_dataset(url, engine="pydap")
        return ds.sortby("time")

    try:
        ds_main = _open_year(date.year)

        # 1) 가까운 날짜 자동 선택 (허용오차 3일)
        try:
            da = (
                ds_main["sst"]
                .sel(time=np.datetime64(date), method="nearest", tolerance=np.timedelta64(3, "D"))
                .sel(lat=slice(*lat_range), lon=slice(*lon_range))
                .squeeze()
            )
            da.load()
            if np.isfinite(da.values).any():
                used_date = pd.to_datetime(da["time"].item()).date()
                return da, used_date
        except Exception:
            pass

        # 2) 실패 시 7일 동안 과거로 하루씩 물러나며 탐색 (연도 경계 포함)
        for back in range(1, 8):
            dt = date - datetime.timedelta(days=back)
            ds = ds_main if dt.year == date.year else _open_year(dt.year)
            try:
                da = (
                    ds["sst"]
                    .sel(time=np.datetime64(dt))  # 정확 일치 시도
                    .sel(lat=slice(*lat_range), lon=slice(*lon_range))
                    .squeeze()
                )
                da.load()
                if np.isfinite(da.values).any():
                    used_date = pd.to_datetime(da["time"].item()).date()
                    return da, used_date
            except Exception:
                continue

        return None, None

    except Exception as e:
        st.error(f"데이터 불러오기 실패: {e}")
        return None, None

# -------------------------------------------------
# 플로팅 (Cartopy 있으면 지도 투영, 없으면 평면 대체)
# -------------------------------------------------
def plot_sst(da, date, extent=(120, 135, 28, 42)):
    # 계절/날짜 변화에 안전한 컬러 스케일
    arr = da.values
    if not np.isfinite(arr).any():
        raise ValueError("SST 값이 모두 NaN입니다.")

    vmin = float(np.nanpercentile(arr, 5))
    vmax = float(np.nanpercentile(arr, 95))
    # 시인성 좋은 중심값(따뜻한 계절 가중) 또는 중간값
    vcenter = min(max(29.0, vmin + (vmax - vmin) * 0.6), vmax - 0.1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    if USE_CARTOPY:
        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        im = da.plot.pcolormesh(
            ax=ax, x="lon", y="lat",
            transform=ccrs.PlateCarree(),
            cmap="YlOrRd", norm=norm, add_colorbar=False
        )
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    else:
        # Cartopy 미사용 평면 대체 (환경 호환 모드)
        fig, ax = plt.subplots(figsize=(9, 6))
        im = da.plot.pcolormesh(
            ax=ax, x="lon", y="lat",
            cmap="YlOrRd", norm=norm, add_colorbar=False
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("경도")
        ax.set_ylabel("위도")
        ax.grid(alpha=0.25)
        st.info("지도가 간소화된 평면 모드로 표시되었습니다 (Cartopy 미사용).", icon="ℹ️")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
    cbar.set_label("해수면 온도 (℃)")
    ax.set_title(f"해수면 온도: {date.strftime('%Y-%m-%d')}")
    return fig

# -------------------------------------------------
# 미니 차트 유틸 (Bullet / Lollipop / Combo / Waffle)
# -------------------------------------------------
def bullet(ax, value, target, label="", color="#F28E2B"):
    lo, hi = min(value, target), max(value, target)
    pad = (hi - lo) * 0.5 + 0.5
    vmin, vmax = lo - pad, hi + pad
    ax.barh([0], [vmax - vmin], left=vmin, color="#EEEEEE", height=0.36)
    ax.barh([0], [value - vmin], left=vmin, color=color, height=0.36)
    ax.axvline(target, color="#333333", lw=2.2)
    ax.set_yticks([]); ax.set_xlim(vmin, vmax); ax.set_xlabel("℃"); ax.set_title(label)
    delta = value - target
    badge = f"+{delta:.1f}℃" if delta >= 0 else f"{delta:.1f}℃"
    ax.text(value, 0.1, f"{value:.1f}℃", ha="left", va="bottom", weight="bold", path_effects=PE)
    ax.text(0.02, 0.9, badge, transform=ax.transAxes,
            fontsize=12, weight="bold", color="white", path_effects=PE,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#C1272D" if delta>=0 else "#2B7A78",
                      edgecolor="none"))

def lollipop_horizontal(ax, labels, values, title, unit="℃", color="#4C78A8", highlight_color="#E45756"):
    idx = np.argsort(values)[::-1]
    labels_sorted = [labels[i] for i in idx]
    values_sorted = [values[i] for i in idx]
    y = np.arange(len(labels_sorted))
    ax.hlines(y, [0]*len(values_sorted), values_sorted, color="#CCCCCC", lw=3)
    vmax_i = int(np.argmax(values_sorted))
    for i, v in enumerate(values_sorted):
        col = highlight_color if i == vmax_i else color
        ax.plot(v, y[i], "o", ms=10, mfc=col, mec=col)
        ax.text(v + max(values_sorted)*0.03, y[i],
                f"{v:.2f}{unit}" if unit.endswith("년") else f"{v:.1f}{unit}",
                va="center", weight="bold" if i == vmax_i else 500, color=col, path_effects=PE)
    ax.set_yticks(y, labels_sorted); ax.set_xlabel(unit); ax.set_title(title); ax.grid(axis="x", alpha=0.25)

def combo_bar_line(ax, x_labels, bars, line, bar_color="#FDB863", line_color="#C1272D"):
    x = np.arange(len(x_labels))
    ax.bar(x, bars, color=bar_color, width=0.55)
    ax.set_xticks(x, x_labels); ax.set_ylabel("총 환자 수(명)")
    ax2 = ax.twinx()
    ax2.plot(x, line, marker="o", ms=7, lw=2.5, color=line_color)
    ax2.set_ylabel("총 사망자 수(명)", color=line_color)

def waffle(ax, percent, rows=10, cols=10, on="#F03B20", off="#EEEEEE", title=None):
    total = rows*cols
    k = int(round(percent/100*total))
    for i in range(total):
        r = i // cols; c = i % cols
        color = on if i < k else off
        rect = patches.Rectangle((c, rows-1-r), 0.95, 0.95, facecolor=color, edgecolor="white")
        ax.add_patch(rect)
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    if title: ax.set_title(title)
    ax.text(cols/2, rows/2, f"{percent:.0f}%", ha="center", va="center",
            fontsize=20, weight="bold", color="#333", path_effects=PE)

# -------------------------------------------------
# 본문 UI
# -------------------------------------------------
st.title("🌊 뜨거워지는 지구: 해수면 온도 상승이 고등학생에게 미치는 영향")

st.header("I. 서론: 뜨거워지는 바다, 위협받는 교실")
st.markdown("""
한반도는 지구 평균보다 2~3배 빠른 해수면 온도 상승을 겪고 있으며, 이는 더 이상 추상적인 환경 문제가 아니라
미래 세대의 학습권과 건강을 직접적으로 위협하는 현실입니다. 본 보고서는 고등학생을 기후 위기의 가장 취약한 집단이자
변화의 핵심 동력으로 조명하며, 해수면 온도(SST) 상승의 실태와 파급효과를 다각도로 분석합니다.
""")

st.header("II. 조사 계획")
st.subheader("1) 조사 기간")
st.markdown("2025년 7월 ~ 2025년 8월")
st.subheader("2) 조사 방법과 대상")
st.markdown("""
- **데이터 분석**: NOAA OISST v2 High Resolution Dataset  
- **문헌 조사**: 기상청, 연구 논문, 보도자료 등  
- **대상**: 대한민국 고등학생의 건강·학업·사회경제적 영향
""")

st.header("III. 조사 결과")
st.subheader("1) 한반도 주변 해수면 온도 상황")

# 날짜 기본값: 매우 최신은 공란일 수 있으므로 D-2
today = datetime.date.today()
default_date = min(today - datetime.timedelta(days=2), today)  # 미래 선택 방지
date = st.date_input("날짜 선택", value=default_date, max_value=today)

with st.spinner("데이터 불러오는 중..."):
    da, used_date = load_sst(date)

if da is not None:
    st.pyplot(plot_sst(da, used_date), clear_figure=True)
    if used_date != date:
        st.caption(f"선택 날짜에 데이터가 없어 **{used_date.strftime('%Y-%m-%d')}** 자료로 대체했습니다.")
else:
    st.warning("해당 기간에 유효한 데이터를 찾지 못했습니다. 날짜를 바꿔보세요.")

# ----------------------- 인포 차트들 -----------------------
st.subheader("📈 최근 기록과 평년 대비 편차 (예시)")
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 23.2, 21.2, label="2024-10 vs 최근10년")
    st.pyplot(fig, clear_figure=True)
with c2:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 19.8, 19.2, label="2023 연평균 vs 2001–2020", color="#2E86AB")
    st.pyplot(fig, clear_figure=True)
with c3:
    fig, ax = plt.subplots(figsize=(5,2.6))
    bullet(ax, 22.6, 22.6-2.8, label="서해 2024-10 vs 최근10년", color="#E67E22")
    st.pyplot(fig, clear_figure=True)

st.subheader("📊 해역별 장·단기 상승과 편차 (예시)")
regions = ["동해", "서해", "남해"]
rise_1968_2008 = [1.39, 1.23, 1.27]
rate_since_2010 = [0.36, 0.54, 0.38]
anom_2024 = [3.4, 2.8, 1.1]
cL1, cL2, cL3 = st.columns(3)
with cL1:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, rise_1968_2008, title="장기 상승폭 (1968–2008)", unit="℃")
    st.pyplot(fig, clear_figure=True)
with cL2:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, rate_since_2010, title="연평균 상승률 (2010~)", unit="℃/년", color="#59A14F")
    st.pyplot(fig, clear_figure=True)
with cL3:
    fig, ax = plt.subplots(figsize=(4.8,3))
    lollipop_horizontal(ax, regions, anom_2024, title="2024 편차", unit="℃", color="#F28E2B")
    st.pyplot(fig, clear_figure=True)

st.subheader("2) 지구에 미치는 영향: 극단적 기상 현상의 심화")
st.markdown("""
해수면 온도 상승은 대기와 상호작용하며 지구 전체의 기상 시스템을 교란합니다.
- **더 강력한 태풍**: 따뜻한 바다는 태풍에 더 많은 에너지를 공급합니다.
- **집중호우 빈발**: 기온이 1℃ 오르면 대기가 머금을 수 있는 수증기량은 약 7% 증가합니다.
- **혹독한 폭염**: 열돔(Heat Dome) 현상으로 폭염이 장기화됩니다.
""")

temps2 = np.arange(0, 6)  # 0~5℃
humidity_increase = 7 * temps2
figH2, axH2 = plt.subplots(figsize=(7,4))
axH2.plot(temps2, humidity_increase, lw=3, marker="o")
axH2.fill_between(temps2, humidity_increase, alpha=0.2)
axH2.set_xlabel("기온 상승 (℃)")
axH2.set_ylabel("대기 수증기량 증가율 (%)")
axH2.set_title("기온 상승에 따른 대기 수증기량 증가")
for t, v in {1:7,2:14,3:21,4:28,5:35}.items():
    axH2.scatter(t, v, zorder=5)
    axH2.annotate(f"+{v:.0f}%", (t, v), textcoords="offset points", xytext=(0,10), ha="center", weight="bold")
st.pyplot(figH2, clear_figure=True)

st.subheader("3) 고등학생에게 미치는 영향 (예시)")
st.markdown("**기온 상승 → 학업 성취도 감소** (NBER 연구 요지 인용)")

temps = np.arange(0, 6)
impact = 100 - (1.8 * temps)  # 1℃ 당 -1.8%
figC, axC = plt.subplots(figsize=(7,4))
axC.bar(temps, impact, alpha=0.7, label="구간별 학업 성취도")
axC.plot(temps, impact, marker="o", lw=2.5, label="추세선 (1℃ 당 -1.8%)")
axC.set_xlabel("기온 상승 (℃)")
axC.set_ylabel("학업 성취도 (%)")
axC.set_title("기온 상승이 학업 성취도에 미치는 영향")
axC.set_ylim(80, 102)
for t, v in zip(temps, impact):
    axC.text(t, v+0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
axC.legend()
st.pyplot(figC, clear_figure=True)

st.markdown("**신체·정신 건강** (예시 수치)")
years = ["2022년", "2023년", "2024년"]
patients = [1564, 2818, 3704]
deaths = [9, 32, 34]
figM, axM = plt.subplots(figsize=(8, 3.6))
combo_bar_line(axM, years, patients, deaths)
axM.set_title("온열질환 환자·사망 추이")
st.pyplot(figM, clear_figure=True)

cwa, cwb = st.columns(2)
with cwa:
    figW1, axW1 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW1, 59, title="기후변화를 매우/극도로 우려")
    st.pyplot(figW1, clear_figure=True)
with cwb:
    figW2, axW2 = plt.subplots(figsize=(4.2, 4.2))
    waffle(axW2, 45, title="일상에 부정적 영향을 받음")
    st.pyplot(figW2, clear_figure=True)


st.subheader("4) 대응과 미래 세대를 위한 제언")
st.markdown("""
- **정책**: 모든 학교에 냉방 및 환기 시스템을 현대화하고, 기후 변화에 따른 청소년 건강 영향을 추적하는 세분화된 통계를 구축해야 합니다.
- **교육**: 기후변화를 정규 교과목으로 편성하고, 문제 해결 중심의 프로젝트 기반 학습을 확대해야 합니다. 또한, '기후테크'와 같은 새로운 진로 분야에 대한 지도가 필요합니다.
- **청소년 행동**: 플라스틱 저감 캠페인, 기후행동 소송 참여, 지역사회 환경 문제 해결 등 청소년이 주도하는 기후 행동을 적극적으로 지원하고 확산해야 합니다.
""")

# --- 결론 및 참고 자료 ---
st.header("IV. 결론")
st.markdown("""
대한민국 주변 해수면 온도의 상승은 단순한 해양 문제가 아니라,  
고등학생들의 건강·학업·생활 전반을 위협하는 **복합 위기**입니다.  
그러나 교육과 청소년 주도의 기후 행동을 통해 이 위기를 기회로 전환할 수 있습니다.  
""")

st.header("V. 참고 자료")
st.markdown("""
- Goodman, J., & Park, R. J. (2018). *Heat and Learning*. NBER Working Paper.
- Hickman, C., et al. (2021). Climate anxiety in children and young people and their beliefs about government responses to climate change: a global survey. *The Lancet Planetary Health*.
- 기상청 보도자료 (2024)  
- 한국해양수산개발원 연구보고서  
- Planet03 해양열파 연구 (2021)  
- Newstree, YTN Science 외 기사 및 연구논문  
""")

st.markdown(
    """
    <hr style='border:1px solid #ccc; margin-top:30px; margin-bottom:10px;'/>
    <div style='text-align: center; padding: 10px; color: gray; font-size: 0.9em;'>
        미림마이스터고등학교 1학년 4반 1조 · 지속가능한지구사랑해조
    </div>
    """,
    unsafe_allow_html=True
)
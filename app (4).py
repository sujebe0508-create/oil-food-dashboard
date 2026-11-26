"""OIL -> FOOD PRICE
기름값이 오르면 식품 물가도 오를까?

데이터
- 오피넷: 전국 보통휘발유 월평균 판매가격 (data/gasoline_monthly.csv)
- KOSIS: 품목별 소비자물가지수, 전국, 2020=100 (data/cpi_items.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
MAX_LAG = 6
MIN_MONTHS_FOR_CORR = 24
POLICY_DATE = pd.Timestamp("2026-03-01")  # 석유 최고가격제 시행(2026.3.13) 달

# 화면에 보이는 이름 -> KOSIS 파일 안의 품목 이름
FOODS = {
    "총지수": "총지수",
    "농축수산물": "농축수산물",
    "곡물": "(농산물)곡물",
    "채소": "(농산물)채소",
    "과실": "(농산물)과실",
    "축산물": "축산물",
    "수산물": "수산물",
    "가공식품": "가공식품",
    "외식": "외식",
}

PRESETS = {
    "전체 (2015.10~2026.10)": ("2015-10", "2026-10"),
    "코로나 이전 (~2020.02)": ("2015-10", "2020-02"),
    "코로나 기간 (2020.03~2021.12)": ("2020-03", "2021-12"),
    "2022년 이후 (2022.01~)": ("2022-01", "2026-10"),
    "직접 선택": None,
}

st.set_page_config(page_title="OIL → FOOD PRICE", layout="wide")


# ---------------------------------------------------------------- 데이터
@st.cache_data
def load_data():
    # 휘발유: 월별 전국 보통휘발유 평균 (원/L)
    g = pd.read_csv(DATA / "gasoline_monthly.csv", encoding="cp949")
    g.index = pd.to_datetime(g["구분"], format="%Y년%m월")
    gas = g["보통휘발유"].astype(float).rename("휘발유")

    # 소비자물가지수: 품목 행 -> 월 열을 뒤집어서 월별 표로
    c = pd.read_csv(DATA / "cpi_items.csv", encoding="cp949")
    c = c.drop(columns=[x for x in c.columns if x.startswith("Unnamed")])
    c = c.drop_duplicates("품목별").set_index("품목별")
    month_cols = [x for x in c.columns if x.endswith("월")]
    c = c[list(month_cols)].loc[list(FOODS.values())].T
    c.index = pd.to_datetime(
        [m.replace(" 월", "").replace(".", "-") + "-01" for m in c.index]
    )
    c = c.astype(float)
    c.columns = list(FOODS.keys())

    df = pd.concat([gas, c], axis=1).dropna()
    # 전년동월대비 변화율은 전체 기간으로 먼저 계산하고 나중에 자른다
    yoy = df.pct_change(12) * 100
    return df, yoy


df, yoy = load_data()
months = [d.strftime("%Y-%m") for d in df.index]


def lag_table(food: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """휘발유 변화율이 L개월 앞선 경우의 상관계수 (0~MAX_LAG)."""
    rows = []
    for lag in range(MAX_LAG + 1):
        gas_lead = yoy["휘발유"].shift(lag)  # 전체 기간에서 먼저 밀어 둔다
        pair = pd.concat([gas_lead, yoy[food]], axis=1).loc[start:end].dropna()
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 3 else np.nan
        rows.append({"시차": lag, "상관계수": corr, "개월수": len(pair)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 입력
st.title("OIL → FOOD PRICE")
st.subheader("기름값이 오르면 식품 물가도 오를까?")

c1, c2 = st.columns(2)
with c1:
    preset = st.selectbox("분석 기간", list(PRESETS.keys()))
    if PRESETS[preset] is None:
        s_label, e_label = st.select_slider(
            "시작 달과 끝 달", options=months, value=(months[0], months[-1])
        )
    else:
        s_label, e_label = PRESETS[preset]
with c2:
    food = st.selectbox("식품 유형", list(FOODS.keys()), index=list(FOODS).index("축산물"))

start, end = pd.Timestamp(s_label + "-01"), pd.Timestamp(e_label + "-01")
if (end.year - start.year) * 12 + (end.month - start.month) < 1:
    st.warning("기간을 두 달 이상으로 골라 주세요.")
    st.stop()

win = df.loc[start:end]
period_text = f"{s_label} ~ {e_label}"

# ---------------------------------------------------------------- KPI
lags = lag_table(food, start, end)
valid = lags.dropna(subset=["상관계수"])
food_change = (win[food].iloc[-1] / win[food].iloc[0] - 1) * 100
gas_change = (win["휘발유"].iloc[-1] / win["휘발유"].iloc[0] - 1) * 100

k1, k2, k3 = st.columns(3)
k1.metric("휘발유 평균 가격", f"{win['휘발유'].mean():,.0f}원/L", f"기간 변화 {gas_change:+.1f}%")
k2.metric(f"{food} 물가지수 변화", f"{food_change:+.1f}%", f"{win[food].iloc[0]:.1f} → {win[food].iloc[-1]:.1f}")
if len(valid):
    best = valid.loc[valid["상관계수"].abs().idxmax()]
    k3.metric("상관이 가장 큰 시차", f"{int(best['시차'])}개월", f"상관계수 {best['상관계수']:+.2f}")
else:
    k3.metric("상관이 가장 큰 시차", "계산 불가", "기간이 너무 짧음")

# ---------------------------------------------------------------- 차트 1
st.markdown("### ① 휘발유 가격과 식품 물가는 같이 움직였을까")
base = win[["휘발유", food]] / win[["휘발유", food]].iloc[0] * 100
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=base.index, y=base["휘발유"], name="휘발유 가격", line=dict(color="#e8833a", width=2.5)))
fig1.add_trace(go.Scatter(x=base.index, y=base[food], name=f"{food} 물가지수", line=dict(color="#2f6fb3", width=2.5)))
if start <= POLICY_DATE <= end:
    fig1.add_shape(type="line", x0=POLICY_DATE, x1=POLICY_DATE, y0=0, y1=1, yref="paper",
                   line=dict(color="gray", dash="dot"))
    fig1.add_annotation(x=POLICY_DATE, y=1, yref="paper", text="석유 최고가격제 시행(2026.3)",
                        showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color="gray"))
fig1.update_layout(
    title=f"{period_text}, 첫 달을 100으로 맞춤",
    yaxis_title="첫 달 = 100", xaxis_title=None,
    legend=dict(orientation="h", y=-0.2), margin=dict(t=60, b=20),
)
st.plotly_chart(fig1, width="stretch")
st.caption("원/L와 물가지수는 단위가 달라서 둘 다 선택 기간의 첫 달을 100으로 맞춰 그렸습니다. 기간을 바꾸면 기준점도 바뀝니다.")

# ---------------------------------------------------------------- 차트 2
st.markdown("### ② 어떤 품목이 가장 많이 올랐을까")
chg = ((win.iloc[-1] / win.iloc[0] - 1) * 100).sort_values(ascending=False)
colors = ["#e8833a" if n == "휘발유" else "#2f6fb3" if n == food else "#b8bfc7" for n in chg.index]
fig2 = go.Figure(go.Bar(x=chg.index, y=chg.values, marker_color=colors,
                        text=[f"{v:+.1f}%" for v in chg.values], textposition="outside"))
fig2.update_layout(title=f"{period_text}, 첫 달 대비 마지막 달 변화율",
                   yaxis_title="변화율(%)", xaxis_title=None, margin=dict(t=60, b=20))
st.plotly_chart(fig2, width="stretch")
st.caption("첫 달과 마지막 달만 비교한 값입니다. 채소와 과실은 계절 영향이 커서 어느 달을 고르느냐에 따라 달라질 수 있습니다.")

# ---------------------------------------------------------------- 차트 3
st.markdown("### ③ 휘발유가 먼저 움직이면 식품 물가가 따라왔을까")
fig3 = go.Figure(go.Bar(
    x=[f"{int(l)}개월" for l in lags["시차"]], y=lags["상관계수"],
    marker_color=["#2f6fb3" if (pd.notna(v) and v >= 0) else "#c0504d" for v in lags["상관계수"]],
    text=[("" if pd.isna(v) else f"{v:+.2f}") for v in lags["상관계수"]], textposition="outside",
))
fig3.update_layout(
    title=f"{food}, {period_text}, 전년동월대비 변화율의 상관계수",
    yaxis=dict(title="상관계수", range=[-1, 1]),
    xaxis_title="휘발유 변화율이 식품 물가보다 앞선 개월 수", margin=dict(t=60, b=20),
)
st.plotly_chart(fig3, width="stretch")

n_used = int(lags["개월수"].min()) if len(lags) else 0
if n_used < MIN_MONTHS_FOR_CORR:
    st.warning(f"계산에 쓰인 달이 {n_used}개라서 상관계수가 크게 흔들릴 수 있습니다. 해석에 주의하세요. "
               f"(전년동월대비는 2016.10부터 계산되어, 그 이전 달은 빠집니다.)")
st.caption("시차 0개월은 같은 달, 2개월은 휘발유 변화율이 식품 물가보다 두 달 앞선 경우입니다. "
           "상관이 높아도 휘발유가 식품 물가를 올렸다는 뜻은 아닙니다.")

# ---------------------------------------------------------------- 출처와 한계
with st.expander("데이터 출처, 집계 기준, 한계"):
    st.markdown(
        """
**출처**
- 휘발유: 한국석유공사 오피넷, 전국 주유소 보통휘발유 월간 평균 판매가격
- 물가: KOSIS 소비자물가조사, 품목별 소비자물가지수(2020=100), 전국

**집계 기준**
- 기간은 2015.10~2026.10, 월 단위입니다. 두 자료의 달을 맞춰 합쳤습니다.
- 상관계수는 전년동월대비 변화율로 계산했습니다. 원자료끼리 보면 둘 다 시간이 지나며 오르는 영향으로 상관이 부풀려지기 때문입니다.
- 전년동월대비는 2016.10부터 계산되므로 상관 분석 구간은 그 이후입니다.

**한계**
- 상관은 원인이 아닙니다. 환율, 작황, 사료와 곡물 가격, 정책 같은 다른 요인을 통제하지 않았습니다.
- 석유 최고가격제는 정유사 공급가격에 상한을 둔 제도이고 주유소 판매가격에 직접 적용되지 않았습니다. 시행 이후 자료는 8개월 정도라 따로 나눠 분석하지 않았습니다.
- 기간을 짧게 고르면 상관계수가 불안정합니다.
"""
    )

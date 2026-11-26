"""OIL → FOOD PRICE
국제유가의 변화는 식탁 물가에 어떻게 전달될까?

데이터 (모두 월 단위, 2015.10~2026.10)
- 오피넷: 국제 원유가격 일간 자료를 월평균으로 바꾼 두바이유 (oil_monthly.csv)
- 오피넷: 전국 보통휘발유 월평균 판매가격 (gasoline_monthly.csv)
- KOSIS: 품목별 소비자물가지수, 전국, 2020=100 (cpi_items.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE = Path(__file__).parent
DATA = BASE / "data" if (BASE / "data").is_dir() else BASE
MAX_LAG = 6
MIN_N = 24  # 상관계수를 믿고 읽으려면 필요한 최소 월 수(기준은 화면에 밝힘)

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
FOOD_GROUPS = ["곡물", "채소", "과실", "축산물", "수산물", "가공식품", "외식"]

PRE = ("2015-10", "2020-02")
COVID = ("2020-03", "2021-12")
POST = ("2022-01", "2026-10")
PRESETS = {
    "전체 (2015.10~2026.10)": ("2015-10", "2026-10"),
    "코로나 이전 (~2020.02)": PRE,
    "코로나 기간 (2020.03~2021.12)": COVID,
    "2022년 이후 (2022.01~)": POST,
    "직접 선택": None,
}

EVENTS = [
    (pd.Timestamp("2020-03-01"), "팬데믹 선언(2020.3)"),
    (pd.Timestamp("2022-02-01"), "러시아 우크라이나 전쟁(2022.2)"),
    (pd.Timestamp("2026-03-01"), "석유 최고가격제 시행(2026.3)"),
]

C_OIL, C_GAS, C_FOOD = "#8c6d1f", "#e8833a", "#2f6fb3"

st.set_page_config(page_title="OIL → FOOD PRICE", layout="wide")


# ---------------------------------------------------------------- 데이터
@st.cache_data
def load_data():
    g = pd.read_csv(DATA / "gasoline_monthly.csv", encoding="cp949")
    gas = pd.Series(g["보통휘발유"].astype(float).values,
                    index=pd.to_datetime(g["구분"], format="%Y년%m월"), name="휘발유")

    o = pd.read_csv(DATA / "oil_monthly.csv")
    oil = pd.Series(o["Dubai"].astype(float).values,
                    index=pd.to_datetime(o["월"]), name="국제유가")

    c = pd.read_csv(DATA / "cpi_items.csv", encoding="cp949")
    c = c.drop(columns=[x for x in c.columns if x.startswith("Unnamed")])
    c = c.drop_duplicates("품목별").set_index("품목별")
    month_cols = [x for x in c.columns if x.endswith("월")]
    c = c[list(month_cols)].loc[list(FOODS.values())].T
    c.index = pd.to_datetime([m.replace(" 월", "").replace(".", "-") + "-01" for m in c.index])
    c = c.astype(float)
    c.columns = list(FOODS.keys())

    df = pd.concat([oil, gas, c], axis=1).dropna()
    yoy = df.pct_change(12) * 100  # 전체 기간으로 먼저 계산하고 나중에 자른다
    return df, yoy


df, yoy = load_data()
months = [d.strftime("%Y-%m") for d in df.index]


def pair_corr(x, y, start, end):
    pair = pd.concat([x, y], axis=1).loc[start:end].dropna()
    n = len(pair)
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if n >= 3 else np.nan
    return r, n


@st.cache_data
def oil_gas_lags(start, end):
    rows = []
    for lag in range(MAX_LAG + 1):
        r, n = pair_corr(yoy["국제유가"].shift(lag), yoy["휘발유"], start, end)
        rows.append({"시차": lag, "상관계수": r, "개월수": n})
    return pd.DataFrame(rows)


@st.cache_data
def gas_food_matrix(start, end):
    mat = pd.DataFrame(index=list(FOODS.keys()), columns=range(MAX_LAG + 1), dtype=float)
    ns = {}
    for lag in range(MAX_LAG + 1):
        for f in FOODS:
            r, n = pair_corr(yoy["휘발유"].shift(lag), yoy[f], start, end)
            mat.loc[f, lag] = r
            ns[lag] = n
    return mat, pd.Series(ns)


def best_of(values: pd.Series):
    v = values.dropna()
    if v.empty:
        return None, np.nan
    lag = v.abs().idxmax()
    return int(lag), float(v[lag])


def josa(word, with_final, without_final):
    """받침이 있으면 with_final, 없으면 without_final을 붙인 말을 돌려준다."""
    ch = word[-1]
    if "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0:
        return word + with_final
    return word + without_final


def strength(r):
    a = abs(r)
    return "강한" if a >= 0.6 else "보통 수준의" if a >= 0.3 else "약한"


def window(start, end):
    return df.loc[start:end]


def pct_change(series):
    return (series.iloc[-1] / series.iloc[0] - 1) * 100


# ---------------------------------------------------------------- 입력
st.title("OIL → FOOD PRICE")
st.subheader("국제유가의 변화는 우리의 식탁 물가에 어떻게 전달될까?")

c1, c2 = st.columns(2)
with c1:
    preset = st.selectbox("분석 기간", list(PRESETS.keys()))
    if PRESETS[preset] is None:
        s_label, e_label = st.select_slider("시작 달과 끝 달", options=months, value=(months[0], months[-1]))
    else:
        s_label, e_label = PRESETS[preset]
with c2:
    food = st.selectbox("식품 유형", list(FOODS.keys()), index=list(FOODS).index("축산물"))

start, end = pd.Timestamp(s_label + "-01"), pd.Timestamp(e_label + "-01")
if (end.year - start.year) * 12 + (end.month - start.month) < 1:
    st.warning("기간을 두 달 이상으로 골라 주세요.")
    st.stop()

win = window(start, end)
period_text = f"{s_label} ~ {e_label}"

og = oil_gas_lags(start, end)
mat, ns = gas_food_matrix(start, end)
og_lag, og_r = best_of(og.set_index("시차")["상관계수"])
gf_lag, gf_r = best_of(mat.loc[food])

oil_chg, gas_chg = pct_change(win["국제유가"]), pct_change(win["휘발유"])
food_chg, cpi_chg = pct_change(win[food]), pct_change(win["총지수"])

# ---------------------------------------------------------------- 한 줄 요약
if og_lag is not None and gf_lag is not None:
    st.info(
        f"**{period_text}**  두바이유 {oil_chg:+.0f}%, 국내 휘발유 {gas_chg:+.0f}%, {food} 물가 {food_chg:+.1f}%. "
        f"국제유가는 휘발유와 {og_lag}개월 시차에서 가장 비슷하게 움직였고(상관계수 {og_r:+.2f}), "
        f"휘발유는 {josa(food, '과', '와')} {gf_lag}개월 시차에서 상관계수가 {gf_r:+.2f}이었습니다."
    )

# ---------------------------------------------------------------- KPI
st.markdown("##### 전체 기간(2015.10~2026.10)의 가격 범위")
a1, a2, a3, a4 = st.columns(4)
gmax, gmin = df["휘발유"].idxmax(), df["휘발유"].idxmin()
omax, omin = df["국제유가"].idxmax(), df["국제유가"].idxmin()
a1.metric("휘발유 최고", f"{df['휘발유'].max():,.0f}원/L", gmax.strftime("%Y.%m"), delta_color="off")
a2.metric("휘발유 최저", f"{df['휘발유'].min():,.0f}원/L", gmin.strftime("%Y.%m"), delta_color="off")
a3.metric("두바이유 최고", f"{df['국제유가'].max():,.0f}원/L", omax.strftime("%Y.%m"), delta_color="off")
a4.metric("두바이유 최저", f"{df['국제유가'].min():,.0f}원/L", omin.strftime("%Y.%m"), delta_color="off")

st.markdown(f"##### 선택 기간({period_text})의 누적 변화")
b1, b2, b3, b4 = st.columns(4)
b1.metric("두바이유", f"{oil_chg:+.1f}%")
b2.metric("국내 휘발유", f"{gas_chg:+.1f}%")
b3.metric(f"{food} 물가", f"{food_chg:+.1f}%")
b4.metric("전체 소비자물가", f"{cpi_chg:+.1f}%")

# ---------------------------------------------------------------- CHART 01
st.markdown("### CHART 01  국제유가 충격과 국내 가격의 변화")
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.55, 0.45],
                     subplot_titles=("두바이유와 국내 휘발유 (원/L)", f"{josa(food, '과', '와')} 전체 소비자물가 (2020=100)"))
fig1.add_trace(go.Scatter(x=win.index, y=win["국제유가"], name="두바이유", line=dict(color=C_OIL, width=2.5)), row=1, col=1)
fig1.add_trace(go.Scatter(x=win.index, y=win["휘발유"], name="국내 휘발유", line=dict(color=C_GAS, width=2.5)), row=1, col=1)
fig1.add_trace(go.Scatter(x=win.index, y=win[food], name=f"{food} 물가지수", line=dict(color=C_FOOD, width=2.5)), row=2, col=1)
if food != "총지수":
    fig1.add_trace(go.Scatter(x=win.index, y=win["총지수"], name="총지수", line=dict(color="gray", width=1.5, dash="dash")), row=2, col=1)

cs, ce = max(start, pd.Timestamp(COVID[0] + "-01")), min(end, pd.Timestamp(COVID[1] + "-01"))
if cs <= ce:
    fig1.add_shape(type="rect", x0=cs, x1=ce, y0=0, y1=1, xref="x", yref="paper",
                   fillcolor="rgba(120,120,120,0.12)", line_width=0, layer="below")
for i, (d, label) in enumerate(EVENTS):
    if start <= d <= end:
        fig1.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, xref="x", yref="paper",
                       line=dict(color="gray", dash="dot", width=1))
        fig1.add_annotation(x=d, y=1.0 + 0.03 * (i % 2), xref="x", yref="paper", text=label, showarrow=False,
                            xanchor="left", yanchor="bottom", font=dict(size=10, color="gray"))
fig1.update_layout(height=560, margin=dict(t=90, b=20), legend=dict(orientation="h", y=-0.1))
st.plotly_chart(fig1, width="stretch")
st.caption("회색 음영은 코로나 기간(2020.03~2021.12)입니다. 점선은 주요 사건의 달입니다. "
           "석유 최고가격제는 정유사 공급가격에 상한을 둔 제도이고 주유소 판매가격에 직접 적용되지는 않았습니다.")

# ---------------------------------------------------------------- CHART 02
st.markdown("### CHART 02  식품 물가 상승의 승자는 누구인가")
chg = pd.Series({f: pct_change(win[f]) for f in FOOD_GROUPS}).sort_values()
excess = chg - cpi_chg
colors = [C_FOOD if v >= 0 else "#b8bfc7" for v in excess]
fig2 = go.Figure(go.Bar(
    x=chg.values, y=chg.index, orientation="h", marker_color=colors,
    text=[f"{v:+.1f}%  (평균 대비 {e:+.1f}%p)" for v, e in zip(chg.values, excess.values)], textposition="outside",
))
fig2.add_vline(x=cpi_chg, line_dash="dash", line_color="gray",
               annotation_text=f"전체 평균 {cpi_chg:+.1f}%", annotation_position="top")
span = max(abs(chg).max(), abs(cpi_chg), 1)
fig2.update_layout(title=f"{period_text}, 첫 달 대비 마지막 달 변화율", xaxis_title="변화율(%)",
                   xaxis=dict(range=[min(0, chg.min()) - span * 0.1, max(0, chg.max()) + span * 0.55]),
                   height=430, margin=dict(t=70, b=20))
st.plotly_chart(fig2, width="stretch")
st.caption("파란 막대는 전체 소비자물가(총지수)보다 더 오른 품목입니다. 첫 달과 마지막 달만 비교한 값이라 "
           "채소와 과실처럼 계절 영향이 큰 품목은 고른 달에 따라 달라질 수 있습니다.")

# ---------------------------------------------------------------- CHART 03
st.markdown("### CHART 03  기름값이 오르면 식품 물가도 따라 오를까")
left, right = st.columns([1, 1.25])
with left:
    figa = go.Figure(go.Bar(
        x=[f"{int(l)}개월" for l in og["시차"]], y=og["상관계수"],
        marker_color=[C_OIL if (pd.notna(v) and v >= 0) else "#c0504d" for v in og["상관계수"]],
        text=[("" if pd.isna(v) else f"{v:+.2f}") for v in og["상관계수"]], textposition="outside"))
    figa.update_layout(title="국제유가에서 국내 휘발유로", yaxis=dict(title="상관계수", range=[-1, 1.1]),
                       xaxis_title="두바이유가 휘발유보다 앞선 개월 수", height=420, margin=dict(t=60, b=20))
    st.plotly_chart(figa, width="stretch")
with right:
    figb = go.Figure(go.Heatmap(
        z=mat.values, x=[f"{l}개월" for l in mat.columns], y=[("▶ " + i if i == food else i) for i in mat.index],
        zmin=-1, zmax=1, colorscale="RdBu", texttemplate="%{z:.2f}", showscale=True,
        colorbar=dict(title="상관")))
    figb.update_layout(title="국내 휘발유에서 식품 물가로", yaxis=dict(autorange="reversed"),
                       xaxis_title="휘발유가 식품 물가보다 앞선 개월 수", height=420, margin=dict(t=60, b=20))
    st.plotly_chart(figb, width="stretch")

n_used = int(ns.min()) if len(ns) else 0
if n_used < MIN_N:
    st.warning(f"상관계수 계산에 쓰인 달이 {n_used}개입니다. {MIN_N}개월보다 적어서 값이 크게 흔들릴 수 있으니 참고용으로만 보세요. "
               "(전년동월대비는 2016.10부터 계산되어 그 이전 달은 빠집니다.)")
st.caption("두 그래프 모두 전년동월대비 변화율끼리의 상관계수입니다. 1에 가까우면 같이 움직였고, 0이면 관계가 없고, "
           "마이너스면 반대로 움직였습니다. 같이 움직였다는 뜻일 뿐 원인이라는 뜻은 아닙니다.")

# ---------------------------------------------------------------- INSIGHT
st.markdown("### INSIGHT  기간별 핵심 인사이트")


def avg_corr(series):
    return float(series.dropna().mean()) if series.notna().any() else np.nan


def period_row(label, rng):
    s, e = pd.Timestamp(rng[0] + "-01"), pd.Timestamp(rng[1] + "-01")
    w = window(s, e)
    o, n_o = oil_gas_lags(s, e), None
    m, n_s = gas_food_matrix(s, e)
    return {
        "구간": label,
        "상관 계산 월 수": int(n_s.iloc[0]),
        "두바이유 변화(%)": round(pct_change(w["국제유가"]), 1),
        "휘발유 변화(%)": round(pct_change(w["휘발유"]), 1),
        "전체 물가 변화(%)": round(pct_change(w["총지수"]), 1),
        "국제유가→휘발유 평균 상관": round(avg_corr(o["상관계수"]), 2),
        f"휘발유→{food} 평균 상관": round(avg_corr(m.loc[food]), 2),
    }


rows = [period_row("코로나 이전", PRE), period_row("코로나 기간", COVID), period_row("2022년 이후", POST)]
cmp_df = pd.DataFrame(rows).set_index("구간")

bullets = []
# 1) 선택 기간 설명
bullets.append(f"**{period_text}**: 두바이유 {oil_chg:+.0f}%, 국내 휘발유 {gas_chg:+.0f}%, 전체 소비자물가 {cpi_chg:+.1f}%.")
if og_lag is not None:
    bullets.append(f"국제유가의 변화는 국내 휘발유와 {og_lag}개월 시차에서 가장 비슷하게 움직였습니다(상관계수 {og_r:+.2f}). "
                   f"국제유가에서 휘발유로 이어지는 연결은 {strength(og_r)} 편입니다.")
top_food = chg.idxmax()
bullets.append(f"식품 중 가장 많이 오른 것은 {top_food}({chg.max():+.1f}%, 전체 평균 대비 {excess[top_food]:+.1f}%p)이고, "
               f"가장 적게 오른 것은 {chg.idxmin()}({chg.min():+.1f}%)입니다.")
fm = mat.loc[FOOD_GROUPS]
if fm.notna().any().any():
    flat = fm.stack().dropna()
    (bf, bl), br = flat.abs().idxmax(), flat[flat.abs().idxmax()]
    bullets.append(f"휘발유와 가장 비슷하게 움직인 식품은 {bf}이고, {bl}개월 시차에서 상관계수가 {br:+.2f}이었습니다.")
    weak = [f for f in FOOD_GROUPS if fm.loc[f].abs().max() < 0.2]
    if weak:
        bullets.append(f"{josa(', '.join(weak), '은', '는')} 어느 시차에서도 상관계수의 크기가 0.2보다 작아 휘발유와 거의 같이 움직이지 않았습니다.")
if n_used < MIN_N:
    bullets.append(f"이 기간은 상관 계산에 쓰인 달이 {n_used}개라 위 상관 결과는 참고용입니다.")

st.markdown("\n".join(f"- {b}" for b in bullets))

st.markdown("**코로나 이전, 기간, 2022년 이후 비교**")
st.dataframe(cmp_df, width="stretch")

# 구간별로 가장 크게 달라진 식품 (시차 0~6개월 평균 상관의 차이)
pre_m, pre_n = gas_food_matrix(pd.Timestamp(PRE[0] + "-01"), pd.Timestamp(PRE[1] + "-01"))
post_m, post_n = gas_food_matrix(pd.Timestamp(POST[0] + "-01"), pd.Timestamp(POST[1] + "-01"))
if int(pre_n.iloc[0]) >= MIN_N and int(post_n.iloc[0]) >= MIN_N:
    diff = pd.Series({f: avg_corr(post_m.loc[f]) - avg_corr(pre_m.loc[f]) for f in FOOD_GROUPS}).dropna()
    big = diff.abs().idxmax()
    st.markdown(
        f"- 코로나 이전과 2022년 이후를 비교하면 휘발유와의 평균 상관이 가장 크게 달라진 식품은 **{big}**입니다 "
        f"({avg_corr(pre_m.loc[big]):+.2f} → {avg_corr(post_m.loc[big]):+.2f}). "
        f"구간별 월 수는 {int(pre_n.iloc[0])}개와 {int(post_n.iloc[0])}개입니다."
    )
st.caption("표의 평균 상관은 시차 0~6개월 상관계수의 평균입니다. 코로나 기간은 월 수가 적어 참고용입니다.")

# ---------------------------------------------------------------- 출처와 한계
with st.expander("데이터 출처, 집계 기준, 한계"):
    st.markdown(
        f"""
**출처**
- 두바이유: 한국석유공사 오피넷 국제 원유가격(일간)을 월평균으로 바꾼 값. 원화로 환산된 값이라 환율 영향이 섞여 있습니다.
- 휘발유: 오피넷, 전국 주유소 보통휘발유 월간 평균 판매가격
- 물가: KOSIS 소비자물가조사, 품목별 소비자물가지수(2020=100), 전국

**집계 기준**
- 기간은 2015.10~2026.10, 월 단위입니다. 세 자료의 달을 맞춰 합쳤고 2026.11은 끝나지 않은 달이라 뺐습니다.
- 상관계수는 전년동월대비 변화율로 계산했습니다. 원자료끼리 보면 둘 다 시간이 지나며 오르는 영향으로 상관이 부풀려지기 때문입니다.
- 전년동월대비는 2016.10부터 계산되므로 상관 분석 구간은 그 이후입니다.
- 상관계수는 {MIN_N}개월 이상일 때만 믿고 읽는 것으로 정했습니다. 이 기준은 이 대시보드가 정한 값입니다.

**한계**
- 상관은 원인이 아닙니다. 환율, 날씨, 사료와 곡물 가격, 정책 같은 다른 요인을 통제하지 않았습니다.
- 여러 시차와 품목을 한꺼번에 비교하므로 그중 높게 나온 값은 우연일 수 있습니다.
- 석유 최고가격제 시행 이후 자료는 8개월 정도라 따로 나눠 분석하지 않았습니다.
- 2022년 이후에는 코로나 영향과 유가 급등이 겹쳐 있어서 구간 차이를 코로나 때문이라고 말할 수 없습니다.
"""
    )

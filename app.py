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

QUESTION = "국제유가가 오르면, 어떤 식품 값이 몇 개월 뒤에 따라 오를까?"
USER_WHO = "유가가 크게 오를 때 어떤 식품 물가를 먼저 살펴야 할지 정해야 하는 물가 담당자 (정책 담당, 식품과 외식업의 가격 담당)"
USER_WHEN = "유가 급등 뉴스가 나왔을 때, 과거에 어느 식품이 휘발유와 비슷하게 움직였는지 확인해서 점검할 품목의 순서를 정한다"
USER_LIMIT = "앞으로의 가격을 맞히는 예측 도구가 아니라, 과거 기록을 보는 참고 자료입니다."
TRANSPORT_FOODS = ["축산물", "가공식품", "외식"]       # 가설 2에서 묶는 식품
FARM_FOODS = ["곡물", "채소", "과실", "수산물"]

C_OIL, C_GAS, C_FOOD = "#c9a227", "#e8833a", "#4c9be8"

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
st.markdown("# OIL → FOOD PRICE")
st.markdown(f"##### 핵심 질문: {QUESTION}")
st.markdown(
    f'<div style="border-left:5px solid #7a5af8;border-radius:8px;padding:12px 18px;line-height:1.8;'
    f'background:rgba(122,90,248,0.10);margin:6px 0 10px 0;">'
    f'<b>누가 쓰나</b>: {USER_WHO}<br><b>언제 쓰나</b>: {USER_WHEN}<br><b>한계</b>: {USER_LIMIT}</div>',
    unsafe_allow_html=True,
)

PERIOD_SHORT = {
    "전체 (2015.10~2026.10)": "전체",
    "코로나 이전 (~2020.02)": "코로나 이전",
    "코로나 기간 (2020.03~2021.12)": "코로나 기간",
    "2022년 이후 (2022.01~)": "2022년 이후",
    "직접 선택": "직접 선택",
}

with st.container(border=True):
    st.markdown("##### 조건을 골라 보세요  (고르면 아래 내용이 모두 바뀝니다)")
    st.markdown("**① 언제를 볼까요**")
    _p = st.segmented_control("분석 기간", list(PRESETS.keys()), default=list(PRESETS.keys())[0],
                              format_func=lambda k: PERIOD_SHORT[k], label_visibility="collapsed", key="period")
    preset = _p or list(PRESETS.keys())[0]
    if PRESETS[preset] is None:
        s_label, e_label = st.select_slider("시작 달과 끝 달", options=months, value=(months[0], months[-1]))
    else:
        s_label, e_label = PRESETS[preset]
        st.caption(f"선택한 기간: {s_label} ~ {e_label}")
    st.markdown("**② 어떤 식품을 볼까요**")
    _f = st.pills("식품 유형", list(FOODS.keys()), default="축산물", label_visibility="collapsed", key="food")
    food = _f or "축산물"
    st.caption("기간 이름은 시기를 나누려는 구분입니다. 2022년 이후에는 코로나와 유가 급등이 함께 들어 있어서, "
               "차이가 코로나 때문이라고 말할 수는 없습니다.")

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
n_used = int(ns.min()) if len(ns) else 0

oil_chg, gas_chg = pct_change(win["국제유가"]), pct_change(win["휘발유"])
food_chg, cpi_chg = pct_change(win[food]), pct_change(win["총지수"])
oil_swing = (win["국제유가"].max() / win["국제유가"].min() - 1) * 100
gas_swing = (win["휘발유"].max() / win["휘발유"].min() - 1) * 100

chg = pd.Series({f: pct_change(win[f]) for f in FOOD_GROUPS}).sort_values()
excess = chg - cpi_chg
top_food = chg.idxmax()

best = {f: best_of(mat.loc[f]) for f in FOOD_GROUPS}          # 식품별 (시차, 같이 움직인 정도)
strong = [f for f in FOOD_GROUPS if best[f][0] is not None and best[f][1] >= 0.4]
none_ = [f for f in FOOD_GROUPS if best[f][0] is not None and abs(best[f][1]) < 0.2]


def when(lag):
    return "같은 달에" if lag == 0 else f"{lag}개월 뒤에"


def level(r):
    if r >= 0.6:
        return "매우 비슷하게 움직임", "#1f6fd1"
    if r >= 0.4:
        return "비슷하게 움직임", "#4c9be8"
    if r >= 0.2:
        return "약하게 연결", "#9cc3ec"
    if r <= -0.2:
        return "반대로 움직임", "#d9534f"
    return "거의 관계없음", "#b8bfc7"


def word_card(color, title, big, sub):
    return (
        f'<div style="border-top:6px solid {color};border-radius:12px;padding:18px 20px;height:100%;'
        f'background:rgba(128,128,128,0.10);">'
        f'<div style="font-size:1rem;font-weight:600;opacity:0.8;">{title}</div>'
        f'<div style="font-size:1.9rem;font-weight:800;color:{color};line-height:1.4;margin:6px 0;">{big}</div>'
        f'<div style="font-size:0.95rem;line-height:1.6;">{sub}</div></div>'
    )


def answer(text):
    """그래프 위에 한 줄로 보여 주는 답."""
    st.markdown(
        f'<div style="font-size:1.15rem;font-weight:600;line-height:1.7;margin:4px 0 10px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def how_to_read(text):
    st.caption("읽는 법: " + text)


# ---------------------------------------------------------------- 가설과 확인 결과
def verdict_colors(v):
    return {"지지됨": "#1f9d55", "일부 지지": "#e0a800", "일부 식품에서만": "#e0a800",
            "지지되지 않음": "#d9534f", "판단 보류": "#8a8f98"}.get(v, "#8a8f98")


# 가설 1: 국제유가 -> 휘발유 (선택 기간 기준)
if og_lag is not None:
    v1 = "지지됨" if og_r >= 0.6 else "일부 지지" if og_r >= 0.3 else "지지되지 않음"
    sub1 = f"국제유가가 오르면 {when(og_lag)} 휘발유가 가장 비슷하게 따라 움직였습니다(같이 움직인 정도 {og_r:+.2f})."
else:
    v1, sub1 = "판단 보류", "선택한 기간이 너무 짧습니다."

# 가설 2: 운송 비중이 큰 식품 vs 농수산물 (선택 기간 기준)
a_vals = [best[f][1] for f in TRANSPORT_FOODS if best[f][0] is not None]
b_vals = [best[f][1] for f in FARM_FOODS if best[f][0] is not None]
if a_vals and b_vals:
    a_mean, b_mean = float(np.mean(a_vals)), float(np.mean(b_vals))
    gap = a_mean - b_mean
    v2 = "지지됨" if gap >= 0.3 else "일부 지지" if gap >= 0.1 else "지지되지 않음"
    sub2 = f"운송 비중이 큰 식품(축산물, 가공식품, 외식)은 평균 {a_mean:+.2f}, 농수산물(곡물, 채소, 과실, 수산물)은 평균 {b_mean:+.2f}입니다."
else:
    v2, sub2 = "판단 보류", "선택한 기간이 너무 짧습니다."

# 가설 3: 코로나 이전 vs 2022년 이후 (기간 선택과 관계없이 고정 비교)
_pre_m, _pre_n = gas_food_matrix(pd.Timestamp(PRE[0] + "-01"), pd.Timestamp(PRE[1] + "-01"))
_post_m, _post_n = gas_food_matrix(pd.Timestamp(POST[0] + "-01"), pd.Timestamp(POST[1] + "-01"))
if int(_pre_n.iloc[0]) >= MIN_N and int(_post_n.iloc[0]) >= MIN_N:
    _d = pd.Series({f: _post_m.loc[f].dropna().mean() - _pre_m.loc[f].dropna().mean() for f in FOOD_GROUPS})
    up_foods = [f for f in FOOD_GROUPS if _d[f] >= 0.2]
    down_foods = [f for f in FOOD_GROUPS if _d[f] <= -0.2]
    v3 = "지지됨" if len(up_foods) >= 5 else "일부 식품에서만" if up_foods else "지지되지 않음"
    sub3 = (f"2022년 이후 연결이 0.2 이상 강해진 식품: {', '.join(up_foods) if up_foods else '없음'}. "
            f"오히려 약해진 식품: {', '.join(down_foods) if down_foods else '없음'}. (기간 선택과 관계없이 고정 비교)")
else:
    v3, sub3 = "판단 보류", "구간별 월 수가 적습니다."


def hypo_card(no, text, verdict, sub):
    c = verdict_colors(verdict)
    return (
        f'<div style="border-top:6px solid {c};border-radius:12px;padding:16px 18px;height:100%;'
        f'background:rgba(128,128,128,0.10);">'
        f'<div style="font-size:0.9rem;font-weight:700;opacity:0.75;">가설 {no}</div>'
        f'<div style="font-size:1rem;font-weight:600;line-height:1.55;margin:4px 0 8px 0;">{text}</div>'
        f'<div style="display:inline-block;font-size:1.05rem;font-weight:800;color:white;background:{c};'
        f'border-radius:999px;padding:3px 14px;margin-bottom:8px;">{verdict}</div>'
        f'<div style="font-size:0.9rem;line-height:1.6;">{sub}</div></div>'
    )


HYPOTHESES = [
    "국제유가가 오르면 국내 휘발유도 시차를 두고 오른다.",
    "휘발유가 오르면 운송과 유통 비중이 큰 식품(축산물, 가공식품, 외식)이 농수산물보다 더 뚜렷하게 따라 오른다.",
    "유가가 급등한 2022년 이후에는 휘발유와 식품 물가의 연결이 코로나 이전보다 강해진다.",
]
st.markdown(
    '<div style="border-left:5px solid #e0a800;border-radius:8px;padding:12px 18px;line-height:1.9;'
    'background:rgba(224,168,0,0.10);margin:6px 0 10px 0;"><b>확인해 볼 가설</b><br>'
    + "<br>".join(f"<b>가설 {i + 1}</b>  {h}" for i, h in enumerate(HYPOTHESES))
    + '<br><span style="font-size:0.85rem;opacity:0.8;">이유: 식품은 트럭으로 옮겨져 운반비에 기름값이 들어갑니다. '
      '반대로 채소와 과실은 날씨와 작황의 영향이 더 큽니다. 맞는지는 맨 아래 결론에서 확인합니다.</span></div>',
    unsafe_allow_html=True,
)

if n_used < MIN_N:
    st.warning(f"선택한 기간은 {n_used}개월이라 짧습니다. 아래 \"같이 움직인 정도\"는 크게 흔들릴 수 있어 참고용으로만 보세요.")

st.divider()

# ---------------------------------------------------------------- 1단계
st.markdown("### 1. 국제유가가 오르면 국내 휘발유도 오를까? (가설 1)")
if og_lag is not None:
    answer(f"{'네, ' if og_r >= 0.3 else '별로 그렇지 않았습니다. '}{strength(og_r)} 편입니다. 국제유가는 {oil_swing:.0f}% 출렁였는데 국내 휘발유는 {gas_swing:.0f}%만 움직였습니다. "
           f"국제유가가 움직이면 {when(og_lag)} 휘발유가 가장 비슷하게 따라갔습니다.")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=win.index, y=win["국제유가"], name="국제유가(두바이유)", line=dict(color=C_OIL, width=3)))
fig1.add_trace(go.Scatter(x=win.index, y=win["휘발유"], name="국내 휘발유", line=dict(color=C_GAS, width=3)))
cs, ce = max(start, pd.Timestamp(COVID[0] + "-01")), min(end, pd.Timestamp(COVID[1] + "-01"))
if cs <= ce:
    fig1.add_shape(type="rect", x0=cs, x1=ce, y0=0, y1=1, xref="x", yref="paper",
                   fillcolor="rgba(150,150,150,0.14)", line_width=0, layer="below")
marks = ["①", "②", "③"]
shown = []
for mk, (d, label) in zip(marks, EVENTS):
    if start <= d <= end:
        fig1.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, xref="x", yref="paper",
                       line=dict(color="gray", dash="dot", width=1))
        fig1.add_annotation(x=d, y=1.0, xref="x", yref="paper", text=mk, showarrow=False,
                            yanchor="bottom", font=dict(size=16, color="gray"))
        shown.append(f"{mk} {label}")
fig1.update_layout(height=420, margin=dict(t=40, b=20, l=10, r=10), yaxis_title="리터당 원(원화로 환산)",
                   legend=dict(orientation="h", y=-0.15, x=0))
st.plotly_chart(fig1, width="stretch")
cap = "회색 음영은 코로나 기간(2020.03~2021.12)입니다."
if shown:
    cap += "  " + "   ".join(shown)
st.caption(cap)
st.caption(f"전체 기간 휘발유 최고 {df['휘발유'].max():,.0f}원/L({df['휘발유'].idxmax().strftime('%Y.%m')}), "
           f"최저 {df['휘발유'].min():,.0f}원/L({df['휘발유'].idxmin().strftime('%Y.%m')}). "
           f"두바이유 최고 {df['국제유가'].max():,.0f}원/L({df['국제유가'].idxmax().strftime('%Y.%m')}), "
           f"최저 {df['국제유가'].min():,.0f}원/L({df['국제유가'].idxmin().strftime('%Y.%m')}). "
           "휘발유에는 세금이 붙어 있어 국제유가보다 덜 출렁이는 면이 있습니다. "
           "석유 최고가격제는 정유사 공급가격에 상한을 둔 제도이고 주유소 판매가격에 직접 적용되지는 않았습니다.")

st.divider()

# ---------------------------------------------------------------- 2단계
st.markdown("### 2. 휘발유가 오르면 식품 값도 오를까? (가설 2)")
if strong:
    answer(f"식품마다 달랐습니다. {josa(', '.join(strong), '은', '는')} 휘발유와 비슷하게 움직였고, "
           + (f"{josa(', '.join(none_), '은', '는')} 거의 관계가 없었습니다." if none_ else "나머지는 약했습니다."))
else:
    answer("휘발유와 뚜렷하게 같이 움직인 식품은 없었습니다.")

rows2 = [(f, best[f][0], best[f][1]) for f in FOOD_GROUPS if best[f][0] is not None]
rows2.sort(key=lambda x: x[2])
fig2 = go.Figure(go.Bar(
    x=[r for _, _, r in rows2], y=[f for f, _, _ in rows2], orientation="h",
    marker_color=[level(r)[1] for _, _, r in rows2],
    text=[f"{level(r)[0]} ({r:+.2f}, {when(l)})" for _, l, r in rows2], textposition="outside", cliponaxis=False,
))
fig2.add_vline(x=0, line_color="gray", line_width=1)
fig2.update_layout(height=430, margin=dict(t=20, b=40, l=10, r=10),
                   xaxis=dict(range=[-1, 1.7], title="휘발유와 같이 움직인 정도 (1에 가까울수록 같이 움직임)"))
st.plotly_chart(fig2, width="stretch")
how_to_read("막대가 오른쪽으로 길수록 휘발유가 오른 뒤 식품 값도 같이 올랐다는 뜻이고, 0 근처면 관계가 없다는 뜻입니다. "
            "괄호 안 숫자는 같이 움직인 정도(전년동월대비 변화율끼리의 상관계수)와, 그 정도가 가장 컸던 시점입니다. "
            "같이 움직였다는 것이지 휘발유 때문에 올랐다는 뜻은 아닙니다.")

st.divider()

# ---------------------------------------------------------------- 3단계
st.markdown("### 3. 어떤 식품이 가장 많이 올랐을까?")
n_above = int((excess > 0).sum())
answer(f"{top_food}{josa(top_food, '이', '가')[len(top_food):]} {chg.max():+.0f}%로 가장 많이 올랐고, 가장 적게 오른 것은 {chg.idxmin()}({chg.min():+.0f}%)입니다. "
       f"7개 식품 중 {n_above}개가 전체 물가({cpi_chg:+.0f}%)보다 더 올랐습니다.")
fig3 = go.Figure(go.Bar(
    x=chg.values, y=chg.index, orientation="h",
    marker_color=[C_FOOD if v >= cpi_chg else "#b8bfc7" for v in chg.values],
    text=[f"{v:+.0f}%" for v in chg.values], textposition="outside", cliponaxis=False))
fig3.add_vline(x=cpi_chg, line_dash="dash", line_color="gray",
               annotation_text=f"전체 물가 {cpi_chg:+.0f}%", annotation_position="top")
fig3.update_layout(height=400, margin=dict(t=40, b=40, l=10, r=10),
                   xaxis=dict(title="기간 동안 오른 비율(%)", range=[min(0, chg.min()) - 5, chg.max() * 1.2 + 5]))
st.plotly_chart(fig3, width="stretch")
how_to_read("파란 막대는 전체 물가보다 더 오른 식품입니다. 첫 달과 마지막 달만 비교한 값이라, 채소와 과실처럼 계절 영향이 큰 식품은 고른 달에 따라 달라질 수 있습니다.")

st.divider()

# ---------------------------------------------------------------- 고른 식품
st.markdown(f"### 4. 내가 고른 {josa(food, '은', '는')} 휘발유와 같이 움직였을까?")
if gf_lag is not None:
    lv = level(gf_r)[0]
    answer(f"{josa(food, '은', '는')} 휘발유와 \"{lv}\" 수준입니다({gf_r:+.2f}). 이 기간 {josa(food, '은', '는')} {food_chg:+.0f}%, 휘발유는 {gas_chg:+.0f}% 변했습니다.")
base = win[["휘발유", food]] / win[["휘발유", food]].iloc[0] * 100
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=base.index, y=base["휘발유"], name="국내 휘발유", line=dict(color=C_GAS, width=3)))
fig4.add_trace(go.Scatter(x=base.index, y=base[food], name=f"{food} 물가지수", line=dict(color=C_FOOD, width=3)))
fig4.update_layout(height=380, margin=dict(t=20, b=20, l=10, r=10), yaxis_title="첫 달을 100으로",
                   legend=dict(orientation="h", y=-0.15, x=0))
st.plotly_chart(fig4, width="stretch")
how_to_read("두 값을 모두 선택 기간의 첫 달 100으로 맞췄습니다. 같은 모양으로 오르내리면 같이 움직인 것입니다.")

st.divider()

# ---------------------------------------------------------------- 기간별 비교
st.markdown("### 5. 기간별로 달라졌을까? (가설 3)")


def avg_corr(series):
    return float(series.dropna().mean()) if series.notna().any() else np.nan


def period_row(label, rng):
    s_, e_ = pd.Timestamp(rng[0] + "-01"), pd.Timestamp(rng[1] + "-01")
    w = window(s_, e_)
    o = oil_gas_lags(s_, e_)
    m, n_s = gas_food_matrix(s_, e_)
    return {
        "구간": label,
        "비교에 쓴 월 수": int(n_s.iloc[0]),
        "휘발유 변화(%)": round(pct_change(w["휘발유"]), 1),
        "전체 물가 변화(%)": round(pct_change(w["총지수"]), 1),
        "국제유가와 휘발유가 같이 움직인 정도": round(avg_corr(o["상관계수"]), 2),
        f"휘발유와 {food}이 같이 움직인 정도": round(avg_corr(m.loc[food]), 2),
    }


cmp_df = pd.DataFrame([period_row("코로나 이전", PRE), period_row("코로나 기간", COVID), period_row("2022년 이후", POST)]).set_index("구간")

pre_m, pre_n = gas_food_matrix(pd.Timestamp(PRE[0] + "-01"), pd.Timestamp(PRE[1] + "-01"))
post_m, post_n = gas_food_matrix(pd.Timestamp(POST[0] + "-01"), pd.Timestamp(POST[1] + "-01"))
if int(pre_n.iloc[0]) >= MIN_N and int(post_n.iloc[0]) >= MIN_N:
    diff = pd.Series({f: avg_corr(post_m.loc[f]) - avg_corr(pre_m.loc[f]) for f in FOOD_GROUPS}).dropna()
    big = diff.abs().idxmax()
    answer(f"코로나 이전과 2022년 이후를 비교하면 휘발유와 같이 움직인 정도가 가장 크게 달라진 식품은 {big}입니다 "
           f"(코로나 이전 {avg_corr(pre_m.loc[big]):+.2f}, 2022년 이후 {avg_corr(post_m.loc[big]):+.2f}).")
st.dataframe(cmp_df, width="stretch")
how_to_read("표의 같이 움직인 정도는 휘발유가 0~6개월 앞선 경우의 평균입니다. 코로나 기간은 월 수가 적어 참고용입니다. "
            f"월 수가 {MIN_N}개보다 적은 구간의 값은 크게 흔들릴 수 있습니다.")

# ---------------------------------------------------------------- 결론
st.divider()
st.markdown("### 결론: 세운 가설은 지지되었을까?")
_verdicts = [v1, v2, v3]
_n_ok = sum(v == "지지됨" for v in _verdicts)
_n_part = sum(v in ("일부 지지", "일부 식품에서만") for v in _verdicts)
_n_no = sum(v == "지지되지 않음" for v in _verdicts)
_parts = [f"지지됨 {_n_ok}개"]
if _n_part:
    _parts.append(f"일부만 지지 {_n_part}개")
if _n_no:
    _parts.append(f"지지되지 않음 {_n_no}개")
answer(f"가설 3개 중 {', '.join(_parts)}입니다. ({period_text} 기준, 가설 3은 기간과 관계없이 고정 비교)")
h1, h2, h3 = st.columns(3, gap="medium")
h1.markdown(hypo_card(1, HYPOTHESES[0], v1, sub1), unsafe_allow_html=True)
h2.markdown(hypo_card(2, HYPOTHESES[1], v2, sub2), unsafe_allow_html=True)
h3.markdown(hypo_card(3, HYPOTHESES[2], v3, sub3), unsafe_allow_html=True)
st.caption("판정 기준은 이 대시보드가 정한 값입니다(가설 1은 같이 움직인 정도 0.6 이상, 가설 2는 두 집단 차이 0.3 이상, "
           "가설 3은 5개 이상 식품에서 0.2 이상 증가). 같이 움직였다는 것이지 휘발유 때문에 올랐다는 증명은 아닙니다.")
if n_used < MIN_N:
    st.warning(f"선택한 기간이 {n_used}개월이라 짧아서 가설 1과 2의 판정은 참고용입니다.")

# ---------------------------------------------------------------- 자세히 보기
with st.expander("자세히 보기: 시차별 상세 값"):
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        figa = go.Figure(go.Bar(
            x=[f"{int(l)}개월" for l in og["시차"]], y=og["상관계수"],
            marker_color=[C_OIL if (pd.notna(v) and v >= 0) else "#c0504d" for v in og["상관계수"]],
            text=[("" if pd.isna(v) else f"{v:+.2f}") for v in og["상관계수"]], textposition="outside"))
        figa.update_layout(title="국제유가에서 국내 휘발유로", yaxis=dict(range=[-1, 1.15]),
                           xaxis_title="국제유가가 앞선 개월 수", height=420, margin=dict(t=60, b=30))
        st.plotly_chart(figa, width="stretch")
    with right:
        figb = go.Figure(go.Heatmap(
            z=mat.values, x=[f"{l}개월" for l in mat.columns], y=[("▶ " + i if i == food else i) for i in mat.index],
            zmin=-1, zmax=1, colorscale="RdBu", texttemplate="%{z:.2f}", showscale=True))
        figb.update_layout(title="국내 휘발유에서 식품으로", yaxis=dict(autorange="reversed"),
                           xaxis_title="휘발유가 앞선 개월 수", height=420, margin=dict(t=60, b=30))
        st.plotly_chart(figb, width="stretch")
    st.caption("숫자는 전년동월대비 변화율끼리의 상관계수입니다. 1에 가까우면 같이 움직였고, 0이면 관계가 없고, 마이너스면 반대로 움직였습니다.")

with st.expander("데이터 출처, 집계 기준, 한계"):
    st.markdown(
        f"""
**출처**
- 두바이유: 한국석유공사 오피넷 국제 원유가격(일간)을 월평균으로 바꾼 값. 원화로 환산된 값이라 환율 영향이 섞여 있습니다.
- 휘발유: 오피넷, 전국 주유소 보통휘발유 월간 평균 판매가격
- 물가: KOSIS 소비자물가조사, 품목별 소비자물가지수(2020=100), 전국

**집계 기준**
- 기간은 2015.10~2026.10, 월 단위입니다. 세 자료의 달을 맞춰 합쳤고 2026.11은 끝나지 않은 달이라 뺐습니다.
- 같이 움직인 정도는 전년동월대비 변화율끼리의 상관계수입니다. 원자료끼리 보면 둘 다 시간이 지나며 오르는 영향으로 값이 부풀려지기 때문입니다.
- 전년동월대비는 2016.10부터 계산되므로 이 분석 구간은 그 이후입니다.
- {MIN_N}개월 이상일 때만 믿고 읽는 것으로 정했고, 이 기준과 "비슷하게 움직임"(0.4 이상) 같은 구분은 이 대시보드가 정한 값입니다.

**한계**
- 같이 움직인 것이 원인은 아닙니다. 환율, 날씨, 사료와 곡물 가격, 정책 같은 다른 요인을 통제하지 않았습니다.
- 여러 시차와 식품을 한꺼번에 비교해 가장 큰 값을 골랐으므로 우연히 높게 나온 값일 수 있습니다.
- 석유 최고가격제 시행 이후 자료는 8개월 정도라 따로 나눠 분석하지 않았습니다.
- 2022년 이후에는 코로나 영향과 유가 급등이 겹쳐 있어서 구간 차이를 코로나 때문이라고 말할 수 없습니다.
"""
    )

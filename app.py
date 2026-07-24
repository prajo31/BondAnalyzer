"""Finance, Decoded — Single-Bond Analyzer prototype."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import bond_engine as be
from fred_service import get_yield_curve, matched_treasury_yield

st.set_page_config(page_title="Bond Analyzer", layout="wide")
st.title("Finance, Decoded — Single-Bond Analyzer")

# ---- Input panel ----
with st.sidebar:
    st.header("Bond inputs")
    face = st.number_input("Face / par value", value=1000.0, step=100.0)
    coupon_rate = st.number_input("Coupon rate (annual %)",
                                  value=5.0, step=0.25) / 100
    freq = st.selectbox("Coupon frequency", [1, 2, 4], index=1)
    years = st.number_input("Years to maturity", value=10.0, step=0.5)
    mode = st.radio("Solve for", ["Price from yield", "Yield from price"])
    if mode == "Price from yield":
        ytm = st.number_input("Yield to maturity (%)",
                              value=6.0, step=0.1) / 100
        price = be.price_from_yield(face, coupon_rate, freq, years, ytm)
    else:
        price = st.number_input("Market price", value=950.0, step=1.0)
        ytm = be.yield_from_price(face, coupon_rate, freq, years, price)
    fred_key = st.text_input("FRED API key (optional)", type="password")

# ---- Metrics ----
mac_dur = be.macaulay_duration(face, coupon_rate, freq, years, ytm)
mod_dur = be.modified_duration(face, coupon_rate, freq, years, ytm)
conv = be.convexity(face, coupon_rate, freq, years, ytm)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{price:,.2f}")
c2.metric("YTM", f"{ytm * 100:.3f}%")
c3.metric("Modified duration", f"{mod_dur:.2f}")
c4.metric("Convexity", f"{conv:.2f}")
c1.caption("Today’s fair value = PV of all coupons + face.")
c2.caption("The single rate that makes PV equal the price.")
c3.caption(f"Cash flows arrive on avg in {mac_dur:.1f} yrs.")
c4.caption("Curvature: cushions losses, boosts gains.")

st.info(
    f"A 100 bps rise in yields ≈ **{mod_dur:.2f}% price drop**, "
    f"partly offset by convexity of {conv:.2f}."
)

# ---- Price vs Yield (true curve + duration tangent) ----
ys = np.linspace(max(1e-4, ytm - 0.05), ytm + 0.05, 100)
ps = [be.price_from_yield(face, coupon_rate, freq, years, y) for y in ys]
tan = [price + be.estimated_price_change(price, mod_dur, 0, y - ytm)
       for y in ys]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ys * 100, y=ps, name="Price (true, convex)"))
fig.add_trace(go.Scatter(x=ys * 100, y=tan, name="Duration estimate",
                         line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=[ytm * 100], y=[price], mode="markers",
                         name="Current", marker=dict(size=10)))
fig.update_layout(title="Price vs. Yield",
                  xaxis_title="Yield (%)", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# ---- Price vs Maturity ----
ms = np.linspace(1, max(2, years * 2), 60)
pm = [be.price_from_yield(face, coupon_rate, freq, m, ytm) for m in ms]
figm = go.Figure(go.Scatter(x=ms, y=pm))
figm.update_layout(title="Price vs. Years to Maturity",
                   xaxis_title="Years", yaxis_title="Price")
st.plotly_chart(figm, use_container_width=True)

# ---- Sensitivity slider ----
dy = st.slider("Shift yields (bps)", -300, 300, 0, 10) / 10000
approx = price + be.estimated_price_change(price, mod_dur, conv, dy)
exact = be.price_from_yield(face, coupon_rate, freq, years, ytm + dy)
st.write(f"**Approx (dur+conv):** {approx:,.2f}  |  "
         f"**Exact reprice:** {exact:,.2f}")

# ---- Term structure via FRED ----
if fred_key:
    curve = get_yield_curve(fred_key)
    figc = go.Figure(go.Scatter(x=curve["years"], y=curve["yield"],
                                mode="lines+markers"))
    figc.update_layout(title="U.S. Treasury Yield Curve (FRED)",
                       xaxis_title="Maturity (yrs)",
                       yaxis_title="Yield (%)")
    st.plotly_chart(figc, use_container_width=True)
    tsy = matched_treasury_yield(curve, years)
    st.metric("Credit spread over Treasury",
              f"{(ytm * 100 - tsy) * 100:.0f} bps")

import streamlit as st
import pandas as pd
from datetime import timedelta
from transformers import pipeline

NEGOTIATION_GUARDRAILS = {
    "High": {"max_discount": 25},
    "Medium": {"max_discount": 15},
    "Low": {"max_discount": 5},
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config("Zero-Touch Renewals", layout="wide")



# =========================
# SESSION STATE
# =========================
for k, v in {
    "page": "dashboard",
    "selected_asset": None,
    "agent_df": None,
    "expanded_rows": set(),
    "quotes": {},
    "current_quote_id": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# SIDEBAR — CONTROLS
# =========================
st.sidebar.header("Controls")

use_llm = st.sidebar.checkbox(
    "Use LLM (local) for explainability",
    value=False,
    help="Enable AI-generated reasoning for renewal prioritization"
)

run_agents = st.sidebar.button(
    "Run Agents",
    type="primary",
    help="Re-run renewal prioritization agents with current settings"
)

st.sidebar.divider()

# =========================
# SIDEBAR — FILTERS
# =========================
st.sidebar.header("Filters")

customer_types_filter = st.sidebar.multiselect(
    "Customer Type",
    options=["Enterprise", "SMB"],
    default=["Enterprise", "SMB"],
    key="filter_customer_type"
)

product_filter = st.sidebar.multiselect(
    "Product",
    options=["Servers", "Storage", "Networking", "Software"],
    default=["Servers", "Storage", "Networking", "Software"],
    key="filter_product"
)

priority_filter = st.sidebar.multiselect(
    "Priority",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
    key="filter_priority"
)

max_days_filter = st.sidebar.slider(
    "Max days to expiry",
    min_value=0,
    max_value=365,
    value=180,
    key="filter_days"
)

# =========================
# FORMATTERS
# =========================
def money(x): return f"${float(x):,.0f}"
def money_m(x): return f"${x/1_000_000:.1f}M"
def pct(x): return f"{int(x)}%"

def priority_badge(p):
    return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}[p]

def status_badge(s):
    return {"Act Now": "⚡ Act Now", "Good to Act": "✅ Good to Act", "Monitor": "⏸️ Monitor"}[s]

def p2c_badge(v):
    return "🟢 High" if v >= 70 else "🟡 Medium" if v >= 40 else "🔴 Low"

# =========================
# SAMPLE DATA
# =========================
def load_assets():
    today = pd.Timestamp.today()
    data = [
        {"asset_id": "A-10001","customer": "ABC Corp","customer_type": "Enterprise","product": "Servers","contract_value": 42000,
         "contract_start": today - timedelta(days=900),"contract_end": today + timedelta(days=15),"usage_pct": 90,
         "usage_decline_pct": 2,"asset_age_years": 4.2,"last_discount_pct": 10,"licensing": "Per-core"},
        {"asset_id": "A-10002","customer": "Delta Inc","customer_type": "SMB","product": "Storage","contract_value": 18000,
         "contract_start": today - timedelta(days=700),"contract_end": today + timedelta(days=25),"usage_pct": 30,
         "usage_decline_pct": 55,"asset_age_years": 2.1,"last_discount_pct": 30,"licensing": "Capacity"},
        {"asset_id": "A-10003","customer": "Zento Pvt Ltd","customer_type": "Enterprise","product": "Networking","contract_value": 68000,
         "contract_start": today - timedelta(days=1200),"contract_end": today + timedelta(days=75),"usage_pct": 85,
         "usage_decline_pct": 10,"asset_age_years": 5.1,"last_discount_pct": 12,"licensing": "Enterprise"},
        {"asset_id": "A-10004","customer": "Nimbus Labs","customer_type": "SMB","product": "Software","contract_value": 9000,
         "contract_start": today - timedelta(days=400),"contract_end": today + timedelta(days=180),"usage_pct": 92,
         "usage_decline_pct": 0,"asset_age_years": 1.0,"last_discount_pct": 3,"licensing": "User-based"},
    ]
    df = pd.DataFrame(data)
    df["days_to_expiry"] = (df["contract_end"] - today).dt.days
    return df

# =========================
# AGENTS
# =========================
def run_agents(df):
    rows = []
    for _, r in df.iterrows():
        if r["days_to_expiry"] <= 30 or r["usage_decline_pct"] >= 40:
            priority, status = "High", "Act Now"
        elif r["contract_value"] > 25000 or r["days_to_expiry"] <= 90:
            priority, status = "Medium", "Good to Act"
        else:
            priority, status = "Low", "Monitor"

        expansion = "Upsell" if r["usage_pct"] >= 80 else "Cross-sell" if r["asset_age_years"] >= 3 else "Renewal Only"
        expected = r["contract_value"] * (1 - {"High": .15, "Medium": .07, "Low": .02}[priority])

        rows.append({
            **r,
            "opportunity_priority": priority,
            "opportunity_status": status,
            "upsell_cross_sell": expansion,
            "expected_revenue_impact": round(expected, 0),
            "probability_to_close": 75 if priority=="High" else 55 if priority=="Medium" else 30,
        })
    return pd.DataFrame(rows)

# =========================
# DASHBOARD
# =========================
def render_dashboard():
    st.title("Zero-Touch Service Renewals")
    st.caption("Enterprise-scale renewal intelligence with explainability and guardrails")

    df = run_agents(load_assets())
    # Apply global sidebar filters
    df = df[
        (df["customer_type"].isin(customer_types_filter)) &
        (df["product"].isin(product_filter)) &
        (df["opportunity_priority"].isin(priority_filter)) &
        (df["days_to_expiry"] <= max_days_filter)
        ].copy()

    if use_llm:
        st.info("🤖 LLM mode enabled: explanations are AI-assisted (simulated)")

    # ---- Enterprise Scale KPI ----
    scale = st.sidebar.selectbox("Simulated installed base", [1, 50, 100, 500, 1000], index=2)
    total_impact = df["expected_revenue_impact"].sum() * scale

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 High", (df.opportunity_priority=="High").sum())
    k2.metric("🟡 Medium", (df.opportunity_priority=="Medium").sum())
    k3.metric("🟢 Low", (df.opportunity_priority=="Low").sum())
    k4.metric("Revenue Impact", money_m(total_impact))

    st.success(f"At enterprise scale ({scale}×), system influences **{money_m(total_impact)}** in renewal revenue")

    st.divider()

    # ---- Worklist Header ----
    headers = ["", "Asset", "Customer", "Priority", "Status", "Expansion", "P2C", "Impact", "Action"]
    widths = [0.5,1.2,2,1.2,1.3,1.4,1.2,1.4,1.4]
    cols = st.columns(widths)
    for i,h in enumerate(headers): cols[i].markdown(f"**{h}**")

    # ---- Worklist Rows ----
    for _, r in df.iterrows():
        asset_id = r["asset_id"]
        expanded = asset_id in st.session_state.expanded_rows
        row = st.columns(widths)

        arrow = "▼" if expanded else "▶"
        if row[0].button(arrow, key=f"exp_{asset_id}"):
            st.session_state.expanded_rows.symmetric_difference_update({asset_id})
            st.rerun()

        row[1].markdown(f"**{asset_id}**")
        row[2].write(r["customer"])
        row[3].markdown(priority_badge(r["opportunity_priority"]))
        row[4].markdown(status_badge(r["opportunity_status"]))
        row[5].write(r["upsell_cross_sell"])
        row[6].markdown(f"{r['probability_to_close']}% — {p2c_badge(r['probability_to_close'])}")
        row[7].write(money(r["expected_revenue_impact"]))


        if row[8].button("Generate Quote" if r["opportunity_priority"]!="Low" else "Review", key=f"q_{asset_id}"):
            st.session_state.selected_asset = r
            st.session_state.page = "quote"
            st.rerun()

        # ---- Expanded Details ----
        if expanded:
            if use_llm:
                st.markdown(
                    "**AI Explanation:** Usage is healthy, renewal risk is low, but timing suggests proactive engagement."
                )

            st.markdown(
                f"""
                <div style="margin-left:40px;padding:12px;border-left:3px solid #e5e7eb;background:#fafafa">
                <b>Contract:</b> {money(r['contract_value'])} | {r['days_to_expiry']} days left<br>
                <b>Usage:</b> {pct(r['usage_pct'])} | Decline: {pct(r['usage_decline_pct'])}<br>
                <b>Discount:</b> {pct(r['last_discount_pct'])} | <b>Licensing:</b> {r['licensing']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    # Apply filters
    # Apply sidebar filters (widgets already created)



# =========================
# QUOTE PAGE (MINIMAL)
# =========================
def render_quote():
    r = st.session_state.selected_asset
    st.title("📄 Renewal Quote")
    st.button("← Back", on_click=lambda: st.session_state.update(page="dashboard"))

    st.metric("Customer", r["customer"])
    st.metric("Asset", r["asset_id"])
    st.metric("Expected Impact", money(r["expected_revenue_impact"]))

# =========================
# ROUTER
# =========================
if st.session_state.page == "dashboard":
    render_dashboard()
elif st.session_state.page == "quote":
    render_quote()

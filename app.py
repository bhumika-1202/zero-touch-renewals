import streamlit as st
import pandas as pd
from datetime import timedelta
from transformers import pipeline
import time


if "app_initialized" not in st.session_state:
    st.session_state.clear()
    st.session_state.app_initialized = True



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
    "accept_count": 0,
    "reject_count": 0,
    "approval_count": 0,
    "show_email_block": False,
    "quote_entry_mode": "initial",
}.items():

    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# SIDEBAR — CONTROLS (CREATE ONCE)
# =========================
st.sidebar.header("Controls")

st.sidebar.divider()
if st.sidebar.button("🔄 Reset Demo State"):
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

use_llm = st.sidebar.checkbox(
    "Use LLM (local) for explainability",
    value=False,
    help="Enable AI-generated reasoning for renewal prioritization"
)

run_agents_clicked = st.sidebar.button(
    "Run Agents",
    type="primary",
    help="Re-run renewal prioritization agents with current settings"
)

st.sidebar.divider()

# =========================
# SIDEBAR — FILTERS (CREATE ONCE)
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

country_filter = st.sidebar.multiselect(
    "Country",
    options=["US", "India", "Germany"],
    default=["US", "India", "Germany"],
    key="filter_country"
)

region_filter = st.sidebar.multiselect(
    "Region",
    options=["North America", "APAC", "EMEA"],
    default=["North America", "APAC", "EMEA"],
    key="filter_region"
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

st.sidebar.divider()
st.sidebar.header("Enterprise Scale Simulation")
portfolio_multiplier = st.sidebar.selectbox(
    "Simulated installed base size",
    options=[1, 50, 100, 500, 1000],
    index=2,
    help="Scales revenue impact to simulate enterprise portfolio size",
    key="portfolio_multiplier"
)

# =========================
# FORMATTERS / BADGES
# =========================
def money(x): return f"${float(x):,.0f}"
def money_m(x): return f"${float(x)/1_000_000:.1f}M"
def pct(x): return f"{int(x)}%"

def priority_badge(p):
    return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(p, p)

def status_badge(s):
    return {"Act Now": "⚡ Act Now", "Good to Act": "✅ Good to Act", "Monitor": "⏸️ Monitor", "On Hold": "⏸️ On Hold"}.get(s, s)

def p2c_badge(score: int) -> str:
    if score >= 70:
        return "🟢 High"
    if score >= 40:
        return "🟡 Medium"
    return "🔴 Low"

# =========================
# LLM (LAZY LOAD)
# =========================
@st.cache_resource
def load_llm():
    # Safe local model. (If model not available, app will gracefully fallback.)
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=120,
    )

def llm_explain(row):
    try:
        llm = load_llm()
        prompt = (
            "Explain in 1-2 short bullet points why this renewal opportunity priority was assigned.\n"
            f"Days to expiry: {row['days_to_expiry']}\n"
            f"Usage %: {row['usage_pct']}\n"
            f"Usage decline %: {row['usage_decline_pct']}\n"
            f"Contract value: {row['contract_value']}\n"
            f"Asset age: {row['asset_age_years']}\n"
            "Return max 2 bullets."
        )
        return llm(prompt)[0]["generated_text"]
    except Exception:
        return "Rule-based decision (LLM unavailable)"

def llm_negotiate(reason_text: str):
    try:
        llm = load_llm()
        prompt = (
            "Classify the customer's intent based on rejection reason.\n"
            "Return ONE of: price, hardware_change, timing, unclear.\n\n"
            f"Reason: {reason_text}"
        )
        result = llm(prompt)[0]["generated_text"].lower()
        if "price" in result:
            return "price"
        if "hardware" in result or "replace" in result or "refresh" in result:
            return "hardware_change"
        if "later" in result or "budget" in result or "next" in result:
            return "timing"
        return "unclear"
    except Exception:
        return "unclear"

# =========================
# SAMPLE DATA
# =========================
def load_assets():
    today = pd.Timestamp.today()
    data = [
        {
            "asset_id": "A-10001",
            "customer": "ABC Bank",
            "customer_type": "Enterprise",
            "product": "Servers",
            "contract_value": 42000,
            "contract_start": today - timedelta(days=900),
            "contract_end": today + timedelta(days=15),
            "usage_pct": 90,
            "usage_decline_pct": 2,
            "asset_age_years": 4.2,
            "last_discount_pct": 10,
            "licensing": "Per-core",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10002",
            "customer": "Delta Inc",
            "customer_type": "SMB",
            "product": "Storage",
            "contract_value": 18000,
            "contract_start": today - timedelta(days=700),
            "contract_end": today + timedelta(days=25),
            "usage_pct": 30,
            "usage_decline_pct": 55,
            "asset_age_years": 2.1,
            "last_discount_pct": 25,  # guardrail breach for High
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10003",
            "customer": "Zento Pvt Ltd",
            "customer_type": "Enterprise",
            "product": "Networking",
            "contract_value": 68000,
            "contract_start": today - timedelta(days=1200),
            "contract_end": today + timedelta(days=75),
            "usage_pct": 85,
            "usage_decline_pct": 10,
            "asset_age_years": 5.1,
            "last_discount_pct": 12,
            "licensing": "Enterprise",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10004",
            "customer": "Nimbus Labs",
            "customer_type": "SMB",
            "product": "Software",
            "contract_value": 9000,
            "contract_start": today - timedelta(days=400),
            "contract_end": today + timedelta(days=180),
            "usage_pct": 92,
            "usage_decline_pct": 0,
            "asset_age_years": 1.0,
            "last_discount_pct": 3,
            "licensing": "User-based",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10005",
            "customer": "Orion Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10006",
            "customer": "Or Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10007",
            "customer": "Ion Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10008",
            "customer": "Orion Bank",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10009",
            "customer": "AL Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10010",
            "customer": "BL Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
        {
            "asset_id": "A-10011",
            "customer": "TL Systems",
            "customer_type": "Enterprise",
            "product": "Storage",
            "contract_value": 32000,
            "contract_start": today - timedelta(days=800),
            "contract_end": today + timedelta(days=60),
            "usage_pct": 78,
            "usage_decline_pct": 15,
            "asset_age_years": 3.8,
            "last_discount_pct": 20,  # guardrail breach for Medium
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },{
            "asset_id": "A-10012",
            "customer": "Renewal Inc",
            "customer_type": "SMB",
            "product": "Storage",
            "contract_value": 18000,
            "contract_start": today - timedelta(days=700),
            "contract_end": today + timedelta(days=25),
            "usage_pct": 30,
            "usage_decline_pct": 55,
            "asset_age_years": 2.1,
            "last_discount_pct": 30,  # guardrail breach for High
            "licensing": "Capacity",
            "country": "US",
            "region": "North America",

        },
    ]
    df = pd.DataFrame(data)
    df["days_to_expiry"] = (df["contract_end"] - today).dt.days
    return df

# =========================
# AGENTS
# =========================
def run_agents(df: pd.DataFrame, use_llm_flag: bool) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["days_to_expiry"] <= 30 or r["usage_decline_pct"] >= 40:
            priority, status = "High", "Act Now"
        elif r["contract_value"] > 25000 or r["days_to_expiry"] <= 90:
            priority, status = "Medium", "Good to Act"
        else:
            priority, status = "Low", "Monitor"

        expansion = (
            "Upsell" if r["usage_pct"] >= 80
            else "Cross-sell" if r["asset_age_years"] >= 3
            else "Renewal Only"
        )

        discount_map = {"High": 0.15, "Medium": 0.07, "Low": 0.02}
        expected_revenue = r["contract_value"] * (1 - discount_map[priority])

        # simple P2C for MVP
        p2c = 75 if priority == "High" else 55 if priority == "Medium" else 30

        rows.append({
            **r,
            "opportunity_priority": priority,
            "opportunity_status": status,
            "upsell_cross_sell": expansion,
            "expected_revenue_impact": round(expected_revenue, 0),
            "probability_to_close": p2c,
            "llm_explanation": llm_explain(r) if use_llm_flag else "Rule-based decision",
        })

    return pd.DataFrame(rows)

# =========================
# QUOTES + GUARDRAILS
# =========================
def check_discount_guardrail(priority: str, discount_pct: float) -> bool:
    max_allowed = NEGOTIATION_GUARDRAILS.get(priority, {}).get("max_discount", 0)
    return float(discount_pct) > float(max_allowed)

def build_quote(
    asset_row,
    version=1,
    parent_quote_id=None,
    discount_reason="Initial system generated discount",
    discount_source="rules_engine",
    previous_discount=None
):

    quote_id = f"{asset_row['asset_id']}-v{version}"

    base_price = float(asset_row["contract_value"])
    add_on_price = 5000.0 if asset_row["upsell_cross_sell"] in ["Upsell", "Cross-sell"] else 0.0

    expansion_type = asset_row.get("upsell_cross_sell", "Renewal Only")

    if expansion_type == "Upsell":
        skus = [{
            "sku": "SUP-PSP-PLUS",
            "item": "ProSupport Plus",
            "price": base_price
        }]
    else:
        skus = [{
            "sku": "SUP-PSP",
            "item": "ProSupport",
            "price": base_price
        }]

    if add_on_price > 0:
        skus.append({"sku": "ANL-ADV-02", "item": "Advanced Analytics", "price": add_on_price})

    subtotal = sum(x["price"] for x in skus)

    discount_pct = float(asset_row["last_discount_pct"])
    discount_amt = round(subtotal * discount_pct / 100.0, 2)
    total = round(subtotal - discount_amt, 2)
    # --- Service level upgrade logic (renewal upsell) ---
    # --- Service level logic (explicit & safe) ---
    expansion_type = asset_row.get("upsell_cross_sell", "Renewal Only")

    if expansion_type == "Upsell":
        service_level = "ProSupport Plus"
    else:
        service_level = "ProSupport"

    return {
        "quote_id": quote_id,
        "version": version,
        "parent_quote_id": parent_quote_id,
        "asset_id": asset_row["asset_id"],
        "customer": asset_row["customer"],
        "created_at": pd.Timestamp.now(),
        "status": "PENDING",
        "pricing": {
            "skus": skus,
            "subtotal": subtotal,
            "discount_pct": discount_pct,
            "discount_amt": discount_amt,
            "discount_reason": discount_reason,
            "discount_source": discount_source,
            "total": total,
            "discount_change": {
                "previous": previous_discount,
                "current": discount_pct
            },

        },
        "contract": {
                "start": pd.Timestamp.today().normalize(),
                "end": pd.Timestamp.today().normalize() + timedelta(days=365),
                "service_level": service_level,
        },
        "decision": None,
        "approval": {
            "required": False,
            "approved": False,
            "approved_at": None,
        }
    }

def negotiation_agent(asset_row, rejection_reason: str, use_llm_flag: bool):
    priority = asset_row["opportunity_priority"]
    base_discount = float(asset_row["last_discount_pct"])
    max_discount = float(NEGOTIATION_GUARDRAILS[priority]["max_discount"])

    intent = llm_negotiate(rejection_reason) if use_llm_flag else "price" if "price" in rejection_reason.lower() else "unclear"

    if intent == "price":
        new_discount = min(base_discount + 5, max_discount)
        if new_discount > base_discount:
            return {"action": "new_quote", "new_discount": new_discount, "message": "Offering revised quote with higher discount."}
        return {"action": "sales_intervention", "message": "Max discount reached. Escalate to sales."}

    if intent == "hardware_change":
        return {"action": "create_lead", "message": "Customer planning hardware change. New sales lead created."}

    if intent == "timing":
        return {"action": "on_hold", "message": "Opportunity put on hold."}

    return {"action": "sales_intervention", "message": "Unable to auto-resolve. Sales follow-up required."}

# =========================
# DASHBOARD
# =========================
def render_dashboard():
    st.title("Renewal Opportunities Dashboard")
    st.caption("Enterprise renewal cockpit that surfaces what matters, with guardrails, explainability, and quote actions.")

    df_base = load_assets()

    if st.session_state.agent_df is None or run_agents_clicked:
        with st.spinner("🤖 Agents are analyzing renewals…"):
            st.session_state.agent_df = run_agents(df_base, use_llm)

    df = st.session_state.agent_df.copy()

    # Apply global filters (widgets created once in sidebar)
    df = df[
        (df["customer_type"].isin(customer_types_filter)) &
        (df["product"].isin(product_filter)) &
        (df["opportunity_priority"].isin(priority_filter)) &
        (df["days_to_expiry"] <= max_days_filter)
    ].copy()

    # KPI row (includes accept/reject KPIs)
    total_impact = float(df["expected_revenue_impact"].sum()) * float(portfolio_multiplier) if not df.empty else 0.0

    high_ct = int((df["opportunity_priority"] == "High").sum()) if not df.empty else 0
    med_ct = int((df["opportunity_priority"] == "Medium").sum()) if not df.empty else 0
    low_ct = int((df["opportunity_priority"] == "Low").sum()) if not df.empty else 0

    accept_ct = int(st.session_state.accept_count)
    reject_ct = int(st.session_state.reject_count)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🔴 High priority", high_ct)
    k2.metric("🟡 Medium priority", med_ct)
    k3.metric("🟢 Low priority", low_ct)
    k4.metric("Scaled revenue impact", money_m(total_impact))
    k5.metric("✅ Accepted quotes", accept_ct)
    k6.metric("❌ Rejected quotes", reject_ct)

    if use_llm:
        st.info("🤖 LLM mode enabled: explanations and negotiation intent are AI-assisted (local model).")

    st.divider()

    if df.empty:
        st.warning("No records match your filters.")
        return

    # Worklist table (restored layout)
    st.subheader("Today’s worklist")

    headers = ["", "Asset", "Customer", "Priority", "Status", "Expansion", "P2C", "Impact", "Action"]
    colw = [0.6, 1.2, 2.0, 1.2, 1.3, 1.4, 1.3, 1.4, 1.4]

    hcols = st.columns(colw)
    for i, h in enumerate(headers):
        hcols[i].markdown(f"**{h}**")

    for _, r in df.iterrows():
        asset_id = r["asset_id"]
        is_expanded = asset_id in st.session_state.expanded_rows

        rowcols = st.columns(colw)

        arrow = "▼" if is_expanded else "▶"
        if rowcols[0].button(arrow, key=f"expand_{asset_id}"):
            if is_expanded:
                st.session_state.expanded_rows.remove(asset_id)
            else:
                st.session_state.expanded_rows.add(asset_id)
            st.rerun()

        rowcols[1].markdown(f"**{r['asset_id']}**")
        rowcols[2].write(r["customer"])
        rowcols[3].markdown(priority_badge(r["opportunity_priority"]))
        rowcols[4].markdown(status_badge(r["opportunity_status"]))
        rowcols[5].write(r["upsell_cross_sell"])
        rowcols[6].markdown(f"**{int(r['probability_to_close'])}% — {p2c_badge(int(r['probability_to_close']))}**")
        rowcols[7].write(money(r["expected_revenue_impact"]))

        action_label = "Generate Quote" if r["opportunity_priority"] in ["High", "Medium"] else "Review"
        if rowcols[8].button(action_label, key=f"quote_{asset_id}"):
            # create or pick latest quote
            existing = [q for q in st.session_state.quotes.values() if q["asset_id"] == r["asset_id"]]
            if not existing:
                q = build_quote(r, version=1)
                st.session_state.quotes[q["quote_id"]] = q
                st.session_state.current_quote_id = q["quote_id"]
            else:
                latest = max(existing, key=lambda x: x["version"])
                st.session_state.current_quote_id = latest["quote_id"]

            st.session_state.selected_asset = r
            st.session_state.show_email_block = True
            st.session_state.quote_entry_mode = "initial"
            st.session_state.page = "quote"
            st.rerun()

        # Expanded details card (restored)
        if is_expanded:
            contract_start = pd.to_datetime(r["contract_start"]).date()
            contract_end = pd.to_datetime(r["contract_end"]).date()
            guardrail_breach = check_discount_guardrail(r["opportunity_priority"], r["last_discount_pct"])

            st.markdown(
                f"""
                <div style="
                    margin-left:38px;
                    padding:12px 16px;
                    border-left:3px solid #e5e7eb;
                    background:#fafafa;
                    border-radius:6px;
                ">
                <b>Asset ID:</b> {r["asset_id"]} &nbsp; | &nbsp;
                <b>Product:</b> {r["product"]} &nbsp; | &nbsp;
                <b>Customer Type:</b> {r["customer_type"]}<br><br>
                <b>Current Support Level:</b> {r.get("service_level_current", "ProSupport")}<br>
                {"<b>Recommended Upgrade:</b> " + r.get("service_level_upgrade", "ProSupport Plus") + "<br><br>"
                if r["upsell_cross_sell"] == "Upsell" else "<br>"}


                <b>Contract:</b> {money(r["contract_value"])} ({contract_start} → {contract_end})<br>
                <b>Days to expiry:</b> {int(r["days_to_expiry"])} days<br><br>

                <b>Usage:</b> {pct(r["usage_pct"])} &nbsp; | &nbsp;
                <b>Usage decline:</b> {pct(r["usage_decline_pct"])}<br>
                <b>Asset age:</b> {r["asset_age_years"]} years<br>
                <b>Licensing:</b> {r["licensing"]}<br>
                <b>Last discount:</b> {int(r["last_discount_pct"])}% {"<span style='color:#b91c1c; font-weight:700;'>(Guardrail breach)</span>" if guardrail_breach else ""}<br><br>

                <b>Agent explanation:</b><br>
                {r["llm_explanation"] if use_llm else "Rule-based decision. Enable LLM for explanation."}
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================
# QUOTE PAGE (RESTORED FLOW)
# =========================
def render_quote():
    r = st.session_state.selected_asset
    quote = st.session_state.quotes.get(st.session_state.current_quote_id)
    if st.session_state.quote_entry_mode == "initial":
        st.session_state.show_email_block = True

    if r is None or quote is None:
        st.warning("No asset/quote selected.")
        st.session_state.page = "dashboard"
        st.rerun()

    st.title("📄 Renewal Quote")
    st.caption("Review a clean summary, guardrails, and decision actions (Accept/Reject).")
    if st.session_state.show_email_block and st.session_state.quote_entry_mode == "initial":
        st.markdown(
            f"""
            <div id="email-block"></div>
            <script>
                setTimeout(() => {{
                    document.getElementById("email-block").scrollIntoView({{ behavior: "smooth" }});
                }}, 300);
            </script>

            <div style="
                padding:14px 18px;
                background:#f8fafc;
                border:1px solid #e5e7eb;
                border-radius:8px;
                margin-bottom:16px;
            ">
            <b>Subject:</b> Renewal Offer for Asset {r["asset_id"]}<br><br>

            Hello,<br><br>

            Your asset <b>{r["asset_id"]}</b> is due for renewal.<br>
            Please find the attached offer for renewing services on this asset,
            ensuring a hassle-free and timely renewal.<br><br>

            <a href="#pricing-section" style="color:#2563eb; font-weight:600;">
                View Quote Details →
            </a><br><br>

            Thank you,<br>
            <i>Services Renewals Team</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div id="pricing-section"></div>', unsafe_allow_html=True)

    if st.session_state.quote_entry_mode == "regenerated":
        st.success("✨ Here is a new quote for you. Hope you will like the new offer.")
        change = quote["pricing"].get("discount_change")
        if change and change["previous"] is not None:
            st.info(
                f"💰 Discount updated: "
                f"{change['previous']:.0f}% → {change['current']:.0f}%"
            )

    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.divider()

    # Top summary
    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.2, 1.2])
    c1.metric("Customer", r["customer"])
    c2.metric("Asset", r["asset_id"])
    c3.metric("Priority", priority_badge(r["opportunity_priority"]))
    c4.metric("Expected impact", money(r["expected_revenue_impact"]))

    st.divider()

    # Quote history timeline (restored)
    with st.expander("Quote history timeline"):
        history = [q for q in st.session_state.quotes.values() if q["asset_id"] == r["asset_id"]]
        history = sorted(history, key=lambda x: x["version"])

        for q in history:
            pricing = q["pricing"]
            breach = check_discount_guardrail(r["opportunity_priority"], pricing["discount_pct"])
            title = f"v{q['version']} — {q['status']} | {pricing['discount_pct']}% discount | Total: {money(pricing['total'])}"
            st.markdown(f"**{title}**")
            st.caption(f"Created: {pd.to_datetime(q['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"Reason: {pricing.get('discount_reason', '—')} (Source: {pricing.get('discount_source', 'system')})")
            if breach:
                st.warning(f"⚠️ Guardrail breach (max {NEGOTIATION_GUARDRAILS[r['opportunity_priority']]['max_discount']}%). Approval required.")
            if q.get("decision"):
                st.info(f"Decision: {q['decision']['decision']} — {q['decision']['reason']}")

            st.divider()

    # Pricing + contract view
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Service & pricing")

        pricing = quote["pricing"]
        skus = pricing["skus"]
        st.table(pd.DataFrame(skus))

        p1, p2, p3 = st.columns(3)
        p1.metric("Subtotal", money(pricing["subtotal"]))
        p2.metric("Discount", f"{pricing['discount_pct']:.0f}% ({money(pricing['discount_amt'])})")
        change = pricing.get("discount_change")
        if change and change["previous"] is not None:
            st.markdown(
                f"""
                <div style="
                    padding:8px 12px;
                    background:#f1f5f9;
                    border-radius:6px;
                    font-size:14px;
                ">
                <b>Discount change:</b>
                {change['previous']:.0f}% → <b>{change['current']:.0f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        p3.metric("Total", money(pricing["total"]))

        st.caption("Discount guardrails applied by priority; breaches require approval.")

    with right:
        st.subheader("Contract")
        st.write({
            "Service level": quote["contract"]["service_level"],
            "Start": str(pd.to_datetime(quote["contract"]["start"]).date()),
            "End": str(pd.to_datetime(quote["contract"]["end"]).date()),
        })

        with st.expander("Why this quote? (Explainability)"):
            st.write(r["llm_explanation"] if use_llm else "Rule-based decision. Enable LLM for explanation.")

    st.divider()

    # Guardrail breach approval flow (restored)
    priority = r["opportunity_priority"]
    discount_pct = float(quote["pricing"]["discount_pct"])
    breach = check_discount_guardrail(priority, discount_pct)

    if breach:
        quote["approval"]["required"] = True
        st.warning(
            f"⚠️ Approval required: discount {discount_pct:.0f}% exceeds {NEGOTIATION_GUARDRAILS[priority]['max_discount']}% max for {priority} priority."
        )
        approved = st.checkbox(
            "I approve this exception (mock approval)",
            value=quote["approval"].get("approved", False),
            key=f"approve_{quote['quote_id']}"
        )
        if approved and not quote["approval"]["approved"]:
            quote["approval"]["approved"] = True
            quote["approval"]["approved_at"] = pd.Timestamp.now()
            st.session_state.approval_count += 1
            st.success("✅ Exception approved (mock). You can proceed to accept.")
    else:
        quote["approval"]["required"] = False
        quote["approval"]["approved"] = True  # no approval needed

    # Decision actions (Accept / Reject)
    st.subheader("Decision")
    a, b = st.columns([1, 1])

    can_accept = (not breach) or (breach and quote["approval"]["approved"])

    if a.button("✅ Accept Quote", type="primary", disabled=not can_accept):
        quote["status"] = "ACCEPTED"
        quote["decision"] = {"decision": "ACCEPTED", "reason": "Customer accepted", "timestamp": pd.Timestamp.now()}
        st.session_state.accept_count += 1
        st.success("Quote accepted. (Mock order placed / email sent)")
        st.session_state.page = "dashboard"
        st.rerun()

    if b.button("❌ Reject Quote"):
        st.session_state.page = "reject"
        st.rerun()

# =========================
# REJECTION PAGE (AUTO RENEGOTIATION)
# =========================
def render_reject():
    r = st.session_state.selected_asset
    # --- sync updated discount back into agent_df ---

    current_quote = st.session_state.quotes.get(st.session_state.current_quote_id)
    # --- SAFETY: ensure email always shows for initial quote ---


    if r is None or current_quote is None:
        st.warning("No asset/quote selected.")
        st.session_state.page = "dashboard"
        st.rerun()

    st.title("❌ Quote Rejected")
    st.caption("Provide a reason; the negotiation agent will recommend next best action.")

    if st.button("← Back to Quote"):
        st.session_state.page = "quote"
        st.rerun()



    st.divider()

    reason = st.text_area(
        "Rejection reason",
        placeholder="Example: Price too high. We are reviewing budgets for next quarter.",
        height=120,
        key=f"reject_reason_{current_quote['quote_id']}"
    )

    if st.button("Submit and get recommendation", type="primary"):
        # -----------------------------
        # Simulate agent processing delay
        # -----------------------------
        with st.spinner("🤖 Our agents are analyzing pricing, usage, and guardrails…"):
            time.sleep(4)

        with st.spinner("🧠 Optimizing the best possible offer for you…"):
            time.sleep(3)

        # mark rejected
        current_quote["status"] = "REJECTED"
        current_quote["decision"] = {
            "decision": "REJECTED",
            "reason": reason,
            "timestamp": pd.Timestamp.now()
        }
        st.session_state.reject_count += 1

        decision = negotiation_agent(r, reason, use_llm)
        # If discount guardrail is breached, delay response
        guardrail_breached = check_discount_guardrail(
            r["opportunity_priority"],
            r["last_discount_pct"]
        )

        if guardrail_breached:
            st.info("🕒 Thank you for your patience. We will be back with a new offer in 2–3 business days.")
            time.sleep(2)
            st.session_state.page = "dashboard"
            st.rerun()

        st.divider()
        st.subheader("System recommendation (Negotiation Agent)")
        st.info(decision["message"])

        if decision["action"] == "new_quote":
            prev = current_quote
            new_version = int(prev["version"]) + 1

            # Update asset discount before building new quote (MVP behavior)
            previous_discount = float(current_quote["pricing"]["discount_pct"])
            new_discount = float(decision["new_discount"])



            updated_asset = dict(r)
            updated_asset["last_discount_pct"] = new_discount

            mask = st.session_state.agent_df["asset_id"] == updated_asset["asset_id"]
            st.session_state.agent_df.loc[mask, "last_discount_pct"] = new_discount

            new_quote = build_quote(
                updated_asset,
                version=new_version,
                parent_quote_id=prev["quote_id"],
                discount_reason=f"Customer rejected due to price: {reason[:120]}",
                discount_source="negotiation_agent",
                previous_discount=previous_discount
            )

            st.session_state.quotes[new_quote["quote_id"]] = new_quote
            st.session_state.current_quote_id = new_quote["quote_id"]
            st.session_state.selected_asset = updated_asset

            st.success(f"New quote version v{new_version} created with {decision['new_discount']:.0f}% discount")
            st.session_state.show_email_block = False
            st.session_state.quote_entry_mode = "regenerated"

            st.session_state.page = "quote"
            st.rerun()

        elif decision["action"] == "create_lead":
            st.success("Sales Lead Created (mock).")
            st.json({
                "lead_type": "Hardware Refresh",
                "customer": r["customer"],
                "asset_id": r["asset_id"],
                "notes": reason[:500],
            })
            st.session_state.page = "dashboard"
            st.rerun()

        elif decision["action"] == "on_hold":
            st.warning("Opportunity marked as On Hold (mock).")
            st.session_state.page = "dashboard"
            st.rerun()

        else:
            st.warning("Sales follow-up required (mock).")
            st.session_state.page = "dashboard"
            st.rerun()

# =========================
# ROUTER
# =========================
if st.session_state.page == "dashboard":
    render_dashboard()
elif st.session_state.page == "quote":
    render_quote()
elif st.session_state.page == "reject":
    render_reject()

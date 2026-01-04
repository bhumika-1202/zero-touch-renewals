import pandas as pd
import streamlit as st
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
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

if "selected_asset" not in st.session_state:
    st.session_state.selected_asset = None

if "agent_df" not in st.session_state:
    st.session_state.agent_df = None

if "expanded_rows" not in st.session_state:
    st.session_state.expanded_rows = set()

if "agents_running" not in st.session_state:
    st.session_state.agents_running = False

# =========================
# QUOTE STORE (MVP)
# =========================
if "quotes" not in st.session_state:
    st.session_state.quotes = {}   # quote_id -> quote payload

if "current_quote_id" not in st.session_state:
    st.session_state.current_quote_id = None



# =========================
# LOAD LOCAL LLM (SAFE)
# =========================
@st.cache_resource
def load_llm():
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
        if "hardware" in result or "replace" in result:
            return "hardware_change"
        if "later" in result or "budget" in result:
            return "timing"
        return "unclear"
    except Exception:
        return "unclear"

def negotiation_agent(asset_row, rejection_reason):
    priority = asset_row["opportunity_priority"]
    base_discount = asset_row["last_discount_pct"]
    max_discount = NEGOTIATION_GUARDRAILS[priority]["max_discount"]

    intent = llm_negotiate(rejection_reason)

    if intent == "price":
        new_discount = min(base_discount + 5, max_discount)
        if new_discount > base_discount:
            return {
                "action": "new_quote",
                "new_discount": new_discount,
                "message": "Offering revised quote with higher discount."
            }
        return {
            "action": "sales_intervention",
            "message": "Max discount reached. Escalate to sales."
        }

    if intent == "hardware_change":
        return {
            "action": "create_lead",
            "message": "Customer planning hardware change. New sales lead created."
        }

    if intent == "timing":
        return {
            "action": "on_hold",
            "message": "Opportunity put on hold."
        }

    return {
        "action": "sales_intervention",
        "message": "Unable to auto-resolve. Sales follow-up required."
    }

def calculate_probability_to_close(row):
    score = 50

    if row["days_to_expiry"] <= 30:
        score += 20
    elif row["days_to_expiry"] <= 60:
        score += 10

    if row["usage_pct"] >= 80:
        score += 20
    if row["usage_decline_pct"] >= 40:
        score -= 15

    if row["asset_age_years"] >= 4:
        score -= 10

    if row["opportunity_priority"] == "High":
        score += 15
    elif row["opportunity_priority"] == "Medium":
        score += 5

    if row["upsell_cross_sell"] in ["Upsell", "Cross-sell"]:
        score += 10

    if row["last_discount_pct"] >= 15:
        score -= 10

    return max(0, min(100, score))

def p2c_badge(score: int) -> str:
    if score >= 70:
        return "🟢 High"
    if score >= 40:
        return "🟡 Medium"
    return "🔴 Low"

def llm_adjust_probability(row, base_score):
    try:
        llm = load_llm()
        prompt = (
            "Given the following renewal data, should probability to close be "
            "slightly higher, lower, or unchanged? Answer only: higher, lower, unchanged.\n\n"
            f"Days to expiry: {row['days_to_expiry']}\n"
            f"Usage %: {row['usage_pct']}\n"
            f"Usage decline %: {row['usage_decline_pct']}\n"
            f"Contract value: {row['contract_value']}\n"
            f"Asset age: {row['asset_age_years']}\n"
        )

        result = llm(prompt)[0]["generated_text"].lower()

        if "higher" in result:
            return min(100, base_score + 5)
        if "lower" in result:
            return max(0, base_score - 5)
        return base_score
    except Exception:
        return base_score

# =========================
# SAMPLE DATA
# =========================
def load_assets():
    today = pd.Timestamp.today()
    data = [
        {
            "asset_id": "A-10001",
            "customer": "ABC Corp",
            "customer_type": "Enterprise",
            "product": "Servers",
            "contract_value": 42000,
            "contract_start": today - timedelta(days=900),
            "contract_end": today + timedelta(days=120),
            "usage_pct": 88,
            "usage_decline_pct": 5,
            "asset_age_years": 4.2,
            "last_discount_pct": 10,
            "licensing": "Per-core",
        },
        {
            "asset_id": "A-10002",
            "customer": "Delta Inc",
            "customer_type": "SMB",
            "product": "Storage",
            "contract_value": 12000,
            "contract_start": today - timedelta(days=600),
            "contract_end": today + timedelta(days=45),
            "usage_pct": 35,
            "usage_decline_pct": 55,
            "asset_age_years": 1.3,
            "last_discount_pct": 18,
            "licensing": "Capacity",
        },
        {
            "asset_id": "A-10003",
            "customer": "Zento Pvt Ltd",
            "customer_type": "Enterprise",
            "product": "Networking",
            "contract_value": 68000,
            "contract_start": today - timedelta(days=1200),
            "contract_end": today + timedelta(days=20),
            "usage_pct": 92,
            "usage_decline_pct": 0,
            "asset_age_years": 5.1,
            "last_discount_pct": 5,
            "licensing": "Enterprise",
        },
    ]
    df = pd.DataFrame(data)
    df["days_to_expiry"] = (df["contract_end"] - today).dt.days
    return df

# =========================
# AGENTS
# =========================
def run_agents(df: pd.DataFrame, use_llm: bool) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["days_to_expiry"] <= 30 or r["usage_decline_pct"] >= 40:
            priority = "High"
            status = "Act Now"
        elif r["contract_value"] > 25000:
            priority = "Medium"
            status = "Good to Act"
        else:
            priority = "Low"
            status = "On Hold"

        if r["usage_pct"] >= 80:
            expansion = "Upsell"
        elif r["asset_age_years"] >= 3:
            expansion = "Cross-sell"
        else:
            expansion = "Renewal Only"

        discount_map = {"High": 0.15, "Medium": 0.07, "Low": 0.02}
        expected_revenue = r["contract_value"] * (1 - discount_map[priority])

        base_p2c = calculate_probability_to_close({
            **r,
            "opportunity_priority": priority,
            "upsell_cross_sell": expansion
        })

        final_p2c = llm_adjust_probability(r, base_p2c) if use_llm else base_p2c

        rows.append({
            **r,
            "opportunity_priority": priority,
            "opportunity_status": status,
            "upsell_cross_sell": expansion,
            "expected_revenue_impact": round(expected_revenue, 0),
            "probability_to_close": final_p2c,
            "llm_explanation": llm_explain(r) if use_llm else "Rule-based decision",
        })

    return pd.DataFrame(rows)

# =========================
# UI HELPERS
# =========================
def priority_badge(p: str) -> str:
    return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(p, p)

def status_badge(s: str) -> str:
    return {"Act Now": "⚡ Act Now", "Good to Act": "✅ Good to Act", "On Hold": "⏸️ On Hold"}.get(s, s)

def money(x) -> str:
    return f"${float(x):,.0f}"

def pct(x) -> str:
    return f"{int(x)}%"

def build_quote(
    asset_row,
    version=1,
    parent_quote_id=None,
    discount_reason="Initial system generated discount",
    discount_source="rules_engine"
):
    quote_id = f"{asset_row['asset_id']}-v{version}"

    base_price = float(asset_row["contract_value"])
    add_on_price = 5000.0 if asset_row["upsell_cross_sell"] in ["Upsell", "Cross-sell"] else 0.0

    skus = [
        {"sku": "SUP-PREM-01", "item": "Premium Support", "price": base_price}
    ]
    if add_on_price > 0:
        skus.append({"sku": "ANL-ADV-02", "item": "Advanced Analytics", "price": add_on_price})

    subtotal = sum(x["price"] for x in skus)

    discount_pct = float(asset_row["last_discount_pct"])
    discount_amt = round(subtotal * discount_pct / 100, 2)
    total = round(subtotal - discount_amt, 2)

    return {
        "quote_id": quote_id,
        "version": version,
        "parent_quote_id": parent_quote_id,
        "asset_id": asset_row["asset_id"],
        "customer": asset_row["customer"],
        "created_at": pd.Timestamp.now(),
        "status": "PENDING",

        # 🔹 Pricing + discount metadata
        "pricing": {
            "skus": skus,
            "subtotal": subtotal,
            "discount_pct": discount_pct,
            "discount_amt": discount_amt,
            "discount_reason": discount_reason,
            "discount_source": discount_source,
            "total": total,
        },

        # 🔹 Contract
        "contract": {
            "start": pd.Timestamp.today().normalize(),
            "end": pd.Timestamp.today().normalize() + timedelta(days=365),
            "service_level": "Premium Support",
        },

        # 🔹 Decision history
        "decision": None,
    }

# =========================
# DASHBOARD PAGE (REDESIGNED)
# =========================
def render_dashboard():
    st.title("Zero-Touch Service Renewals")
    st.caption("A renewal cockpit that surfaces what matters, with detail-on-demand and a guided quote flow.")

    df_base = load_assets()

    # Sidebar controls + filters (kept, but grouped)
    st.sidebar.header("Controls")
    use_llm = st.sidebar.checkbox("Use LLM (local) for explainability", value=False)

    if st.sidebar.button("Run Agents", type="primary"):
        with st.spinner("🤖 Agents are analyzing renewals, risk, and pricing…"):
            st.session_state.agent_df = run_agents(df_base, use_llm)
        st.rerun()

    if st.session_state.agent_df is None:
        st.session_state.agent_df = run_agents(df_base, use_llm=False)

    df = st.session_state.agent_df.copy()

    st.sidebar.divider()
    st.sidebar.header("Filters")

    customer_types = st.sidebar.multiselect(
        "Customer Type",
        options=sorted(df["customer_type"].unique()),
        default=sorted(df["customer_type"].unique())
    )
    products = st.sidebar.multiselect(
        "Product",
        options=sorted(df["product"].unique()),
        default=sorted(df["product"].unique())
    )
    priorities = st.sidebar.multiselect(
        "Priority",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    max_days = st.sidebar.slider("Max days to expiry", 0, 365, 90)

    df = df[
        (df["customer_type"].isin(customer_types)) &
        (df["product"].isin(products)) &
        (df["opportunity_priority"].isin(priorities)) &
        (df["days_to_expiry"] <= max_days)
    ].copy()

    # KPI row (Layer 1)
    high_ct = int((df["opportunity_priority"] == "High").sum()) if not df.empty else 0
    med_ct = int((df["opportunity_priority"] == "Medium").sum()) if not df.empty else 0
    low_ct = int((df["opportunity_priority"] == "Low").sum()) if not df.empty else 0
    total_impact = float(df["expected_revenue_impact"].sum()) if not df.empty else 0
    avg_p2c = int(round(float(df["probability_to_close"].mean()))) if not df.empty else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🔴 High priority", high_ct)
    k2.metric("🟡 Medium priority", med_ct)
    k3.metric("🟢 Low priority", low_ct)
    k4.metric("Expected revenue impact", money(total_impact))
    k5.metric("Avg probability to close", f"{avg_p2c}%")

    st.divider()

    # Tabs (Layer 2/3)
    tab_worklist, tab_insights = st.tabs(["Worklist", "Insights"])

    with tab_worklist:
        st.subheader("Today’s worklist")

        if st.session_state.agents_running:
            with st.spinner("🤖 Agents are analyzing renewals, risk, and pricing…"):
                st.empty()

        if df.empty:
            st.info("No records match your filters.")
            return

        # Action-oriented table (reduced columns)
        headers = ["", "Asset", "Customer", "Priority", "Status", "Expansion", "P2C", "Impact", "Action"]
        colw = [0.6, 1.2, 2.0, 1.2, 1.3, 1.4, 1.3, 1.4, 1.4]
        hcols = st.columns(colw)
        for i, h in enumerate(headers):
            hcols[i].markdown(f"**{h}**")

        for _, r in df.iterrows():
            asset_id = r["asset_id"]
            is_expanded = asset_id in st.session_state.expanded_rows

            rowcols = st.columns(colw)

            # ▶ / ▼ expansion arrow (first column)
            # ▶ / ▼ expansion arrow (column 0)
            arrow = "▼" if is_expanded else "▶"
            if rowcols[0].button(arrow, key=f"expand_{asset_id}"):
                if is_expanded:
                    st.session_state.expanded_rows.remove(asset_id)
                else:
                    st.session_state.expanded_rows.add(asset_id)
                st.rerun()

            # Asset ID (always visible)
            rowcols[1].markdown(f"**{r['asset_id']}**")

            # Remaining columns
            rowcols[2].write(r["customer"])
            rowcols[3].markdown(priority_badge(r["opportunity_priority"]))
            rowcols[4].markdown(status_badge(r["opportunity_status"]))
            rowcols[5].write(r["upsell_cross_sell"])

            p2c = int(r["probability_to_close"])
            rowcols[6].markdown(f"**{p2c}% — {p2c_badge(p2c)}**")

            rowcols[7].write(money(r["expected_revenue_impact"]))

            action_label = (
                "Generate Quote"
                if r["opportunity_priority"] in ["High", "Medium"]
                else "Review"
            )
            if rowcols[8].button(action_label, key=f"quote_{asset_id}"):

                # Create quote only if it doesn't exist
                existing = [
                    q for q in st.session_state.quotes.values()
                    if q["asset_id"] == r["asset_id"]
                ]

                if not existing:
                    quote = build_quote(r, version=1)
                    st.session_state.quotes[quote["quote_id"]] = quote
                    st.session_state.current_quote_id = quote["quote_id"]
                else:
                    # pick latest version
                    latest = max(existing, key=lambda q: q["version"])
                    st.session_state.current_quote_id = latest["quote_id"]

                st.session_state.selected_asset = r
                st.session_state.page = "quote"
                st.rerun()

            # =========================
            # Expanded row (detail-on-demand)
            # =========================
            if is_expanded:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            margin-left:38px;
                            padding:12px 16px;
                            border-left:3px solid #e5e7eb;
                            background:#fafafa;
                            border-radius:4px;
                        ">
                        <b>Asset ID:</b> {r["asset_id"]} &nbsp; | &nbsp;
                        <b>Product:</b> {r["product"]} &nbsp; | &nbsp;
                        <b>Customer Type:</b> {r["customer_type"]}<br><br>

                        <b>Contract:</b> {money(r["contract_value"])}  
                        ({pd.to_datetime(r["contract_start"]).date()} → {pd.to_datetime(r["contract_end"]).date()})  
                        <br>
                        <b>Days to expiry:</b> {int(r["days_to_expiry"])} days<br><br>

                        <b>Usage:</b> {pct(r["usage_pct"])} &nbsp; | &nbsp;
                        <b>Usage decline:</b> {pct(r["usage_decline_pct"])}<br>
                        <b>Asset age:</b> {r["asset_age_years"]} years<br>
                        <b>Licensing:</b> {r["licensing"]}<br>
                        <b>Last discount:</b> {int(r["last_discount_pct"])}%<br><br>

                        <b>Agent explanation:</b><br>
                        {r["llm_explanation"] if use_llm else "Rule-based decision. Enable LLM for explanation."}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_insights:
        st.subheader("Portfolio snapshot")
        st.caption("Lightweight insights for scanning. Keep the worklist as the primary surface.")

        by_prod = df.groupby("product", as_index=False).agg(
            assets=("asset_id", "count"),
            impact=("expected_revenue_impact", "sum"),
            avg_p2c=("probability_to_close", "mean"),
        )
        by_prod["avg_p2c"] = by_prod["avg_p2c"].round(0).astype(int)
        st.dataframe(by_prod, use_container_width=True)

# =========================
# QUOTE PAGE (REDESIGNED)
# =========================
def render_quote():
    quote = st.session_state.quotes.get(st.session_state.current_quote_id)
    r = st.session_state.selected_asset

    if r is None:
        st.warning("No asset selected.")
        st.session_state.page = "dashboard"
        st.rerun()

    st.title("📄 Renewal Quote")
    st.caption("Review a clean before/after summary, then accept or reject.")

    # Navigation
    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("← Back to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()

    st.divider()

    # Customer + Asset (compact)
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    c1.metric("Customer", r["customer"])
    c2.metric("Asset", r["asset_id"])
    c3.metric("Priority", priority_badge(r["opportunity_priority"]))

    st.divider()

    with st.expander("Quote history"):
        history = [
            q for q in st.session_state.quotes.values()
            if q["asset_id"] == r["asset_id"]
        ]
        history = sorted(history, key=lambda x: x["version"])

        for q in history:
            pricing = q["pricing"]
            label = f"v{q['version']} — {q['status']} | {pricing['discount_pct']}% discount"

            st.markdown(f"**{label}**")
            st.caption(
                f"Reason: {pricing.get('discount_reason', '—')} "
                f"(Source: {pricing.get('discount_source', 'system')})"
            )

            if q["status"] == "REJECTED" and q.get("decision"):
                st.warning(f"Rejected reason: {q['decision']['reason']}")

    # Quote "document" layout
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Service & pricing")

        # Simulated SKU selection (can be expanded later)
        base_price = float(r["contract_value"])
        add_on_price = 5000.0 if r["upsell_cross_sell"] in ["Upsell", "Cross-sell"] else 0.0

        quote_lines = [
            {"SKU": "SUP-PREM-01", "Item": "Premium Support", "Price": base_price},
        ]
        if add_on_price > 0:
            quote_lines.append({"SKU": "ANL-ADV-02", "Item": "Advanced Analytics Add-on", "Price": add_on_price})

        quote_df = pd.DataFrame(quote_lines)
        st.table(quote_df)

        discount_pct = float(r["last_discount_pct"])
        subtotal = float(quote_df["Price"].sum())
        discount_amt = round(subtotal * (discount_pct / 100.0), 2)
        total = round(subtotal - discount_amt, 2)

        st.markdown("### Pricing summary")
        p1, p2, p3 = st.columns(3)
        p1.metric("Subtotal", money(subtotal))
        p2.metric("Discount", f"{discount_pct:.0f}% ({money(discount_amt)})")
        p3.metric("Total", money(total))

    with right:
        st.subheader("Before vs after")

        new_start = pd.Timestamp.today().normalize()
        new_end = new_start + timedelta(days=365)

        current_service = "Standard Support"
        proposed_service = "Premium Support"

        compare = pd.DataFrame([
            {"": "Service level", "Current": current_service, "Proposed": proposed_service},
            {"": "Contract period", "Current": f"Ends {pd.to_datetime(r['contract_end']).date()}",
             "Proposed": f"{new_start.date()} → {new_end.date()}"},
            {"": "Discount", "Current": f"{int(r['last_discount_pct'])}%", "Proposed": f"{int(r['last_discount_pct'])}%"},
            {"": "Expected impact", "Current": "-", "Proposed": money(r["expected_revenue_impact"])},
        ])
        st.table(compare)

        st.markdown("### Asset context")
        st.write({
            "Product": r["product"],
            "Customer type": r["customer_type"],
            "Usage": pct(r["usage_pct"]),
            "Usage decline": pct(r["usage_decline_pct"]),
            "Asset age (yrs)": float(r["asset_age_years"]),
            "Licensing": r["licensing"],
        })

        with st.expander("Why this quote? (Explainability)"):
            st.write(r["llm_explanation"] if "llm_explanation" in r else "Rule-based decision")

    st.divider()

    # Decision area (clear single primary action)
    st.subheader("Decision")
    a, b = st.columns([1, 1])
    if a.button("✅ Accept Quote", type="primary"):
        st.success("Quote accepted. (Email sent – mock)")
        st.session_state.page = "dashboard"
        st.rerun()

    if b.button("❌ Reject Quote"):
        st.session_state.page = "reject"
        st.rerun()

# =========================
# REJECTION PAGE (REDESIGNED)
# =========================
def render_reject():
    r = st.session_state.selected_asset
    if r is None:
        st.warning("No asset selected.")
        st.session_state.page = "dashboard"
        st.rerun()

    st.title("❌ Quote Rejected")
    st.caption("Tell us why. The negotiation agent will propose the next best action.")



    if st.button("← Back to Quote"):
        st.session_state.page = "quote"
        st.rerun()

    st.divider()

    reason = st.text_area(
        "Step 1 — Rejection reason",
        placeholder="Example: Price too high. We’re considering a hardware refresh next quarter.",
        height=120,
    )
    current_quote = st.session_state.quotes[st.session_state.current_quote_id]

    # Save rejection
    current_quote["status"] = "REJECTED"
    current_quote["decision"] = {
        "decision": "REJECTED",
        "reason": reason,
        "timestamp": pd.Timestamp.now()
    }

    if st.button("Step 2 — Submit and get recommendation", type="primary"):
        decision = negotiation_agent(r, reason)

        st.divider()
        st.subheader("Step 3 — System recommendation (Negotiation Agent)")
        st.info(decision["message"])

        if decision["action"] == "new_quote":
            prev = current_quote
            new_version = prev["version"] + 1

            # Update asset discount before building new quote
            st.session_state.selected_asset["last_discount_pct"] = decision["new_discount"]

            new_quote = build_quote(
                st.session_state.selected_asset,
                version=new_version,
                parent_quote_id=prev["quote_id"],
                discount_reason=f"Customer rejected previous quote due to price: {reason[:120]}",
                discount_source="negotiation_agent"
            )

            st.session_state.quotes[new_quote["quote_id"]] = new_quote
            st.session_state.current_quote_id = new_quote["quote_id"]

            st.success(f"New quote version v{new_version} created with {decision['new_discount']}% discount")
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
            st.warning("Opportunity marked as On Hold.")
            st.session_state.page = "dashboard"
            st.rerun()

        else:
            st.warning("Sales follow-up required.")
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

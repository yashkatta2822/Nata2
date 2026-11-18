import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Campaign Response Predictor", page_icon="🛒", layout="wide")

MODEL_FILE = "nata_model.joblib"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
ACCEPT_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
BASE_COLS = (
    ["Age", "Income", "Education", "Marital_Status", "Kidhome", "Teenhome", "Recency"]
    + SPEND_COLS
    + ["NumDealsPurchases"]
    + PURCHASE_COLS
    + ["NumWebVisitsMonth"]
    + ACCEPT_COLS
    + ["Complain", "TenureDays"]
)
EDUCATION = ["Basic", "Graduation", "Master", "PhD"]
MARITAL = ["Single", "Married", "Together", "Divorced", "Widow"]


@st.cache_resource
def load_artifacts(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


artifacts = load_artifacts(MODEL_FILE)
if artifacts is None:
    st.error(f"Model file '{MODEL_FILE}' not found. Keep it in the same folder as app.py.")
    st.stop()

model = artifacts["model"]
model_columns = artifacts["columns"]


def build_features(raw):
    """Turn the base customer columns into exactly the features the model was trained on."""
    d = raw.copy()
    d["Education"] = d["Education"].replace({"2n Cycle": "Master"})
    d["Marital_Status"] = d["Marital_Status"].replace({"Alone": "Single"})

    d["TotalSpend"] = d[SPEND_COLS].sum(axis=1)
    d["Children"] = d["Kidhome"] + d["Teenhome"]
    d["HasChildren"] = (d["Children"] > 0).astype(int)
    d["TotalPurchases"] = d[PURCHASE_COLS].sum(axis=1)
    d["PastAccepted"] = d[ACCEPT_COLS].sum(axis=1)

    purchases = d["TotalPurchases"].replace(0, np.nan)
    spend = d["TotalSpend"].replace(0, np.nan)
    d["AvgBasket"] = (d["TotalSpend"] / purchases).fillna(0)
    d["WineShare"] = (d["MntWines"] / spend).fillna(0)
    d["MeatShare"] = (d["MntMeatProducts"] / spend).fillna(0)
    d["DealShare"] = (d["NumDealsPurchases"] / purchases).fillna(0)
    d["IsRecent"] = (d["Recency"] <= 30).astype(int)

    X = pd.get_dummies(d, columns=["Education", "Marital_Status"])
    return X.reindex(columns=model_columns, fill_value=0).astype(float)


def score(raw, cost, revenue):
    out = raw.copy()
    out["ResponseProbability"] = model.predict_proba(build_features(raw))[:, 1]
    out["ExpectedProfit"] = out["ResponseProbability"] * revenue - cost
    out["Decision"] = np.where(out["ResponseProbability"] >= cost / revenue, "Contact", "Skip")
    return out


# ---------------- Sidebar ----------------
st.sidebar.header("Campaign economics")
cost = st.sidebar.number_input("Cost per contact", min_value=0.1, value=float(artifacts.get("cost", 3)), step=0.5)
revenue = st.sidebar.number_input("Revenue per acceptance", min_value=0.1, value=float(artifacts.get("revenue", 11)), step=0.5)
break_even = cost / revenue
st.sidebar.metric("Break-even response rate", f"{break_even:.1%}")
st.sidebar.caption("A customer is worth contacting when their predicted chance of responding is above this level.")

# ---------------- Main ----------------
st.title("🛒 Campaign Response Predictor")
st.write("Estimate how likely a customer is to accept the next campaign offer, and whether contacting them makes money.")

tab_single, tab_batch = st.tabs(["Single customer", "Score a file"])

# ---- Single customer ----
with tab_single:
    with st.form("customer_form"):
        st.subheader("Profile")
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 90, 45)
        income = c1.number_input("Yearly income", min_value=0, value=50000, step=1000)
        tenure = c1.number_input("Days as a customer", min_value=0, value=500, step=10)
        education = c2.selectbox("Education", EDUCATION, index=1)
        marital = c2.selectbox("Marital status", MARITAL)
        complain = c2.selectbox("Complaint in last 2 years", ["No", "Yes"])
        kidhome = c3.number_input("Young children at home", min_value=0, max_value=5, value=0)
        teenhome = c3.number_input("Teenagers at home", min_value=0, max_value=5, value=0)
        recency = c3.slider("Days since last purchase", 0, 100, 30)

        st.subheader("Spending in the last 2 years")
        s1, s2, s3 = st.columns(3)
        wines = s1.number_input("Wine", min_value=0, value=300)
        fruits = s1.number_input("Fruits", min_value=0, value=25)
        meat = s2.number_input("Meat", min_value=0, value=150)
        fish = s2.number_input("Fish", min_value=0, value=30)
        sweets = s3.number_input("Sweets", min_value=0, value=25)
        gold = s3.number_input("Gold / premium products", min_value=0, value=40)

        st.subheader("Buying behaviour")
        b1, b2, b3 = st.columns(3)
        deals = b1.number_input("Purchases with a discount", min_value=0, value=2)
        web = b1.number_input("Web purchases", min_value=0, value=4)
        catalog = b2.number_input("Catalog purchases", min_value=0, value=2)
        store = b2.number_input("Store purchases", min_value=0, value=5)
        visits = b3.number_input("Web visits per month", min_value=0, value=5)

        st.subheader("Past campaigns accepted")
        a1, a2, a3, a4, a5 = st.columns(5)
        cmp1 = a1.checkbox("Campaign 1")
        cmp2 = a2.checkbox("Campaign 2")
        cmp3 = a3.checkbox("Campaign 3")
        cmp4 = a4.checkbox("Campaign 4")
        cmp5 = a5.checkbox("Campaign 5")

        submitted = st.form_submit_button("🔍 Predict response", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Age": age, "Income": income, "Education": education, "Marital_Status": marital,
            "Kidhome": kidhome, "Teenhome": teenhome, "Recency": recency,
            "MntWines": wines, "MntFruits": fruits, "MntMeatProducts": meat,
            "MntFishProducts": fish, "MntSweetProducts": sweets, "MntGoldProds": gold,
            "NumDealsPurchases": deals, "NumWebPurchases": web, "NumCatalogPurchases": catalog,
            "NumStorePurchases": store, "NumWebVisitsMonth": visits,
            "AcceptedCmp1": int(cmp1), "AcceptedCmp2": int(cmp2), "AcceptedCmp3": int(cmp3),
            "AcceptedCmp4": int(cmp4), "AcceptedCmp5": int(cmp5),
            "Complain": 1 if complain == "Yes" else 0, "TenureDays": tenure,
        }])
        result = score(row, cost, revenue).iloc[0]
        prob = float(result["ResponseProbability"])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Chance of responding", f"{prob:.1%}")
        m2.metric("Break-even level", f"{break_even:.1%}")
        m3.metric("Expected profit if contacted", f"{result['ExpectedProfit']:.2f}")
        st.progress(min(max(prob, 0.0), 1.0))

        if result["Decision"] == "Contact":
            st.success("✅ Worth contacting: the expected revenue is higher than the contact cost.")
        else:
            st.warning("⚠️ Skip: the expected revenue is lower than the contact cost.")

# ---- Batch scoring ----
with tab_batch:
    st.write(
        "Upload a CSV or Excel file with one customer per row. The file needs these columns: "
        + ", ".join(BASE_COLS)
        + ". The cleaned file from the notebook (nata_clean.csv) already has all of them."
    )
    uploaded = st.file_uploader("Customer file", type=["csv", "xlsx"])

    if uploaded is not None:
        data = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        missing = [c for c in BASE_COLS if c not in data.columns]
        if missing:
            st.error("These columns are missing from the file: " + ", ".join(missing))
        else:
            scored = score(data, cost, revenue)
            contacts = int((scored["Decision"] == "Contact").sum())
            profit_targeted = scored.loc[scored["Decision"] == "Contact", "ExpectedProfit"].sum()

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers in file", f"{len(scored):,}")
            k2.metric("Worth contacting", f"{contacts:,}")
            k3.metric("Expected profit (contacting only these)", f"{profit_targeted:,.0f}")

            st.dataframe(
                scored.sort_values("ResponseProbability", ascending=False).head(200),
            )
            st.caption("Showing the 200 customers most likely to respond. The download has every row.")
            st.download_button(
                "⬇️ Download scored customers (CSV)",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv",
            )

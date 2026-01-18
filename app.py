import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Aadhaar Hackathon Dashboard", page_icon="🪪", layout="wide")

# -----------------------------
# Auto load local CSVs (no upload UI if found)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_auto():
    candidates = {
        "enr": ["Enrollement.csv", "Enrollment.csv", "enrollment.csv"],
        "demo": ["Demographic.csv", "demographic.csv"],
        "bio": ["Biometrics.csv", "biometrics.csv"],
    }

    def first_existing(lst):
        for f in lst:
            if os.path.exists(f):
                return f
        return None

    enr_path = first_existing(candidates["enr"])
    demo_path = first_existing(candidates["demo"])
    bio_path = first_existing(candidates["bio"])

    if not (enr_path and demo_path and bio_path):
        return None, None, None, {"mode": "missing", "paths": (enr_path, demo_path, bio_path)}

    enrollment_df = pd.read_csv(enr_path)
    demographic_df = pd.read_csv(demo_path)
    biometrics_df = pd.read_csv(bio_path)

    for df in (enrollment_df, demographic_df, biometrics_df):
        df.columns = df.columns.str.strip()

    return enrollment_df, demographic_df, biometrics_df, {"mode": "local", "paths": (enr_path, demo_path, bio_path)}

def read_uploaded_or_stop():
    st.warning("Local CSVs not found. Upload once to run the dashboard.")
    up_enr = st.file_uploader("Upload Enrollment/Enrollement CSV", type=["csv"])
    up_demo = st.file_uploader("Upload Demographic CSV", type=["csv"])
    up_bio = st.file_uploader("Upload Biometrics CSV", type=["csv"])

    if not (up_enr and up_demo and up_bio):
        st.stop()

    enrollment_df = pd.read_csv(up_enr); enrollment_df.columns = enrollment_df.columns.str.strip()
    demographic_df = pd.read_csv(up_demo); demographic_df.columns = demographic_df.columns.str.strip()
    biometrics_df = pd.read_csv(up_bio); biometrics_df.columns = biometrics_df.columns.str.strip()
    return enrollment_df, demographic_df, biometrics_df

# -----------------------------
# Cleaning
# -----------------------------
def clean_common(df: pd.DataFrame, date_col="date", state_col="state"):
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if state_col in out.columns:
        out[state_col] = out[state_col].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    return out

def clean_enrollment(df):
    df = clean_common(df)
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    if "pincode" in df.columns:
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce").astype("Int64")
    return df

def clean_demographic(df):
    df = clean_common(df)
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    if "pincode" in df.columns:
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce").astype("Int64")
    return df

def clean_biometrics(df):
    df = clean_common(df)
    if "district_norm" in df.columns:
        df["district_norm"] = df["district_norm"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    elif "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    if "pincode" in df.columns:
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce").astype("Int64")
    return df

# -----------------------------
# Filters
# -----------------------------
def state_district_filter(df: pd.DataFrame, state_col: str, district_col: str, label_prefix: str):
    states = sorted(df[state_col].dropna().unique().tolist()) if state_col in df.columns else []
    state = st.sidebar.selectbox(f"{label_prefix} State", ["All"] + states, key=f"{label_prefix}_state")

    ddf = df.copy()
    if state != "All" and state_col in ddf.columns:
        ddf = ddf[ddf[state_col] == state]

    districts = sorted(ddf[district_col].dropna().unique().tolist()) if district_col in ddf.columns else []
    district = st.sidebar.selectbox(f"{label_prefix} District", ["All"] + districts, key=f"{label_prefix}_district")

    out = df.copy()
    if state != "All" and state_col in out.columns:
        out = out[out[state_col] == state]
    if district != "All" and district_col in out.columns:
        out = out[out[district_col] == district]

    return out, state, district

def date_filter(df: pd.DataFrame, date_col="date", label="Date range", key="date_range"):
    if date_col not in df.columns:
        return df
    min_d = df[date_col].min()
    max_d = df[date_col].max()
    if pd.isna(min_d) or pd.isna(max_d):
        return df

    dr = st.sidebar.date_input(label, (min_d.date(), max_d.date()), key=key)
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        return df[(df[date_col] >= start) & (df[date_col] <= end)]
    return df

# -----------------------------
# Render sections
# -----------------------------
def render_enrollment(enr_df):
    st.header("1) Enrollment")

    total = enr_df["total_enrolment"].fillna(0).sum() if "total_enrolment" in enr_df.columns else 0
    st.metric("Total Enrollment", f"{int(total):,}")

    # Top 10 States
    if "state" in enr_df.columns and "total_enrolment" in enr_df.columns:
        st.subheader("Top 10 States (Enrollment)")
        s = (enr_df.groupby("state")["total_enrolment"].sum()
             .reset_index().sort_values("total_enrolment", ascending=False).head(10))
        st.plotly_chart(px.bar(s, x="state", y="total_enrolment"), use_container_width=True)

    # Trend
    if "date" in enr_df.columns and "total_enrolment" in enr_df.columns:
        st.subheader("Trend")
        t = enr_df.groupby("date", dropna=False)["total_enrolment"].sum().reset_index().sort_values("date")
        st.plotly_chart(px.line(t, x="date", y="total_enrolment", markers=True), use_container_width=True)

    # District leaderboard
    if "district" in enr_df.columns and "total_enrolment" in enr_df.columns:
        st.subheader("Top Districts (Enrollment)")
        d = (enr_df.groupby("district")["total_enrolment"].sum()
             .reset_index().sort_values("total_enrolment", ascending=False).head(20))
        st.plotly_chart(px.bar(d, x="district", y="total_enrolment"), use_container_width=True)

    with st.expander("Raw data (filtered)"):
        st.dataframe(enr_df, use_container_width=True)

def render_demographic(demo_df):
    st.header("2) Demographic")

    total = demo_df["total_enrolment"].fillna(0).sum() if "total_enrolment" in demo_df.columns else 0
    fake_sum = pd.to_numeric(demo_df["is_fake"], errors="coerce").fillna(0).sum() if "is_fake" in demo_df.columns else 0

    c1, c2 = st.columns(2)
    c1.metric("Total Demographic Enrolment", f"{int(total):,}")
    c2.metric("Fake flagged (sum)", f"{int(fake_sum):,}")

    # Top 10 States
    if "state" in demo_df.columns and "total_enrolment" in demo_df.columns:
        st.subheader("Top 10 States (Demographic)")
        s = (demo_df.groupby("state")["total_enrolment"].sum()
             .reset_index().sort_values("total_enrolment", ascending=False).head(10))
        st.plotly_chart(px.bar(s, x="state", y="total_enrolment"), use_container_width=True)

    # Trend
    if "date" in demo_df.columns and "total_enrolment" in demo_df.columns:
        st.subheader("Trend")
        t = demo_df.groupby("date", dropna=False)["total_enrolment"].sum().reset_index().sort_values("date")
        st.plotly_chart(px.line(t, x="date", y="total_enrolment", markers=True), use_container_width=True)

    # District leaderboard
    if "district" in demo_df.columns and "total_enrolment" in demo_df.columns:
        st.subheader("Top Districts (Demographic)")
        d = (demo_df.groupby("district")["total_enrolment"].sum()
             .reset_index().sort_values("total_enrolment", ascending=False).head(20))
        st.plotly_chart(px.bar(d, x="district", y="total_enrolment"), use_container_width=True)

    with st.expander("Raw data (filtered)"):
        st.dataframe(demo_df, use_container_width=True)

def render_biometrics(bio_df, district_col):
    st.header("3) Biometrics")

    total_bio = bio_df["total_bio"].fillna(0).sum() if "total_bio" in bio_df.columns else 0
    st.metric("Total Biometrics", f"{int(total_bio):,}")

    # Top 10 States
    if "state" in bio_df.columns and "total_bio" in bio_df.columns:
        st.subheader("Top 10 States (Biometrics)")
        s = (bio_df.groupby("state")["total_bio"].sum()
             .reset_index().sort_values("total_bio", ascending=False).head(10))
        st.plotly_chart(px.bar(s, x="state", y="total_bio"), use_container_width=True)

    # Trend
    if "date" in bio_df.columns and "total_bio" in bio_df.columns:
        st.subheader("Trend")
        t = bio_df.groupby("date", dropna=False)["total_bio"].sum().reset_index().sort_values("date")
        st.plotly_chart(px.line(t, x="date", y="total_bio", markers=True), use_container_width=True)

    # District leaderboard
    if district_col in bio_df.columns and "total_bio" in bio_df.columns:
        st.subheader("Top Districts (Biometrics)")
        d = (bio_df.groupby(district_col)["total_bio"].sum()
             .reset_index().sort_values("total_bio", ascending=False).head(20))
        st.plotly_chart(px.bar(d, x=district_col, y="total_bio"), use_container_width=True)

    with st.expander("Raw data (filtered)"):
        st.dataframe(bio_df, use_container_width=True)

def render_recommendations(enr_df, demo_df, bio_df):
    st.header("4) Recommendations")
    st.caption("Auto insights based on the selected filters in this section.")

    join_cols = [c for c in ["date", "state", "pincode"] if c in enr_df.columns and c in demo_df.columns and c in bio_df.columns]
    if not join_cols:
        st.info("Not enough common columns to combine for recommendations (need date/state/pincode).")
        return

    enr_agg = enr_df.groupby(join_cols, dropna=False)["total_enrolment"].sum().reset_index().rename(columns={"total_enrolment": "enr_total"})
    demo_agg = demo_df.groupby(join_cols, dropna=False)["total_enrolment"].sum().reset_index().rename(columns={"total_enrolment": "demo_total"})
    bio_agg = bio_df.groupby(join_cols, dropna=False)["total_bio"].sum().reset_index().rename(columns={"total_bio": "bio_total"})

    merged = enr_agg.merge(demo_agg, on=join_cols, how="outer").merge(bio_agg, on=join_cols, how="outer")
    merged["bio_coverage"] = np.where(merged["enr_total"].fillna(0) > 0, merged["bio_total"].fillna(0) / merged["enr_total"].fillna(0), np.nan)
    merged["gap_demo_minus_enr"] = merged["demo_total"].fillna(0) - merged["enr_total"].fillna(0)

    low_cov = merged.sort_values("bio_coverage", ascending=True).head(10)
    high_gap = merged.sort_values("gap_demo_minus_enr", ascending=False).head(10)

    st.subheader("Top areas needing biometrics push (lowest bio coverage)")
    st.dataframe(low_cov, use_container_width=True)

    st.subheader("Top areas where Demographic > Enrollment gap is high")
    st.dataframe(high_gap, use_container_width=True)

def render_combined_insights(enr_df, demo_df, bio_df):
    st.header("5) Combined Insights")
    st.caption("Compare Enrollment vs Demographic vs Biometrics trends + pie charts.")

    if not ("date" in enr_df.columns and "date" in demo_df.columns and "date" in bio_df.columns):
        st.info("Combined insights needs a shared 'date' column.")
        return

    # Trend line
    enr_t = enr_df.groupby("date", dropna=False)["total_enrolment"].sum().reset_index().rename(columns={"total_enrolment": "Enrollment"})
    demo_t = demo_df.groupby("date", dropna=False)["total_enrolment"].sum().reset_index().rename(columns={"total_enrolment": "Demographic"})
    bio_t = bio_df.groupby("date", dropna=False)["total_bio"].sum().reset_index().rename(columns={"total_bio": "Biometrics"})
    trend = enr_t.merge(demo_t, on="date", how="outer").merge(bio_t, on="date", how="outer").sort_values("date")

    st.plotly_chart(px.line(trend, x="date", y=["Enrollment", "Demographic", "Biometrics"], markers=True),
                    use_container_width=True)

    # Totals
    total_enr = float(enr_df["total_enrolment"].fillna(0).sum()) if "total_enrolment" in enr_df.columns else 0.0
    total_demo = float(demo_df["total_enrolment"].fillna(0).sum()) if "total_enrolment" in demo_df.columns else 0.0
    total_bio = float(bio_df["total_bio"].fillna(0).sum()) if "total_bio" in bio_df.columns else 0.0
    cov = (total_bio / total_enr) if total_enr else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Enrollment", f"{int(total_enr):,}")
    c2.metric("Total Demographic", f"{int(total_demo):,}")
    c3.metric("Total Biometrics", f"{int(total_bio):,}")
    st.metric("Overall Bio Coverage (bio/enrollment)", f"{cov:.2%}" if pd.notna(cov) else "—")

    # Pie: totals composition
    st.subheader("Pie: Overall Composition (Totals)")
    share_df = pd.DataFrame({"Metric": ["Enrollment", "Demographic", "Biometrics"], "Total": [total_enr, total_demo, total_bio]})
    st.plotly_chart(px.pie(share_df, names="Metric", values="Total"), use_container_width=True)

    # Pies: age mix
    st.subheader("Pie: Age Composition")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Enrollment age mix**")
        if all(c in enr_df.columns for c in ["age_0_5", "age_5_17", "age_18_greater"]):
            age_enr = pd.DataFrame({
                "Age Bucket": ["0-5", "5-17", "18+"],
                "Count": [
                    float(enr_df["age_0_5"].fillna(0).sum()),
                    float(enr_df["age_5_17"].fillna(0).sum()),
                    float(enr_df["age_18_greater"].fillna(0).sum()),
                ]
            })
            st.plotly_chart(px.pie(age_enr, names="Age Bucket", values="Count"), use_container_width=True)
        else:
            st.info("Enrollment age columns not found.")

    with col2:
        st.markdown("**Demographic age mix**")
        if all(c in demo_df.columns for c in ["demo_age_5_17", "demo_age_17_"]):
            age_demo = pd.DataFrame({
                "Age Bucket": ["5-17", "17+"],
                "Count": [
                    float(demo_df["demo_age_5_17"].fillna(0).sum()),
                    float(demo_df["demo_age_17_"].fillna(0).sum()),
                ]
            })
            st.plotly_chart(px.pie(age_demo, names="Age Bucket", values="Count"), use_container_width=True)
        else:
            st.info("Demographic age columns not found.")

    with col3:
        st.markdown("**Biometrics age mix**")
        if all(c in bio_df.columns for c in ["bio_age_5_17", "bio_age_17_"]):
            age_bio = pd.DataFrame({
                "Age Bucket": ["5-17", "17+"],
                "Count": [
                    float(bio_df["bio_age_5_17"].fillna(0).sum()),
                    float(bio_df["bio_age_17_"].fillna(0).sum()),
                ]
            })
            st.plotly_chart(px.pie(age_bio, names="Age Bucket", values="Count"), use_container_width=True)
        else:
            st.info("Biometrics age columns not found.")

# -----------------------------
# Main
# -----------------------------
st.title("🪪 Aadhaar Dashboard")

enrollment_df, demographic_df, biometrics_df, meta = load_data_auto()
if meta["mode"] == "missing":
    enrollment_df, demographic_df, biometrics_df = read_uploaded_or_stop()
else:
    enr_path, demo_path, bio_path = meta["paths"]
    st.caption(
        f"Loaded local files ✅  Enrollment: {os.path.basename(enr_path)} | "
        f"Demographic: {os.path.basename(demo_path)} | Biometrics: {os.path.basename(bio_path)}"
    )

# Clean
enrollment_df = clean_enrollment(enrollment_df)
demographic_df = clean_demographic(demographic_df)
biometrics_df = clean_biometrics(biometrics_df)

bio_district_col = "district_norm" if "district_norm" in biometrics_df.columns else "district"

# Sidebar navigation (5 sections)
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["1) Enrollment", "2) Demographic", "3) Biometrics", "4) Recommendations", "5) Combined Insights"],
    key="nav_section"
)

st.sidebar.divider()
st.sidebar.caption("Filters are section-specific (each dataset uses its own State/District reference).")

if section == "1) Enrollment":
    st.sidebar.subheader("Enrollment Filters")
    enr_f = date_filter(enrollment_df, "date", "Enrollment Date range", key="enr_date")
    enr_f, _, _ = state_district_filter(enr_f, "state", "district", "Enrollment")
    render_enrollment(enr_f)

elif section == "2) Demographic":
    st.sidebar.subheader("Demographic Filters")
    demo_f = date_filter(demographic_df, "date", "Demographic Date range", key="demo_date")
    demo_f, _, _ = state_district_filter(demo_f, "state", "district", "Demographic")
    render_demographic(demo_f)

elif section == "3) Biometrics":
    st.sidebar.subheader("Biometrics Filters")
    bio_f = date_filter(biometrics_df, "date", "Biometrics Date range", key="bio_date")
    bio_f, _, _ = state_district_filter(bio_f, "state", bio_district_col, "Biometrics")
    render_biometrics(bio_f, bio_district_col)

elif section == "4) Recommendations":
    st.sidebar.subheader("Recommendations Filters")
    lens = st.sidebar.selectbox("Filter lens", ["Enrollment", "Demographic", "Biometrics"], key="rec_lens")

    enr_f = date_filter(enrollment_df, "date", "Date range", key="rec_date_enr")
    demo_f = date_filter(demographic_df, "date", "Date range", key="rec_date_demo")
    bio_f = date_filter(biometrics_df, "date", "Date range", key="rec_date_bio")

    if lens == "Enrollment":
        enr_f, st_sel, _ = state_district_filter(enr_f, "state", "district", "RecEnrollment")
        if st_sel != "All":
            demo_f = demo_f[demo_f["state"] == st_sel]
            bio_f = bio_f[bio_f["state"] == st_sel]
    elif lens == "Demographic":
        demo_f, st_sel, _ = state_district_filter(demo_f, "state", "district", "RecDemographic")
        if st_sel != "All":
            enr_f = enr_f[enr_f["state"] == st_sel]
            bio_f = bio_f[bio_f["state"] == st_sel]
    else:
        bio_f, st_sel, _ = state_district_filter(bio_f, "state", bio_district_col, "RecBiometrics")
        if st_sel != "All":
            enr_f = enr_f[enr_f["state"] == st_sel]
            demo_f = demo_f[demo_f["state"] == st_sel]

    render_recommendations(enr_f, demo_f, bio_f)

else:
    st.sidebar.subheader("Combined Insights Filters")
    enr_f = date_filter(enrollment_df, "date", "Date range", key="comb_date_enr")
    demo_f = date_filter(demographic_df, "date", "Date range", key="comb_date_demo")
    bio_f = date_filter(biometrics_df, "date", "Date range", key="comb_date_bio")

    common_states = sorted(
        set(enr_f["state"].dropna().unique())
        .intersection(set(demo_f["state"].dropna().unique()))
        .intersection(set(bio_f["state"].dropna().unique()))
    )
    st_sel = st.sidebar.selectbox("State (common)", ["All"] + common_states, key="comb_state")
    if st_sel != "All":
        enr_f = enr_f[enr_f["state"] == st_sel]
        demo_f = demo_f[demo_f["state"] == st_sel]
        bio_f = bio_f[bio_f["state"] == st_sel]

    render_combined_insights(enr_f, demo_f, bio_f)

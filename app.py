import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Unemployment Analysis",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.main {
    background-color: #F5F7FA;
}
h1 {
    color: #17365D;
}
h2 {
    color: #1F4E79;
}
.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("📊 Unemployment Analysis in India")

st.markdown("""
### Data Analysis and COVID-19 Impact Study

This dashboard explores unemployment trends across Indian regions,
compares urban and rural areas, and examines changes during the COVID-19 period.
""")

st.divider()

# -----------------------------------
# DATA LOADING
# -----------------------------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    required_columns = [
        "Region",
        "Date",
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)",
        "Area"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean text columns
    for col in ["Region", "Area"]:
        df[col] = df[col].astype("string").str.strip()

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid records
    df = df.dropna(
        subset=["Date", "Region", "Area",
                "Estimated Unemployment Rate (%)"]
    )

    df = df.drop_duplicates()

    # Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month Name"] = df["Date"].dt.strftime("%B")

    # COVID classification
    df["COVID Period"] = np.select(
        [
            df["Date"] < pd.Timestamp("2020-03-01"),
            df["Date"] <= pd.Timestamp("2021-12-31")
        ],
        [
            "Before COVID",
            "During COVID"
        ],
        default="After COVID"
    )

    return df


# -----------------------------------
# FILE UPLOAD
# -----------------------------------

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Unemployment CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Unable to load dataset: {e}")
        st.stop()

else:
    try:
        df = load_data("Unemployment in India.csv")
    except FileNotFoundError:
        st.warning(
            "Please upload your Unemployment in India.csv dataset "
            "using the sidebar."
        )
        st.stop()
    except Exception as e:
        st.error(f"Unable to load dataset: {e}")
        st.stop()


if df.empty:
    st.error("No valid data available after cleaning.")
    st.stop()


# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------

st.sidebar.divider()
st.sidebar.header("🔍 Dashboard Filters")

regions = sorted(df["Region"].dropna().unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

areas = sorted(df["Area"].dropna().unique())

selected_areas = st.sidebar.multiselect(
    "Select Area",
    areas,
    default=areas
)

years = sorted(df["Year"].dropna().unique())

selected_years = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

filtered_df = df[
    (df["Region"].isin(selected_regions)) &
    (df["Area"].isin(selected_areas)) &
    (df["Year"].isin(selected_years))
].copy()

if filtered_df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()


# -----------------------------------
# KPI METRICS
# -----------------------------------

st.header("📌 Key Performance Indicators")

avg_unemployment = filtered_df[
    "Estimated Unemployment Rate (%)"
].mean()

max_unemployment = filtered_df[
    "Estimated Unemployment Rate (%)"
].max()

min_unemployment = filtered_df[
    "Estimated Unemployment Rate (%)"
].min()

avg_employment = filtered_df["Estimated Employed"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Unemployment",
    f"{avg_unemployment:.2f}%"
)

col2.metric(
    "Highest Unemployment",
    f"{max_unemployment:.2f}%"
)

col3.metric(
    "Lowest Unemployment",
    f"{min_unemployment:.2f}%"
)

col4.metric(
    "Average Employed",
    f"{avg_employment:,.0f}"
)

st.divider()


# -----------------------------------
# DATASET OVERVIEW
# -----------------------------------

st.header("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(filtered_df))

col2.metric(
    "Total Regions",
    filtered_df["Region"].nunique()
)

col3.metric(
    "Years Covered",
    filtered_df["Year"].nunique()
)

with st.expander("View Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("View Missing Values"):
    st.dataframe(
        filtered_df.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )


# -----------------------------------
# UNEMPLOYMENT DISTRIBUTION
# -----------------------------------

st.header("📈 Unemployment Rate Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    filtered_df["Estimated Unemployment Rate (%)"].dropna(),
    bins=25,
    kde=True,
    ax=ax
)

ax.set_title("Distribution of Unemployment Rate")
ax.set_xlabel("Unemployment Rate (%)")
ax.set_ylabel("Number of Observations")

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# REGION-WISE ANALYSIS
# -----------------------------------

st.header("🌍 Region-wise Unemployment Analysis")

region_data = (
    filtered_df.groupby("Region")[
        "Estimated Unemployment Rate (%)"
    ]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(12, 6))

region_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Unemployment Rate by Region")
ax.set_xlabel("Region")
ax.set_ylabel("Average Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=75)

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# URBAN VS RURAL ANALYSIS
# -----------------------------------

st.header("🏙️ Urban vs Rural Analysis")

area_data = (
    filtered_df.groupby("Area")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

fig, ax = plt.subplots(figsize=(8, 5))

area_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Unemployment Rate: Urban vs Rural")
ax.set_xlabel("Area")
ax.set_ylabel("Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# TIME SERIES ANALYSIS
# -----------------------------------

st.header("📅 Unemployment Trend Over Time")

trend_data = (
    filtered_df.groupby("Date")[
        "Estimated Unemployment Rate (%)"
    ].mean()
    .reset_index()
    .sort_values("Date")
)

fig, ax = plt.subplots(figsize=(12, 5))

sns.lineplot(
    data=trend_data,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    ax=ax
)

ax.set_title("Unemployment Rate Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# COVID-19 IMPACT ANALYSIS
# -----------------------------------

st.header("🦠 COVID-19 Impact Analysis")

covid_data = (
    filtered_df.groupby("COVID Period")[
        "Estimated Unemployment Rate (%)"
    ].mean()
    .reindex(["Before COVID", "During COVID", "After COVID"])
    .dropna()
)

fig, ax = plt.subplots(figsize=(9, 5))

covid_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Unemployment Rate Before, During and After COVID-19")
ax.set_xlabel("COVID Period")
ax.set_ylabel("Average Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)

if "Before COVID" in covid_data.index and "During COVID" in covid_data.index:

    difference = (
        covid_data["During COVID"] -
        covid_data["Before COVID"]
    )

    st.info(
        f"The average unemployment rate changed by "
        f"{difference:.2f} percentage points during COVID "
        f"compared with before COVID in the selected data."
    )


# -----------------------------------
# MONTHLY ANALYSIS
# -----------------------------------

st.header("📆 Monthly Unemployment Analysis")

monthly_data = (
    filtered_df.groupby("Month")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

fig, ax = plt.subplots(figsize=(10, 5))

monthly_data.plot(
    marker="o",
    ax=ax
)

ax.set_title("Average Unemployment Rate by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xticks(range(1, 13))

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# YEARLY ANALYSIS
# -----------------------------------

st.header("📊 Yearly Unemployment Analysis")

yearly_data = (
    filtered_df.groupby("Year")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

fig, ax = plt.subplots(figsize=(10, 5))

yearly_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Unemployment Rate by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)


# -----------------------------------
# CORRELATION ANALYSIS
# -----------------------------------

st.header("🔗 Correlation Analysis")

numeric_data = filtered_df[
    [
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)"
    ]
].dropna()

if len(numeric_data) >= 2:

    correlation = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Between Numerical Variables")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Not enough numerical data for correlation analysis.")


# -----------------------------------
# TOP 10 REGIONS
# -----------------------------------

st.header("🏆 Top 10 Regions by Unemployment Rate")

top_regions = (
    filtered_df.groupby("Region")[
        "Estimated Unemployment Rate (%)"
    ].mean()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(
    top_regions.reset_index().rename(
        columns={
            "Region": "Region",
            "Estimated Unemployment Rate (%)": "Average Unemployment Rate (%)"
        }
    ),
    use_container_width=True
)


# -----------------------------------
# AUTOMATIC INSIGHTS
# -----------------------------------

st.header("💡 Key Insights")

highest_region = region_data.idxmax()
lowest_region = region_data.idxmin()

highest_area = area_data.idxmax()
lowest_area = area_data.idxmin()

st.success(
    f"""
    **1. Highest Unemployment Region:** {highest_region}

    **2. Lowest Unemployment Region:** {lowest_region}

    **3. Highest Unemployment Area:** {highest_area}

    **4. Lowest Unemployment Area:** {lowest_area}

    **5. Average Unemployment Rate:** {avg_unemployment:.2f}%
    """
)


# -----------------------------------
# DOWNLOAD CLEANED DATA
# -----------------------------------

st.header("⬇️ Download Cleaned Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV File",
    data=csv,
    file_name="cleaned_unemployment_data.csv",
    mime="text/csv"
)


# -----------------------------------
# CONCLUSION
# -----------------------------------

st.divider()

st.header("📝 Conclusion")

st.write("""
This project analyses unemployment trends in India using Python.

The dashboard provides insights into:

- Regional unemployment differences.
- Urban and rural unemployment patterns.
- Changes in unemployment over time.
- Unemployment patterns during COVID-19.
- Monthly and yearly unemployment trends.

These findings can support further research into employment
conditions and economic policy.

Note: These are descriptive comparisons. They do not independently
prove that COVID-19 caused changes in unemployment.
""")

st.caption(
    "Developed using Python, Pandas, NumPy, Matplotlib, Seaborn and Streamlit."
)
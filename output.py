
# Streamlit app for exploratory analysis of an airline flights dataset
# Checklist of actions the app will perform:
# - Parse and validate the provided metadata JSON embedded in the app.
# - Load a CSV dataset (default file: sample_airlines_flights_data.csv) or accept user upload.
# - Display dataset general info: rows/cols, dtypes, missing values, unique counts and sample rows.
# - Visualize the correlation matrix as a heatmap if present in metadata or computed from the data.
# - Gracefully handle missing/absent metadata fields and inform the user.
# - Provide additional visualizations: price distribution, price by airline/source/destination, scatter plots, stops counts.

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import json
from pathlib import Path
from io import StringIO

st.set_page_config(layout="wide", page_title="Airlines Flights Data Explorer")

# ----------------------------
# Embedded metadata (from user-provided JSON)
# ----------------------------
# This dictionary is included so the app can show the original metadata even if the CSV file is unavailable.
metadata = {
    "file_info": {"filename": "sample_airlines_flights_data.csv", "format": "csv", "size_bytes": 83086},
    "shape": {"rows": 1000, "columns": 12},
    "dtypes": {
        "index": "int64", "airline": "object", "flight": "object", "source_city": "object",
        "departure_time": "object", "stops": "object", "arrival_time": "object",
        "destination_city": "object", "class": "object", "duration": "float64",
        "days_left": "int64", "price": "int64"
    },
    "missing": {"index": 0, "airline": 0, "flight": 0, "source_city": 0, "departure_time": 0,
                "stops": 0, "arrival_time": 0, "destination_city": 0, "class": 0,
                "duration": 0, "days_left": 0, "price": 0},
    "nunique": {"index": 1000, "airline": 6, "flight": 453, "source_city": 6, "departure_time": 6,
                "stops": 3, "arrival_time": 6, "destination_city": 6, "class": 2,
                "duration": 291, "days_left": 49, "price": 679},
    "describe": {
        "index": {"count": 1000, "unique": None, "top": None, "freq": None, "mean": 148339.967,
                  "std": 86920.48014714899, "min": 424, "25%": 70585.75, "50%": 148267,
                  "75%": 220863.25, "max": 299515},
        "airline": {"count": 1000, "unique": 6, "top": "Vistara", "freq": 417, "mean": None,
                    "std": None, "min": None, "25%": None, "50%": None, "75%": None, "max": None},
        "flight": {"count": 1000, "unique": 453, "top": "UK-870", "freq": 10},
        "source_city": {"count": 1000, "unique": 6, "top": "Delhi", "freq": 213},
        "departure_time": {"count": 1000, "unique": 6, "top": "Evening", "freq": 227},
        "stops": {"count": 1000, "unique": 3, "top": "one", "freq": 814},
        "arrival_time": {"count": 1000, "unique": 6, "top": "Night", "freq": 314},
        "destination_city": {"count": 1000, "unique": 6, "top": "Mumbai", "freq": 194},
        "class": {"count": 1000, "unique": 2, "top": "Economy", "freq": 695},
        "duration": {"count": 1000, "mean": 12.11394, "std": 7.455224439803797, "min": 1,
                     "25%": 6.58, "50%": 11.04, "75%": 16, "max": 39.92},
        "days_left": {"count": 1000, "mean": 26.696, "std": 13.54507486332405, "min": 1,
                      "25%": 16, "50%": 27, "75%": 38, "max": 49},
        "price": {"count": 1000, "mean": 20465.645, "std": 22880.623103591028, "min": 1105,
                  "25%": 4672, "50%": 6834, "75%": 41289.25, "max": 114705}
    },
    "sample": {
        "index": {"521": 220897, "737": 51944, "740": 23823},
        "airline": {"521": "Vistara", "737": "Vistara", "740": "Vistara"},
        "flight": {"521": "UK-927", "737": "UK-853", "740": "UK-981"},
        "source_city": {"521": "Delhi", "737": "Mumbai", "740": "Delhi"},
        "departure_time": {"521": "Morning", "737": "Afternoon", "740": "Night"},
        "stops": {"521": "one", "737": "one", "740": "one"},
        "arrival_time": {"521": "Night", "737": "Night", "740": "Evening"},
        "destination_city": {"521": "Hyderabad", "737": "Delhi", "740": "Kolkata"},
        "class": {"521": "Business", "737": "Economy", "740": "Economy"},
        "duration": {"521": 11.42, "737": 6.42, "740": 22.25},
        "days_left": {"521": 33, "737": 45, "740": 22},
        "price": {"521": 46097, "737": 6122, "740": 7070}
    },
    "correlation": {
        "index": {"index": 1, "duration": 0.20883476062243114, "days_left": -0.004055823503756591, "price": 0.7558334466281115},
        "duration": {"index": 0.20883476062243114, "duration": 1, "days_left": -0.054543516780054264, "price": 0.26253757669243377},
        "days_left": {"index": -0.004055823503756591, "duration": -0.054543516780054264, "days_left": 1, "price": -0.09883219345182705},
        "price": {"index": 0.7558334466281115, "duration": 0.26253757669243377, "days_left": -0.09883219345182705, "price": 1}
    },
    "unique_values": {
        "airline": ["Air_India", "Vistara", "SpiceJet", "GO_FIRST", "Indigo", "AirAsia"],
        "source_city": ["Kolkata", "Bangalore", "Delhi", "Hyderabad", "Mumbai", "Chennai"],
        "departure_time": ["Morning", "Afternoon", "Early_Morning", "Evening", "Night", "Late_Night"],
        "stops": ["one", "two_or_more", "zero"],
        "arrival_time": ["Night", "Morning", "Evening", "Afternoon", "Early_Morning", "Late_Night"],
        "destination_city": ["Mumbai", "Chennai", "Kolkata", "Hyderabad", "Delhi", "Bangalore"],
        "class": ["Economy", "Business"]
    },
    "description": "this is an airline flights dataset"
}

# ----------------------------
# Sidebar: file upload or use default
# ----------------------------
st.sidebar.header("Data input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file (optional). If not provided, the app will try to load the default filename shown below.", type=["csv"])
st.sidebar.write("Default file name from metadata:")
st.sidebar.write(metadata.get("file_info", {}).get("filename", "N/A"))

# ----------------------------
# Data loading (with caching)
# ----------------------------
@st.cache_data
def load_dataframe_from_file(file_path: str):
    """
    Try to load a CSV into a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

@st.cache_data
def load_dataframe_from_buffer(buffer):
    df = pd.read_csv(buffer)
    return df

df = None
load_error = None

if uploaded_file is not None:
    try:
        df = load_dataframe_from_buffer(uploaded_file)
        st.sidebar.success("Loaded uploaded CSV file.")
    except Exception as e:
        load_error = f"Error loading uploaded file: {e}"
        st.sidebar.error(load_error)
else:
    default_path = Path(metadata.get("file_info", {}).get("filename", "sample_airlines_flights_data.csv"))
    if default_path.exists():
        try:
            df = load_dataframe_from_file(str(default_path))
            st.sidebar.success(f"Loaded default CSV: {default_path.name}")
        except Exception as e:
            load_error = f"Error loading default CSV: {e}"
            st.sidebar.error(load_error)
    else:
        load_error = f"Default file '{default_path.name}' not found in the working directory."
        st.sidebar.warning(load_error)

# ----------------------------
# Main app header and dataset description
# ----------------------------
st.title("Airlines Flights Data Explorer")
st.markdown("Quick exploratory dashboard for an airlines flights dataset. The app uses both the provided metadata (from JSON) and the actual CSV if available.")

# Show embedded dataset description
st.subheader("Dataset description (from metadata)")
st.write(metadata.get("description", "No description provided in metadata."))

# ----------------------------
# Show metadata summary
# ----------------------------
with st.expander("Show provided metadata summary (from JSON)"):
    st.write("File info:")
    st.json(metadata.get("file_info", {}))
    st.write("Shape (rows, columns):")
    st.write(metadata.get("shape", {}))
    st.write("Dtypes (as provided):")
    st.json(metadata.get("dtypes", {}))
    st.write("Missing values per column (as provided):")
    st.json(metadata.get("missing", {}))
    st.write("Number of unique values per column (as provided):")
    st.json(metadata.get("nunique", {}))

# ----------------------------
# If dataframe loaded, display computed info and visualizations
# ----------------------------
if df is None:
    st.error("No dataset loaded. Please upload a CSV or place the default file in the app directory.")
    # Even if CSV not loaded, we can still display some metadata sample rows provided in JSON
    st.subheader("Sample rows (from provided metadata)")
    sample_meta = metadata.get("sample", None)
    if sample_meta:
        # Convert the sample nested dict to a small DataFrame for display
        try:
            # sample_meta columns map to dicts of index:value; we'll reconstruct three rows based on keys inside each column dict
            col_names = list(sample_meta.keys())
            # Determine row keys (e.g., "521","737","740")
            example_keys = []
            for col in col_names:
                inner = sample_meta[col]
                if isinstance(inner, dict):
                    example_keys = list(inner.keys())
                    break
            rows = []
            for rk in example_keys:
                row = {}
                for col in col_names:
                    val = sample_meta[col].get(rk, None) if isinstance(sample_meta[col], dict) else None
                    row[col] = val
                rows.append(row)
            sample_df = pd.DataFrame(rows)
            st.dataframe(sample_df)
        except Exception:
            st.write("Could not reconstruct sample rows from metadata.")
    st.stop()

# If we get here, df is loaded
st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns (computed).")

# ----------------------------
# Basic dataset info: shape, dtypes, missing values
# ----------------------------
st.subheader("Basic dataset information (computed)")

col1, col2 = st.columns([1,2])
with col1:
    st.markdown("**Shape**")
    st.write({"rows": df.shape[0], "columns": df.shape[1]})
    st.markdown("**Memory usage (approx)**")
    st.write(f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

with col2:
    st.markdown("**Data types**")
    # show dtypes as a dataframe
    dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=["dtype"])
    st.dataframe(dtypes_df)

st.markdown("**Missing values per column**")
missing_series = df.isna().sum()
st.dataframe(missing_series.astype(int).rename("missing_count"))

# Compare with provided metadata missing info
st.markdown("Provided metadata missing counts (if available):")
st.json(metadata.get("missing", {}))

# ----------------------------
# Show sample rows (both metadata and computed)
# ----------------------------
st.subheader("Sample rows")
sample_display = st.radio("Choose sample source:", ("Random sample from CSV", "Sample from metadata (JSON)"), index=0)
if sample_display == "Random sample from CSV":
    st.dataframe(df.sample(min(5, len(df))).reset_index(drop=True))
else:
    sample_meta = metadata.get("sample", None)
    if sample_meta:
        # try reconstruct sample, similar to earlier
        try:
            col_names = list(sample_meta.keys())
            example_keys = []
            for col in col_names:
                inner = sample_meta[col]
                if isinstance(inner, dict):
                    example_keys = list(inner.keys())
                    break
            rows = []
            for rk in example_keys:
                row = {}
                for col in col_names:
                    val = sample_meta[col].get(rk, None) if isinstance(sample_meta[col], dict) else None
                    row[col] = val
                rows.append(row)
            sample_df = pd.DataFrame(rows)
            st.dataframe(sample_df)
        except Exception:
            st.write("Could not display metadata sample.")
    else:
        st.write("No sample rows available in metadata.")

# ----------------------------
# Display describe() output
# ----------------------------
st.subheader("Statistical summary (.describe)")

# Show metadata describe if present
meta_describe = metadata.get("describe", None)
if meta_describe:
    st.markdown("Describe from metadata (JSON):")
    # Pretty display: convert metadata.describe into DataFrame when possible
    try:
        meta_desc_df = pd.DataFrame(meta_describe).T
        st.dataframe(meta_desc_df)
    except Exception:
        st.write(meta_describe)
else:
    st.write("No 'describe' information present in metadata.")

st.markdown("Computed describe from the loaded CSV:")
try:
    st.dataframe(df.describe(include='all').T)
except Exception as e:
    st.write(f"Error computing describe(): {e}")

# ----------------------------
# Unique values for categorical columns
# ----------------------------
st.subheader("Categorical overview / unique values")
# Use metadata unique_values if present
meta_unique = metadata.get("unique_values", None)
if meta_unique:
    st.markdown("Unique values from metadata (JSON):")
    st.json(meta_unique)
else:
    st.write("No unique values provided in metadata.")

# Complement with computed unique for object columns
st.markdown("Computed unique counts (for object / category columns):")
obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if obj_cols:
    unique_counts = {col: int(df[col].nunique(dropna=False)) for col in obj_cols}
    st.dataframe(pd.Series(unique_counts, name="unique_count"))
    # Optionally show unique samples for columns the user picks
    col_for_unique = st.selectbox("Show unique values for which categorical column?", options=["(none)"] + obj_cols, index=0)
    if col_for_unique != "(none)":
        st.write(f"Unique values for {col_for_unique}:")
        st.write(sorted(df[col_for_unique].dropna().unique().tolist()))
else:
    st.write("No categorical/object columns detected.")

# ----------------------------
# Correlation matrix handling and visualization
# ----------------------------
st.subheader("Correlation matrix")

# 1) Show correlation matrix from metadata if present
meta_corr = metadata.get("correlation", None)
if meta_corr:
    st.markdown("Correlation from metadata (JSON):")
    try:
        # Convert nested dict to DataFrame
        corr_meta_df = pd.DataFrame(meta_corr).T
        st.dataframe(corr_meta_df)
        # Plot heatmap for metadata correlation
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr_meta_df.astype(float), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation matrix (metadata)")
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Could not plot metadata correlation: {e}")
else:
    st.write("No correlation matrix present in metadata.")

# 2) Compute correlation from the actual numeric columns (if available)
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 2:
    st.markdown("Computed correlation from the loaded dataset (numeric columns):")
    corr_df = numeric_df.corr()
    st.dataframe(corr_df)
    # Heatmap with seaborn
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax2)
    ax2.set_title("Correlation matrix (computed)")
    st.pyplot(fig2)
    # Option to download correlation as CSV
    csv_corr = corr_df.to_csv(index=True)
    st.download_button("Download computed correlation CSV", data=csv_corr, file_name="computed_correlation.csv", mime="text/csv")
else:
    st.write("Not enough numeric columns to compute correlation.")

# ----------------------------
# Visualizations: distributions and relationships
# ----------------------------
st.subheader("Key visualizations")

# Price distribution
st.markdown("1) Price distribution")
fig_p = px.histogram(df, x="price", nbins=80, title="Price distribution", marginal="box", labels={"price":"Price"})
st.plotly_chart(fig_p, use_container_width=True)

# Price by airline (boxplot / violin)
if "airline" in df.columns:
    st.markdown("2) Price by airline (box / violin)")
    fig_air = px.box(df, x="airline", y="price", color="airline", points="outliers", title="Price distribution by airline")
    st.plotly_chart(fig_air, use_container_width=True)
else:
    st.write("Column 'airline' not present for airline-based plots.")

# Average price by source city and destination city
if "source_city" in df.columns and "destination_city" in df.columns:
    st.markdown("3) Average price by Source and Destination")
    avg_src = df.groupby("source_city")["price"].median().sort_values()
    avg_dest = df.groupby("destination_city")["price"].median().sort_values()
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(avg_src.rename("median_price"))
        st.write("Median price by source_city")
    with c2:
        st.bar_chart(avg_dest.rename("median_price"))
        st.write("Median price by destination_city")
else:
    st.write("Source/destination city columns not available for aggregated plots.")

# Scatter: Price vs Duration, colored by airline
if set(["price", "duration"]).issubset(df.columns):
    st.markdown("4) Price vs Duration (points sized by days_left if available, colored by airline if available)")
    size_arg = "days_left" if "days_left" in df.columns else None
    color_arg = "airline" if "airline" in df.columns else None
    fig_sc = px.scatter(df, x="duration", y="price", color=color_arg, size=size_arg,
                        hover_data=["source_city","destination_city","stops","class"], title="Price vs Duration")
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.write("Columns 'price' and 'duration' required for scatter plot are missing.")

# Relationship: days_left vs price (useful for sales/booking behavior)
if set(["days_left", "price"]).issubset(df.columns):
    st.markdown("5) Days left vs Price (average price by days left)")
    agg_days = df.groupby("days_left")["price"].median().reset_index()
    fig_days = px.line(agg_days, x="days_left", y="price", markers=True, title="Median price by days_left")
    st.plotly_chart(fig_days, use_container_width=True)
else:
    st.write("Columns 'days_left' and 'price' required for days-left analysis are missing.")

# Stops distribution
if "stops" in df.columns:
    st.markdown("6) Stops distribution")
    stops_count = df["stops"].value_counts().reset_index().rename(columns={"index":"stops","stops":"count"})
    fig_stops = px.pie(stops_count, names="stops", values="count", title="Proportion of flights by number of stops")
    st.plotly_chart(fig_stops, use_container_width=True)
else:
    st.write("Column 'stops' not available to plot stops distribution.")

# Departure time vs price (boxplot)
if "departure_time" in df.columns and "price" in df.columns:
    st.markdown("7) Price by departure time")
    fig_dep = px.box(df, x="departure_time", y="price", color="departure_time", title="Price by departure time")
    st.plotly_chart(fig_dep, use_container_width=True)
else:
    st.write("Columns 'departure_time' or 'price' missing for this visualization.")

# Pairwise scatter matrix for numeric columns (if not too many)
numeric_cols = numeric_df.columns.tolist()
if 1 < len(numeric_cols) <= 6:
    st.markdown("8) Pairwise scatter matrix (numeric features)")
    fig_matrix = px.scatter_matrix(df, dimensions=numeric_cols, color="airline" if "airline" in df.columns else None,
                                   title="Scatter matrix of numeric features")
    st.plotly_chart(fig_matrix, use_container_width=True)
elif len(numeric_cols) > 6:
    st.write("Too many numeric columns to show a pairwise scatter matrix; select a subset manually if desired.")
else:
    st.write("Not enough numeric columns for a scatter matrix.")

# ----------------------------
# Filtering and quick analysis tools
# ----------------------------
st.subheader("Interactive filtering and quick analysis")

# Provide filters for quick exploration
with st.form("filter_form"):
    st.write("Filter dataset and compute summaries")
    col_a = st.selectbox("Filter by Airline (optional)", options=["(all)"] + sorted(df["airline"].dropna().unique().tolist()) if "airline" in df.columns else ["(not available)"])
    col_src = st.selectbox("Filter by Source City (optional)", options=["(all)"] + sorted(df["source_city"].dropna().unique().tolist()) if "source_city" in df.columns else ["(not available)"])
    col_dest = st.selectbox("Filter by Destination City (optional)", options=["(all)"] + sorted(df["destination_city"].dropna().unique().tolist()) if "destination_city" in df.columns else ["(not available)"])
    max_price = int(df["price"].max()) if "price" in df.columns else 0
    price_range = st.slider("Max price (show flights with price <= )", min_value=0, max_value=max_price, value=max_price)
    submitted = st.form_submit_button("Apply filters")
if submitted:
    filt = df.copy()
    if "airline" in df.columns and col_a != "(all)":
        filt = filt[filt["airline"] == col_a]
    if "source_city" in df.columns and col_src != "(all)":
        filt = filt[filt["source_city"] == col_src]
    if "destination_city" in df.columns and col_dest != "(all)":
        filt = filt[filt["destination_city"] == col_dest]
    if "price" in df.columns:
        filt = filt[filt["price"] <= price_range]
    st.write(f"Filtered results: {len(filt)} rows")
    st.dataframe(filt.head(50))

    # Show quick aggregates
    st.write("Aggregates on filtered data:")
    agg_cols = {}
    if "price" in filt.columns:
        agg_cols["price_median"] = filt["price"].median()
        agg_cols["price_mean"] = filt["price"].mean()
    if "duration" in filt.columns:
        agg_cols["duration_mean"] = filt["duration"].mean()
    if "days_left" in filt.columns:
        agg_cols["days_left_mean"] = filt["days_left"].mean()
    st.json(agg_cols)

# ----------------------------
# Final notes and help
# ----------------------------
st.sidebar.markdown("## Notes / Troubleshooting")
st.sidebar.write("- If CSV fails to load, ensure the file is in CSV format and columns are as expected.")
st.sidebar.write("- The app displays both the provided metadata and computed metrics from the loaded CSV; discrepancies may indicate different file versions.")
st.sidebar.write("- You can download computed correlation matrix from the 'Correlation' section.")

st.markdown("---")
st.caption("App generated to explore an airline flights dataset. Uses metadata provided to guide displays and comparisons.")

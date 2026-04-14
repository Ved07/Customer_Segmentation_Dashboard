import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main {
        padding: 20px;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ==================== HEADER ====================
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1>🎯 Customer Segmentation Pipeline</h1>
    <h3>K-Means Clustering Dashboard</h3>
    <p style='font-size: 14px; color: #7f8c8d;'>Analyze, preprocess, and segment customers based on their behavior patterns</p>
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Data Upload",
    "📈 Data Insights",
    "🔧 Preprocessing",
    "🎯 Feature Selection",
    "🤖 Model Training",
    "🔮 Predictions & Performance"
])

# ==================== TAB 1: DATA UPLOAD ====================
with tab1:
    st.markdown("## 📤 Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload customer data for segmentation"
        )
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📊 Rows", len(st.session_state.data))
            with col_b:
                st.metric("📋 Columns", len(st.session_state.data.columns))
            with col_c:
                st.metric("🔍 Missing Values", st.session_state.data.isnull().sum().sum())
            
            st.markdown("### Preview of Data")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            st.markdown("### Dataset Information")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write("**Data Types:**")
                st.dataframe(st.session_state.data.dtypes, use_container_width=True)
            

    with col2:
        if st.session_state.data is not None:
            st.markdown("### 📌 Quick Stats")
            st.markdown(f"""
            - **Dataset Size:** {st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}
            - **Memory Usage:** {st.session_state.data.memory_usage(deep=True).sum() / 1024:.2f} KB
            - **Numeric Columns:** {st.session_state.data.select_dtypes(include=[np.number]).shape[1]}
            - **Categorical Columns:** {st.session_state.data.select_dtypes(include=['object']).shape[1]}
            """)
            
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.data = None
                st.rerun()

# ==================== TAB 2: DATA INSIGHTS ====================
with tab2:
    st.markdown("## 📈 Data Insights & Exploratory Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the Data Upload tab first")
    else:
        data = st.session_state.data
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.error("❌ No numeric columns found in the dataset")
        else:
            # Distribution plots
            st.markdown("### 📊 Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
                fig = px.histogram(
                    data,
                    x=selected_col,
                    nbins=30,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                selected_col_box = st.selectbox("Select column for box plot", numeric_cols, key="box_col")
                fig = px.box(
                    data,
                    y=selected_col_box,
                    title=f"Box Plot of {selected_col_box}",
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            
            # Scatter plots
            st.markdown("### 📍 Feature Relationships")
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
                col_y = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
                
                fig = px.scatter(
                    data,
                    x=col_x,
                    y=col_y,
                    title=f"{col_x} vs {col_y}",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
            
            # Missing values visualization
            st.markdown("### ❌ Missing Values Analysis")
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data[missing_data > 0].index,
                    y=missing_data[missing_data > 0].values,
                    title="Missing Values by Column",
                    labels={"x": "Column", "y": "Count"},
                    color_discrete_sequence=['#e74c3c']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

# ==================== TAB 3: DATA PREPROCESSING ====================
with tab3:
    st.markdown("## 🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the Data Upload tab first")
    else:
        data = st.session_state.data.copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🧹 Preprocessing Steps")
            
            # Missing values handling
            st.markdown("#### 1️⃣ Missing Values Treatment")
            missing_cols = data.columns[data.isnull().any()].tolist()
            
            if len(missing_cols) > 0:
                st.write(f"Columns with missing values: {missing_cols}")
                
                for col in missing_cols:
                    if data[col].dtype in [np.float64, np.int64]:
                        method = st.radio(
                            f"Select method for '{col}'",
                            ["Mean", "Median", "Drop Row"],
                            horizontal=True,
                            key=f"missing_{col}"
                        )
                        if method == "Mean":
                            data[col].fillna(data[col].mean(), inplace=True)
                        elif method == "Median":
                            data[col].fillna(data[col].median(), inplace=True)
                        else:
                            data.dropna(subset=[col], inplace=True)
                    else:
                        data[col].fillna(data[col].mode()[0], inplace=True)
                st.success("✅ Missing values handled!")
            else:
                st.info("✓ No missing values found")
            
            # Outlier detection and removal
            st.markdown("#### 2️⃣ Outlier Detection & Removal")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            outlier_threshold = st.slider("IQR Threshold (×)", 1.0, 3.0, 1.5, 0.1)

            initial_rows = len(data)

            # Create a mask (keep track of valid rows)
            mask = np.ones(len(data), dtype=bool)

            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                mask &= (data[col] >= lower_bound) & (data[col] <= upper_bound)

            # Apply filtering ONCE
            data = data[mask]

            outliers_removed = initial_rows - len(data)
            st.success(f"✅ Outliers removed: {outliers_removed} rows")
            
            # Remove duplicates
            st.markdown("#### 3️⃣ Duplicate Removal")
            initial_rows = len(data)
            data.drop_duplicates(inplace=True)
            duplicates_removed = initial_rows - len(data)
            st.success(f"✅ Duplicates removed: {duplicates_removed} rows")
            
            # Encode categorical variables
            st.markdown("#### 4️⃣ Categorical Encoding")
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

            if len(categorical_cols) > 0:
                st.write(f"Columns to encode: {categorical_cols}")
                
                data = pd.get_dummies(data, columns=categorical_cols)
                
                st.success("✅ One-Hot Encoding applied!")
            else:
                st.info("✓ No categorical columns to encode")
            
            st.session_state.processed_data = data.copy()
            # Preprocessed data summary
            st.markdown("### ✅ Preprocessed Data")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Rows", len(data))
            with col_b:
                st.metric("📋 Columns", len(data.columns))
            
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### 📌 Summary")
            st.write(f"""
            **Before Preprocessing:**
            - Rows: {len(st.session_state.data)}
            - Columns: {len(st.session_state.data.columns)}
            
            **After Preprocessing:**
            - Rows: {len(data)}
            - Columns: {len(data.columns)}
            """)

# ==================== TAB 4: FEATURE SELECTION ====================
with tab4:
    st.markdown("## 🎯 Feature Selection")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete preprocessing first")
    else:
        data = st.session_state.processed_data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Feature Information")
            
            # # Statistical information
            st.write("**Statistical Summary of Features:**")
            stats_df = data[numeric_cols].describe().T
            stats_df['CV'] = (stats_df['std'] / stats_df['mean']) * 100  # Coefficient of variation
            st.dataframe(stats_df, use_container_width=True)
            
            
            # Feature selection
            st.markdown("### ✅ Select Features for Clustering")
            
            selected_features = st.multiselect(
                "Choose features for your model",
                numeric_cols,
                default=numeric_cols,
                help="Select the features you want to use for clustering"
            )
            
            if selected_features:
                st.session_state.selected_features = selected_features
                st.success(f"✅ Selected {len(selected_features)} features for clustering")
            
        
        with col2:
            st.markdown("### 📌 Summary")
            st.write(f"""
            **Total Features:** {len(numeric_cols)}
            **Selected Features:** {len(st.session_state.selected_features) if st.session_state.selected_features else 0}
            
            **Data Shape:**
            - Shape: {data.shape}
            - Memory: {data.memory_usage(deep=True).sum() / 1024:.2f} KB
            """)

# ==================== TAB 5: MODEL TRAINING ====================
with tab5:
    st.markdown("## 🤖 K-Means Model Training")
    
    if st.session_state.selected_features is None:
        st.warning("⚠️ Please select features in Feature Selection tab first")
    else:
        data = st.session_state.processed_data[st.session_state.selected_features].copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### ⚙️ Model Configuration")
            
            # Data scaling
            st.markdown("#### 1️⃣ Data Scaling")
           # ALWAYS retrain scaler on selected features
            scaler = StandardScaler()

            # Store feature order used in training
            st.session_state.feature_order = data.columns.tolist()

            # Fit + transform
            scaled_data = scaler.fit_transform(data)

            # Save in session
            st.session_state.scaler = scaler
            st.session_state.scaled_data = scaled_data
            st.success("✅ Data scaled using StandardScaler")
            
            # Determine optimal number of clusters
            st.markdown("#### 2️⃣ Find Optimal Clusters")
            
            max_k = min(10, len(data) - 1)
            
            if st.button("🔍 Analyze Elbow Method", use_container_width=True):
                inertias = []
                silhouette_scores = []
                K_range = range(2, max_k + 1)
                
                progress_bar = st.progress(0)
                for i, k in enumerate(K_range):
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_temp.fit(scaled_data)
                    inertias.append(kmeans_temp.inertia_)
                    silhouette_scores.append(silhouette_score(scaled_data, kmeans_temp.labels_))
                    progress_bar.progress((i + 1) / len(K_range))
                
                # Plot elbow curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(K_range),
                    y=inertias,
                    mode='lines+markers',
                    name='Inertia',
                    marker=dict(size=8, color='#667eea')
                ))
                fig.update_layout(
                    title="Elbow Method",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Inertia",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot silhouette scores
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=list(K_range),
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='Silhouette Score',
                    marker=dict(size=8, color='#764ba2')
                ))
                fig2.update_layout(
                    title="Silhouette Score vs Number of Clusters",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Silhouette Score",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Select number of clusters
            st.markdown("#### 3️⃣ Select Number of Clusters")
            n_clusters = st.slider("Choose number of clusters (k)", 2, max_k, 3)
            
            # Train model
            st.markdown("#### 4️⃣ Train K-Means Model")
            if st.button("🚀 Train Model", use_container_width=True):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                predictions = kmeans.fit_predict(scaled_data)
                
                st.session_state.model = kmeans
                st.session_state.predictions = predictions
                
                # Calculate metrics
                silhouette_avg = silhouette_score(scaled_data, predictions)
                davies_bouldin = davies_bouldin_score(scaled_data, predictions)
                calinski_harabasz = calinski_harabasz_score(scaled_data, predictions)
                
                st.success("✅ Model trained successfully!")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("🎯 Silhouette Score", f"{silhouette_avg:.4f}")
                with col_m2:
                    st.metric("📊 Davies-Bouldin Index", f"{davies_bouldin:.4f}")
                with col_m3:
                    st.metric("🔍 Calinski-Harabasz Index", f"{calinski_harabasz:.2f}")
                
                # Visualize clusters using PCA
                st.markdown("#### 📍 Cluster Visualization (PCA)")
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)
                
                fig = go.Figure()
                for cluster in range(n_clusters):
                    cluster_points = pca_data[predictions == cluster]
                    fig.add_trace(go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=pca.transform(kmeans.cluster_centers_)[:, 0],
                    y=pca.transform(kmeans.cluster_centers_)[:, 1],
                    mode='markers',
                    name='Centroids',
                    marker=dict(size=20, symbol='star', color='red')
                ))
                
                fig.update_layout(
                    title=f"K-Means Clusters (k={n_clusters})",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
                    height=450,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster distribution
                st.markdown("#### 📈 Cluster Distribution")
                cluster_counts = pd.Series(predictions).value_counts().sort_index()
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={"x": "Cluster", "y": "Number of Customers"},
                    title="Customers per Cluster",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📌 Model Info")
            if st.session_state.model is not None:
                st.write(f"""
                **Model Status:** ✅ Trained
                
                **Configuration:**
                - Algorithm: K-Means
                - Clusters: {n_clusters}
                - Features: {len(st.session_state.selected_features)}
                - Data Points: {len(data)}
                
                **Feature List:**
                {', '.join(st.session_state.selected_features)}
                """)

# ==================== TAB 6: PREDICTIONS & PERFORMANCE ====================
with tab6:
    st.markdown("## 🔮 Predictions & Model Performance")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model in Model Training tab first")
    else:
        data = st.session_state.processed_data[st.session_state.selected_features].copy()
        predictions = st.session_state.predictions
        model = st.session_state.model
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Model Performance Metrics")
            
            # Calculate metrics
            silhouette_avg = silhouette_score(st.session_state.scaled_data, predictions)
            davies_bouldin = davies_bouldin_score(st.session_state.scaled_data, predictions)
            calinski_harabasz = calinski_harabasz_score(st.session_state.scaled_data, predictions)
            inertia = model.inertia_
            
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("🎯 Silhouette Score", f"{silhouette_avg:.4f}", 
                         help="Higher is better (range: -1 to 1)")
            with p2:
                st.metric("📊 Davies-Bouldin Index", f"{davies_bouldin:.4f}",
                         help="Lower is better")
            with p3:
                st.metric("🔍 Calinski-Harabasz Index", f"{calinski_harabasz:.2f}",
                         help="Higher is better")
            with p4:
                st.metric("💯 Inertia", f"{inertia:.2f}")
            
            # Cluster statistics
            st.markdown("### 📈 Cluster Statistics")
            cluster_df = pd.DataFrame({
                'Cluster': range(model.n_clusters),
                'Size': [np.sum(predictions == i) for i in range(model.n_clusters)],
                'Percentage': [(np.sum(predictions == i) / len(predictions) * 100) for i in range(model.n_clusters)]
            })
            
            fig = px.pie(
                cluster_df,
                values='Size',
                names=['Cluster ' + str(i) for i in cluster_df['Cluster']],
                title="Cluster Distribution",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            

            # Add predictions to original data
            st.markdown("### 🔮 Predictions")

            # Create copy of dataset
            result_df = data.copy()

            # Add cluster labels
            result_df['Predicted_Cluster'] = predictions

            # ======================
            # 🎯 SMART Segment Naming (Improved)
            # ======================

            cluster_names = {}

            centroids = model.cluster_centers_

            # Convert centroids to DataFrame for better readability
            centroid_df = pd.DataFrame(
                centroids,
                columns=st.session_state.feature_order
            )

            for i in range(len(centroid_df)):
                row = centroid_df.iloc[i]

                label = ""

                # Try to detect meaningful features
                income_cols = [col for col in row.index if "income" in col.lower()]
                spend_cols = [col for col in row.index if "spend" in col.lower()]

                # Default averages
                income = row[income_cols].mean() if income_cols else row.mean()
                spend = row[spend_cols].mean() if spend_cols else row.mean()

                # 🔥 Smart Rules
                if income > 0 and spend > 0:
                    label = "💎 High Income - High Spending"
                elif income > 0 and spend < 0:
                    label = "🤑 High Income - Low Spending"
                elif income < 0 and spend > 0:
                    label = "💸 Low Income - High Spending"
                else:
                    label = "⚖️ Low Income - Low Spending"

                cluster_names[i] = label

            # Map cluster → segment
            result_df["Segment"] = result_df["Predicted_Cluster"].map(cluster_names)

            # ======================
            # 📊 Display Results
            # ======================
            st.write(f"Total Customers: {len(result_df)}")

            # Show top rows
            st.dataframe(result_df.head(20), use_container_width=True)

            # ======================
            # 📈 Segment Distribution
            # ======================
            st.subheader("Segment Distribution")

            segment_counts = result_df["Segment"].value_counts()

            st.bar_chart(segment_counts)

            # ======================
            # 📥 Download Option
            # ======================
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="customer_segments.csv",
                mime="text/csv"
            )

            # ======================
            # 🧑‍💻 User Input Prediction (FINAL FIX)
            # ======================
            st.markdown("### 🧑‍💻 Predict for New Customer")

            input_data = {}

            # Use EXACT same feature order as training
            for feature in st.session_state.feature_order:
                input_data[feature] = st.number_input(
                    f"Enter value for {feature}",
                    value=float(data[feature].mean()),
                    key=f"input_{feature}"
                )

            if st.button("🔮 Predict Cluster for Input", use_container_width=True):

                # Create DataFrame
                input_df = pd.DataFrame([input_data])

                # 🔥 CRITICAL FIX
                input_df = input_df.reindex(columns=st.session_state.feature_order, fill_value=0)

                # Scale
                scaled_input = st.session_state.scaler.transform(input_df)

                # Predict
                predicted_cluster = st.session_state.model.predict(scaled_input)[0]

                # Segment naming
                # Use same logic for prediction
                center = st.session_state.model.cluster_centers_[predicted_cluster]
                row = pd.Series(center, index=st.session_state.feature_order)

                income_cols = [col for col in row.index if "income" in col.lower()]
                spend_cols = [col for col in row.index if "spend" in col.lower()]

                income = row[income_cols].mean() if income_cols else row.mean()
                spend = row[spend_cols].mean() if spend_cols else row.mean()

                if income > 0 and spend > 0:
                    segment = "💎 High Income - High Spending"
                elif income > 0 and spend < 0:
                    segment = "🤑 High Income - Low Spending"
                elif income < 0 and spend > 0:
                    segment = "💸 Low Income - High Spending"
                else:
                    segment = "⚖️ Low Income - Low Spending"
                # Output
                st.success(f"✅ Predicted Cluster: {predicted_cluster}")
                st.info(f"📌 Segment: {segment}")
                st.dataframe(input_df, use_container_width=True)
            
        
        with col2:
            st.markdown("### 📌 Summary")
            st.write(f"""
            **Model Status:** ✅ Trained & Ready
            
            **Clustering Results:**
            - Total Clusters: {model.n_clusters}
            - Total Points: {len(data)}
            
            **Features Used:**
            {len(st.session_state.selected_features)}
            
            **Performance:**
            - Silhouette: {silhouette_avg:.4f}
            - Davies-Bouldin: {davies_bouldin:.4f}
            - Calinski-Harabasz: {calinski_harabasz:.2f}
            
            **Interpretation:**
            - Higher Silhouette → Better separation
            - Lower Davies-Bouldin → Better clustering
            - Higher Calinski-Harabasz → Better defined clusters
            """)





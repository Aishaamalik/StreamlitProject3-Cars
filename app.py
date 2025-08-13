import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import pickle
import io
import base64
from datetime import datetime
import re

# Configure page
st.set_page_config(
    page_title=" Car Data Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9a9e;
        margin: 1rem 0;
    }
    
    /* Improved tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        min-width: 140px;
        max-width: 200px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #333333;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        word-wrap: break-word;
        white-space: normal;
        line-height: 1.2;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f0;
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Filter card styling */
    .filter-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .filter-card h3 {
        color: #333;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Chart container styling */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Additional tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        justify-content: flex-start;
    }
    
    /* Ensure tab content is properly contained */
    .stTabs [data-baseweb="tab"] span {
        display: block;
        width: 100%;
        text-align: center;
        word-break: break-word;
        hyphens: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

@st.cache_data
def load_data():
    """Load and preprocess the car dataset"""
    try:
        # Load data
        df = pd.read_csv('cars.csv')
        
        # Clean categorical columns - replace NaN with 'Unknown'
        df['Make'] = df['Make'].fillna('Unknown')
        df['Fuel_Type'] = df['Fuel_Type'].fillna('Unknown')
        df['Body_Type'] = df['Body_Type'].fillna('Unknown')
        
        # Clean price column - extract numeric values
        df['Price_Numeric'] = df['Ex-Showroom_Price'].str.extract(r'(\d+(?:,\d+)*)').astype(str)
        df['Price_Numeric'] = df['Price_Numeric'].str.replace(',', '').astype(float)
        
        # Clean displacement column
        df['Displacement_Numeric'] = df['Displacement'].str.extract(r'(\d+)').astype(float)
        
        # Clean power column
        df['Power_Numeric'] = df['Power'].str.extract(r'(\d+)').astype(float)
        
        # Clean torque column
        df['Torque_Numeric'] = df['Torque'].str.extract(r'(\d+)').astype(float)
        
        # Clean mileage columns
        df['City_Mileage_Numeric'] = pd.to_numeric(df['City_Mileage'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        df['Highway_Mileage_Numeric'] = pd.to_numeric(df['Highway_Mileage'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        df['ARAI_Mileage_Numeric'] = pd.to_numeric(df['ARAI_Certified_Mileage'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        
        # Clean dimensions
        df['Height_Numeric'] = pd.to_numeric(df['Height'].str.extract(r'(\d+)')[0], errors='coerce')
        df['Length_Numeric'] = pd.to_numeric(df['Length'].str.extract(r'(\d+)')[0], errors='coerce')
        df['Width_Numeric'] = pd.to_numeric(df['Width'].str.extract(r'(\d+)')[0], errors='coerce')
        
        # Clean weight
        df['Weight_Numeric'] = pd.to_numeric(df['Kerb_Weight'].str.extract(r'(\d+)')[0], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_data_overview(df):
    """Display comprehensive data overview"""
    st.markdown('<div class="main-header"><h1>🚗 Car Data Analytics Dashboard</h1><p>Comprehensive Analysis of Indian Car Market</p></div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Cars</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏭 Brands</h3>
            <h2>{df['Make'].nunique()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_price = df['Price_Numeric'].dropna().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Price</h3>
            <h2>₹{avg_price:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_mileage = df['ARAI_Mileage_Numeric'].dropna().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⛽ Avg Mileage</h3>
            <h2>{avg_mileage:.1f} km/l</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data info
    st.subheader("📋 Dataset Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", len(df.columns))
        st.write("**Missing Values:**", df.isnull().sum().sum())
        
        # Data types
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        # Missing values heatmap
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.values, y=missing_data.index, 
                        title="Missing Values by Column",
                        orientation='h')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

def display_visualizations(df):
    """Display comprehensive data visualizations"""
    st.subheader("📈 Data Visualizations")
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Price_Numeric', nbins=50,
                          title="Price Distribution",
                          labels={'Price_Numeric': 'Price (₹)', 'count': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='Make', y='Price_Numeric',
                    title="Price by Brand",
                    labels={'Price_Numeric': 'Price (₹)', 'Make': 'Brand'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Fuel type and body type analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fuel_counts = df['Fuel_Type'].value_counts()
        fig = px.pie(values=fuel_counts.values, names=fuel_counts.index,
                    title="Fuel Type Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        body_counts = df['Body_Type'].value_counts()
        fig = px.bar(x=body_counts.values, y=body_counts.index,
                    title="Body Type Distribution",
                    orientation='h')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Engine specifications
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='Displacement_Numeric', y='Power_Numeric',
                        color='Fuel_Type', size='Price_Numeric',
                        title="Engine Power vs Displacement",
                        labels={'Displacement_Numeric': 'Displacement (cc)', 
                               'Power_Numeric': 'Power (PS)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Weight_Numeric', y='ARAI_Mileage_Numeric',
                        color='Fuel_Type', size='Price_Numeric',
                        title="Mileage vs Weight",
                        labels={'Weight_Numeric': 'Weight (kg)', 
                               'ARAI_Mileage_Numeric': 'Mileage (km/l)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = ['Price_Numeric', 'Displacement_Numeric', 'Power_Numeric', 
                   'Torque_Numeric', 'ARAI_Mileage_Numeric', 'Weight_Numeric',
                   'Height_Numeric', 'Length_Numeric', 'Width_Numeric']
    
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    st.subheader("🔍 Exploratory Data Analysis")
    
    # Statistical summary
    st.write("**📊 Statistical Summary**")
    numeric_cols = ['Price_Numeric', 'Displacement_Numeric', 'Power_Numeric', 
                   'Torque_Numeric', 'ARAI_Mileage_Numeric', 'Weight_Numeric']
    
    summary_df = df[numeric_cols].describe()
    st.dataframe(summary_df, use_container_width=True)
    
    # Brand analysis
    st.write("**🏭 Brand Analysis**")
    brand_stats = df.groupby('Make').agg({
        'Price_Numeric': ['count', 'mean', 'std'],
        'ARAI_Mileage_Numeric': 'mean',
        'Power_Numeric': 'mean'
    }).round(2)
    
    brand_stats.columns = ['Count', 'Avg_Price', 'Price_Std', 'Avg_Mileage', 'Avg_Power']
    brand_stats = brand_stats.sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Brands by Count:**")
        st.dataframe(brand_stats.head(10), use_container_width=True)
    
    with col2:
        fig = px.bar(brand_stats.head(10), x=brand_stats.head(10).index, y='Count',
                    title="Top 10 Brands by Number of Models")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price analysis by features
    st.write("**💰 Price Analysis by Features**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price by fuel type
        fuel_price = df.groupby('Fuel_Type')['Price_Numeric'].agg(['mean', 'count']).round(2)
        fig = px.bar(x=fuel_price.index, y=fuel_price['mean'],
                    title="Average Price by Fuel Type",
                    labels={'x': 'Fuel Type', 'y': 'Average Price (₹)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price by body type
        body_price = df.groupby('Body_Type')['Price_Numeric'].agg(['mean', 'count']).round(2)
        fig = px.bar(x=body_price.index, y=body_price['mean'],
                    title="Average Price by Body Type",
                    labels={'x': 'Body Type', 'y': 'Average Price (₹)'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.write("**🔗 Feature Relationships**")
    
    # Price vs Power
    fig = px.scatter(df, x='Power_Numeric', y='Price_Numeric',
                    color='Fuel_Type',
                    title="Price vs Power Relationship",
                    labels={'Power_Numeric': 'Power (PS)', 'Price_Numeric': 'Price (₹)'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def perform_statistical_analysis(df):
    """Perform statistical analysis"""
    st.subheader("📊 Statistical Analysis")
    
    # Hypothesis testing and statistical insights
    st.write("**🎯 Key Statistical Insights**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price statistics
        price_stats = {
            'Mean': df['Price_Numeric'].mean(),
            'Median': df['Price_Numeric'].median(),
            'Std': df['Price_Numeric'].std(),
            'Min': df['Price_Numeric'].min(),
            'Max': df['Price_Numeric'].max(),
            'Q1': df['Price_Numeric'].quantile(0.25),
            'Q3': df['Price_Numeric'].quantile(0.75)
        }
        
        st.write("**💰 Price Statistics (₹)**")
        for key, value in price_stats.items():
            st.write(f"{key}: {value:,.0f}")
    
    with col2:
        # Mileage statistics
        mileage_stats = {
            'Mean': df['ARAI_Mileage_Numeric'].mean(),
            'Median': df['ARAI_Mileage_Numeric'].median(),
            'Std': df['ARAI_Mileage_Numeric'].std(),
            'Min': df['ARAI_Mileage_Numeric'].min(),
            'Max': df['ARAI_Mileage_Numeric'].max()
        }
        
        st.write("**⛽ Mileage Statistics (km/l)**")
        for key, value in mileage_stats.items():
            st.write(f"{key}: {value:.2f}")
    
    # ANOVA analysis for price differences
    st.write("**📈 Price Analysis by Categories**")
    
    # Price by fuel type ANOVA
    fuel_groups = [group['Price_Numeric'].values for name, group in df.groupby('Fuel_Type')]
    fuel_groups = [group for group in fuel_groups if len(group) > 0]
    
    if len(fuel_groups) > 1:
        from scipy import stats
        f_stat, p_value = stats.f_oneway(*fuel_groups)
        
        st.write(f"**Fuel Type Price ANOVA:**")
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        st.write(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Correlation analysis
    st.write("**🔗 Correlation Analysis**")
    
    numeric_cols = ['Price_Numeric', 'Displacement_Numeric', 'Power_Numeric', 
                   'Torque_Numeric', 'ARAI_Mileage_Numeric', 'Weight_Numeric']
    
    correlation_with_price = df[numeric_cols].corr()['Price_Numeric'].sort_values(ascending=False)
    
    fig = px.bar(x=correlation_with_price.index, y=correlation_with_price.values,
                title="Correlation with Price",
                labels={'x': 'Features', 'y': 'Correlation Coefficient'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def prepare_data_for_modeling(df):
    """Prepare data for machine learning models"""
    # Select features for modeling
    feature_cols = ['Displacement_Numeric', 'Power_Numeric', 'Torque_Numeric', 
                   'ARAI_Mileage_Numeric', 'Weight_Numeric', 'Height_Numeric', 
                   'Length_Numeric', 'Width_Numeric']
    
    # Remove rows with missing values
    model_df = df[feature_cols + ['Price_Numeric']].dropna()
    
    X = model_df[feature_cols]
    y = model_df['Price_Numeric']
    
    return X, y

def train_models(X, y):
    """Train multiple models and return results"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
    
    return results

def display_model_results(results):
    """Display model comparison results"""
    st.subheader("🤖 Model Performance Comparison")
    
    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'RMSE': f"₹{result['rmse']:,.0f}",
            'MAE': f"₹{result['mae']:,.0f}",
            'R² Score': f"{result['r2']:.4f}",
            'CV R² Mean': f"{result['cv_mean']:.4f}",
            'CV R² Std': f"{result['cv_std']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualize model performance
    col1, col2 = st.columns(2)
    
    with col1:
        # R² scores comparison
        r2_scores = [result['r2'] for result in results.values()]
        model_names = list(results.keys())
        
        fig = px.bar(x=model_names, y=r2_scores,
                    title="R² Scores Comparison",
                    labels={'x': 'Models', 'y': 'R² Score'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE comparison
        rmse_scores = [result['rmse'] for result in results.values()]
        
        fig = px.bar(x=model_names, y=rmse_scores,
                    title="RMSE Comparison",
                    labels={'x': 'Models', 'y': 'RMSE (₹)'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    st.success(f"🏆 **Best Model: {best_model_name}**")
    st.write(f"R² Score: {best_model['r2']:.4f}")
    st.write(f"RMSE: ₹{best_model['rmse']:,.0f}")
    st.write(f"MAE: ₹{best_model['mae']:,.0f}")
    
    return best_model_name, best_model

def model_interpretation(best_model_name, best_model, X, y):
    """Provide model interpretation and insights"""
    st.subheader("🧠 Model Interpretation")
    
    st.write(f"**📊 {best_model_name} Analysis**")
    
    # Feature importance (for tree-based models)
    if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
        feature_importance = best_model['model'].feature_importances_
        feature_names = X.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature',
                    title="Feature Importance",
                    orientation='h')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**🔍 Key Insights:**")
        st.write("• Most important features for price prediction")
        st.write("• Engine power and displacement are key drivers")
        st.write("• Vehicle dimensions also play significant role")
    
    # Model coefficients (for linear models)
    elif 'Linear' in best_model_name or 'Ridge' in best_model_name or 'Lasso' in best_model_name:
        coefficients = best_model['model'].coef_
        feature_names = X.columns
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig = px.bar(coef_df, x='Coefficient', y='Feature',
                    title="Model Coefficients",
                    orientation='h')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction examples
    st.write("**🎯 Prediction Examples**")
    
    # Sample predictions
    X_sample = X.head(5)
    predictions = best_model['model'].predict(best_model['scaler'].transform(X_sample))
    
    example_df = pd.DataFrame({
        'Actual Price': y.head(5),
        'Predicted Price': predictions,
        'Difference': y.head(5) - predictions
    })
    
    st.dataframe(example_df, use_container_width=True)
    
    # Model insights
    st.write("**💡 Model Insights:**")
    st.write("• The model captures the relationship between car features and price")
    st.write("• Engine specifications are the strongest predictors")
    st.write("• Vehicle size and weight also influence pricing")
    st.write("• Fuel efficiency has moderate impact on price")

def export_section(best_model_name, best_model, X, y):
    """Export functionality for the best model"""
    st.subheader("📤 Export Section")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**💾 Export Best Model**")
        
        # Create model package
        model_data = {
            'model': best_model['model'],
            'scaler': best_model['scaler'],
            'feature_names': list(X.columns),
            'model_name': best_model_name,
            'performance': {
                'r2': best_model['r2'],
                'rmse': best_model['rmse'],
                'mae': best_model['mae']
            },
            'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Serialize model
        model_bytes = pickle.dumps(model_data)
        
        # Create download button
        st.download_button(
            label="📥 Download Best Model",
            data=model_bytes,
            file_name=f"car_price_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            mime="application/octet-stream"
        )
    
    with col2:
        st.write("**📊 Export Analysis Report**")
        
        # Generate report
        report = f"""
# Car Price Prediction Model Report

## Model Information
- **Model Type**: {best_model_name}
- **Export Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Dataset Size**: {len(X)} samples
- **Features**: {len(X.columns)} features

## Performance Metrics
- **R² Score**: {best_model['r2']:.4f}
- **RMSE**: ₹{best_model['rmse']:,.0f}
- **MAE**: ₹{best_model['mae']:,.0f}

## Features Used
{', '.join(X.columns)}

## Usage Instructions
1. Load the model using pickle
2. Prepare input data with the same features
3. Scale the data using the included scaler
4. Make predictions using the model

## Model Insights
- Engine specifications are key predictors
- Vehicle dimensions influence pricing
- Fuel efficiency has moderate impact
- Power and displacement are strongest features
        """
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"car_price_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    # Usage example
    st.write("**💻 Usage Example**")
    
    usage_code = f"""
import pickle
import numpy as np

# Load the model
with open('car_price_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# Prepare input data (example)
input_data = np.array([[
    1200,  # Displacement
    85,    # Power
    110,   # Torque
    20,    # Mileage
    1000,  # Weight
    1500,  # Height
    4000,  # Length
    1700   # Width
]])

# Scale the data
input_scaled = scaler.transform(input_data)

# Make prediction
predicted_price = model.predict(input_scaled)[0]
print(f"Predicted Price: ₹{{predicted_price:,.0f}}")
    """
    
    st.code(usage_code, language='python')

def main():
    """Main application function"""
    # Load data
    with st.spinner("Loading car dataset..."):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if 'cars.csv' exists in the current directory.")
        return
    
    st.session_state.data_loaded = True
    
    # Sidebar with filters
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🚗 Car Analytics</h2>
            <p>Filter your data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Brand filter
        st.markdown('<div class="filter-card"><h3>🏭 Brand Filter</h3></div>', unsafe_allow_html=True)
        # Clean and sort brand names, handling NaN values
        brand_options = sorted([str(x) for x in df['Make'].dropna().unique()])
        selected_brands = st.multiselect(
            "Select Brands",
            options=brand_options,
            default=brand_options,  # Select all brands initially
            help="Choose car brands to analyze"
        )
        
        # Price range filter
        st.markdown('<div class="filter-card"><h3>💰 Price Range</h3></div>', unsafe_allow_html=True)
        price_min = float(df['Price_Numeric'].dropna().min())
        price_max = float(df['Price_Numeric'].dropna().max())
        price_range = st.slider(
            "Price Range (₹)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            step=10000.0,
            help="Select price range for analysis"
        )
        
        # Fuel type filter
        st.markdown('<div class="filter-card"><h3>⛽ Fuel Type</h3></div>', unsafe_allow_html=True)
        # Clean and sort fuel types, handling NaN values
        fuel_options = sorted([str(x) for x in df['Fuel_Type'].dropna().unique()])
        selected_fuel_types = st.multiselect(
            "Select Fuel Types",
            options=fuel_options,
            default=fuel_options,  # Select all fuel types initially
            help="Choose fuel types to include"
        )
        
        # Body type filter
        st.markdown('<div class="filter-card"><h3>🚙 Body Type</h3></div>', unsafe_allow_html=True)
        # Clean and sort body types, handling NaN values
        body_options = sorted([str(x) for x in df['Body_Type'].dropna().unique()])
        selected_body_types = st.multiselect(
            "Select Body Types",
            options=body_options,
            default=body_options,  # Select all body types initially
            help="Choose body types to include"
        )
        
        # Power range filter
        st.markdown('<div class="filter-card"><h3>⚡ Power Range</h3></div>', unsafe_allow_html=True)
        power_min = float(df['Power_Numeric'].dropna().min())
        power_max = float(df['Power_Numeric'].dropna().max())
        power_range = st.slider(
            "Power Range (PS)",
            min_value=power_min,
            max_value=power_max,
            value=(power_min, power_max),
            step=10.0,
            help="Select power range for analysis"
        )
        
        # Apply filters
        if st.button("🔍 Apply Filters", type="primary"):
            st.session_state.filters_applied = True
        
        # Clear filters
        if st.button("🗑️ Clear Filters"):
            st.session_state.filters_applied = False
    
    # Apply filters to data
    filtered_df = df.copy()
    
    # Always apply filters, but show all data initially
    if len(selected_brands) > 0 and len(selected_fuel_types) > 0 and len(selected_body_types) > 0:
        filtered_df = df[
            (df['Make'].isin(selected_brands)) &
            (df['Price_Numeric'] >= price_range[0]) &
            (df['Price_Numeric'] <= price_range[1]) &
            (df['Fuel_Type'].isin(selected_fuel_types)) &
            (df['Body_Type'].isin(selected_body_types)) &
            (df['Power_Numeric'] >= power_range[0]) &
            (df['Power_Numeric'] <= power_range[1])
        ]
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data matches the selected filters. Please adjust your criteria.")
            filtered_df = df
    else:
        filtered_df = df
    
    # Show filter summary
    if st.session_state.get('filters_applied', False):
        st.info(f"📊 Showing {len(filtered_df)} cars out of {len(df)} total cars")
    else:
        st.info(f"📊 Showing all {len(filtered_df)} cars (no filters applied)")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Overview", 
        "📈 Charts", 
        "🔍 EDA", 
        "📊 Stats",
        "🤖 Models",
        "🧠 Insights",
        "📤 Export"
    ])
    
    with tab1:
        display_data_overview(filtered_df)
    
    with tab2:
        display_visualizations(filtered_df)
    
    with tab3:
        perform_eda(filtered_df)
    
    with tab4:
        perform_statistical_analysis(filtered_df)
    
    with tab5:
        st.subheader("🤖 Machine Learning Model Application")
        
        # Check if user wants to proceed with modeling
        if st.button("🚀 Start Model Training", type="primary"):
            with st.spinner("Preparing data and training models..."):
                try:
                    # Prepare data
                    X, y = prepare_data_for_modeling(filtered_df)
                    
                    if len(X) == 0:
                        st.error("❌ Insufficient data for modeling. Please check data quality.")
                        return
                    
                    # Train models
                    results = train_models(X, y)
                    
                    if not results:
                        st.error("❌ No models were successfully trained.")
                        return
                    
                    # Display results
                    best_model_name, best_model = display_model_results(results)
                    
                    # Store best model in session state
                    st.session_state.best_model = best_model
                    st.session_state.model_name = best_model_name
                    
                    st.success("✅ Model training completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
        
        # Show model results if available
        if st.session_state.best_model is not None:
            st.success(f"✅ Best model available: {st.session_state.model_name}")
            
            # Show model metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{st.session_state.best_model['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"₹{st.session_state.best_model['rmse']:,.0f}")
            with col3:
                st.metric("MAE", f"₹{st.session_state.best_model['mae']:,.0f}")
    
    with tab6:
        if st.session_state.best_model is not None:
            X, y = prepare_data_for_modeling(filtered_df)
            model_interpretation(st.session_state.model_name, st.session_state.best_model, X, y)
        else:
            st.warning("⚠️ Please train models first in the Model Application tab.")
    
    with tab7:
        if st.session_state.best_model is not None:
            X, y = prepare_data_for_modeling(filtered_df)
            export_section(st.session_state.model_name, st.session_state.best_model, X, y)
        else:
            st.warning("⚠️ Please train models first in the Model Application tab.")

if __name__ == "__main__":
    main() 
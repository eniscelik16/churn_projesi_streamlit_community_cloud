import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  
import pandas as pd
import numpy as np
import joblib
import shap
import base64
import os
from plotly.subplots import make_subplots

# Page settings must always be at the top
st.set_page_config(page_title="Bank Churn Risk Dashboard", page_icon="🏦", layout="wide")

# ==========================================
# 1. SESSION MANAGEMENT
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'current_customer' not in st.session_state:
    st.session_state.current_customer = None
if 'base_risk' not in st.session_state:
    st.session_state.base_risk = None

# ==========================================
# GLOBAL VERİ YÜKLEME
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Churn_Modelling.csv')
        return df
    except:
        return pd.DataFrame()

df_global = load_data()

# ==========================================
# 2. MODEL YÜKLEME (BULUT İÇİN TEKİL YAPI)
# ==========================================
@st.cache_resource
def load_model_and_explainer():
    pack = joblib.load('churn_thesis_model.pkl')
    explainer = shap.TreeExplainer(pack['model'])
    return pack['model'], pack['scaler'], pack['features'], explainer

model, scaler, expected_features, explainer = load_model_and_explainer()

# TAHMİN VE SHAP HESAPLAMA (DOĞRUDAN MODEL KULLANILIYOR)
def get_api_prediction(data_dict):
    df_input = pd.DataFrame([data_dict])
    df_input = pd.get_dummies(df_input, drop_first=True)
    
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_features]
    
    scaled_input = scaler.transform(df_input)
    prob = model.predict_proba(scaled_input)[0][1]
    
    shap_values = explainer.shap_values(scaled_input, check_additivity=False)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
        
    return {
        "churn_probability": float(prob),
        "shap_values": np.array(shap_vals).flatten().tolist(),
        "features": list(expected_features)
    }

# ==========================================
# 3. LOGIN SCREEN
# ==========================================
def login_screen():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏦</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Manager Login</h3>", unsafe_allow_html=True)
        st.write("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log in", width="stretch"):
            if username == st.secrets["auth"]["username"] and password == st.secrets["auth"]["password"]:
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password!")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
def main_dashboard():
    st.sidebar.title("Admin Panel")
    st.sidebar.info("Welcome, **Manager**")

    st.sidebar.subheader("Filter By:")
    geo_filter = st.sidebar.multiselect("Geography", options=["France", "Germany", "Spain"], default=["France", "Germany", "Spain"])
    gender_filter = st.sidebar.multiselect("Gender", options=["Male", "Female"], default=["Male", "Female"])
    card_filter = st.sidebar.selectbox("Has Card", options=["All", "Yes", "No"])
    status_filter = st.sidebar.selectbox("Customer Status", options=["All", "Active", "Inactive"])

    df_filtered = df_global.copy()
    if not df_filtered.empty:
        df_filtered = df_filtered[df_filtered['Geography'].isin(geo_filter)]
        df_filtered = df_filtered[df_filtered['Gender'].isin(gender_filter)]
        if card_filter == "Yes": df_filtered = df_filtered[df_filtered['HasCrCard'] == 1]
        elif card_filter == "No": df_filtered = df_filtered[df_filtered['HasCrCard'] == 0]
        if status_filter == "Active": df_filtered = df_filtered[df_filtered['IsActiveMember'] == 1]
        elif status_filter == "Inactive": df_filtered = df_filtered[df_filtered['IsActiveMember'] == 0]

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🏦 Customer Risk Analysis Dashboard")
    tab_overview, tab1, tab2, tab3 = st.tabs(["📊 Overview", "👤 Single Customer Analysis", "🧪 What-If Simulator", "📂 Batch Analysis"])

    # === TAB: OVERVIEW ===
    with tab_overview:
        if df_global.empty: 
            st.warning("Please upload 'Churn_Modelling.csv' to the project root directory to see the overview.")
        else:
            title_col, obj_col = st.columns([1, 1])
            with title_col:
                st.markdown("<h2 style='margin-bottom:0; font-family:Georgia,serif; color:var(--text-color);'>Bank Customer Churn Analysis</h2>", unsafe_allow_html=True)
            with obj_col:
                st.markdown("<div style='background-color: rgba(187, 63, 63, 0.08); border-left: 4px solid #bb3f3f; padding: 12px 16px; border-radius: 4px; font-size: 13px; line-height: 1.6; color: var(--text-color);'><b style='color: #bb3f3f;'>Objective :</b> To identify the customer segments currently experiencing churn and formulate actionable recommendations tailored for these groups, with the goal of enhancing retention efforts.</div>", unsafe_allow_html=True)

            st.markdown("")

            total_cust = len(df_filtered)
            active_cust = df_filtered[df_filtered['IsActiveMember'] == 1].shape[0] if total_cust > 0 else 0
            exited_cust = df_filtered[df_filtered['Exited'] == 1].shape[0] if total_cust > 0 else 0
            churn_rate = (exited_cust / total_cust) * 100 if total_cust > 0 else 0

            st.markdown(f"""
            <div style='display:flex; gap:15px; margin-bottom:15px;'>
                <div style='flex:1; background-color: rgba(130, 130, 130, 0.1); border: 1px solid rgba(130, 130, 130, 0.2); border-top: 3px solid #bb3f3f; padding: 18px 12px; text-align: center; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='color: var(--text-color); opacity: 0.8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>Total Customers</div>
                    <div style='color: var(--text-color); font-size: 30px; font-weight: bold; margin-top: 5px;'>{total_cust:,}</div>
                </div>
                <div style='flex:1; background-color: rgba(130, 130, 130, 0.1); border: 1px solid rgba(130, 130, 130, 0.2); border-top: 3px solid #bb3f3f; padding: 18px 12px; text-align: center; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='color: var(--text-color); opacity: 0.8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>Active Customers</div>
                    <div style='color: var(--text-color); font-size: 30px; font-weight: bold; margin-top: 5px;'>{active_cust:,}</div>
                </div>
                <div style='flex:1; background-color: rgba(130, 130, 130, 0.1); border: 1px solid rgba(130, 130, 130, 0.2); border-top: 3px solid #bb3f3f; padding: 18px 12px; text-align: center; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='color: var(--text-color); opacity: 0.8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>Exited Customers</div>
                    <div style='color: var(--text-color); font-size: 30px; font-weight: bold; margin-top: 5px;'>{exited_cust:,}</div>
                </div>
                <div style='flex:1; background-color: rgba(130, 130, 130, 0.1); border: 1px solid rgba(130, 130, 130, 0.2); border-top: 3px solid #bb3f3f; padding: 18px 12px; text-align: center; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='color: var(--text-color); opacity: 0.8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>Churn Rate</div>
                    <div style='color: #bb3f3f; font-size: 30px; font-weight: bold; margin-top: 5px;'>{churn_rate:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY GENDER</h4>", unsafe_allow_html=True)
                if total_cust == 0:
                    female_churn, male_churn = 0, 0
                    st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>No data to display</p>", unsafe_allow_html=True)
                else:
                    gender_churn = df_filtered.groupby('Gender')['Exited'].mean() * 100
                    female_churn = gender_churn.get('Female', 0)
                    male_churn = gender_churn.get('Male', 0)
                    diff = female_churn - male_churn
                    if diff > 0:
                        st.markdown(f"<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Females show {diff:.0f}% higher churn</p>", unsafe_allow_html=True)
                    elif diff < 0:
                        st.markdown(f"<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Males show {abs(diff):.0f}% higher churn</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Both genders show equal churn</p>", unsafe_allow_html=True)

                fig_gen = make_subplots(rows=2, cols=1, specs=[[{'type': 'domain'}], [{'type': 'domain'}]], vertical_spacing=0.1)
                fig_gen.add_trace(go.Pie(
                    values=[female_churn, 100 - female_churn] if total_cust > 0 else [0, 100], labels=['Churned', 'Retained'],
                    hole=0.75, marker_colors=['#bb3f3f', '#e0e0e0'],
                    textinfo='percent', textposition='outside',
                    sort=False, direction='clockwise', rotation=90
                ), row=1, col=1)
                fig_gen.add_trace(go.Pie(
                    values=[male_churn, 100 - male_churn] if total_cust > 0 else [0, 100], labels=['Churned', 'Retained'],
                    hole=0.75, marker_colors=['#bb3f3f', '#e0e0e0'],
                    textinfo='percent', textposition='outside',
                    sort=False, direction='clockwise', rotation=90
                ), row=2, col=1)
                fig_gen.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=400,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

                def get_base64_image(image_path):
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as img_file:
                            return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')
                    return None

                female_b64 = get_base64_image("female.png")
                male_b64 = get_base64_image("male.png")
                if female_b64:
                    fig_gen.add_layout_image(dict(source=female_b64, xref="paper", yref="paper",
                        x=0.5, y=0.775, sizex=0.30, sizey=0.30, xanchor="center", yanchor="middle", sizing="contain"))
                if male_b64:
                    fig_gen.add_layout_image(dict(source=male_b64, xref="paper", yref="paper",
                        x=0.5, y=0.225, sizex=0.35, sizey=0.35, xanchor="center", yanchor="middle", sizing="contain"))
                st.plotly_chart(fig_gen, width="stretch", theme="streamlit")

            with c2:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY ACTIVE Vs INACTIVE MEMBER</h4>", unsafe_allow_html=True)
                if total_cust == 0:
                    inactive_rate, active_rate_val = 0, 0
                else:
                    active_churn_rate = df_filtered.groupby('IsActiveMember')['Exited'].mean() * 100
                    inactive_rate = active_churn_rate.get(0, 0)
                    active_rate_val = active_churn_rate.get(1, 0)
                
                st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Inactive members show higher churn rate</p>", unsafe_allow_html=True)

                categories = ['Active Member', 'Inactive Member']
                retained_vals = [100 - active_rate_val, 100 - inactive_rate] if total_cust > 0 else [0, 0]
                churned_vals = [active_rate_val, inactive_rate] if total_cust > 0 else [0, 0]

                fig_active = go.Figure()
                fig_active.add_trace(go.Bar(y=categories, x=retained_vals, name='Retained', orientation='h',
                    marker_color='#cccccc', text=[f'{v:.0f}%' for v in retained_vals],
                    textposition='inside', textfont=dict(color='#333', size=12)))
                fig_active.add_trace(go.Bar(y=categories, x=churned_vals, name='Churned', orientation='h',
                    marker_color='#bb3f3f', text=[f'{v:.0f}%' for v in churned_vals],
                    textposition='inside', textfont=dict(color='white', size=12)))
                fig_active.update_layout(barmode='stack', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    margin=dict(t=30, b=10, l=10, r=10), height=400,
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False), yaxis=dict(showgrid=False),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_active, width="stretch", theme="streamlit")

            with c3:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY GEOGRAPHY</h4>", unsafe_allow_html=True)
                if total_cust == 0:
                    st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>No data to display</p>", unsafe_allow_html=True)
                    fig_geo = go.Figure()
                else:
                    geo_churn = df_filtered.groupby('Geography')['Exited'].mean().reset_index()
                    geo_churn['Churn Rate (%)'] = geo_churn['Exited'] * 100
                    if not geo_churn.empty:
                        max_c = geo_churn.loc[geo_churn['Churn Rate (%)'].idxmax()]
                        st.markdown(f"<p style='font-size:13px; color:var(--text-color); opacity:0.8;'><b>{max_c['Geography']}</b> has the highest churn rate at <b>{max_c['Churn Rate (%)']:.0f}%</b></p>", unsafe_allow_html=True)

                    fig_geo = px.choropleth(geo_churn, locations="Geography", locationmode="country names",
                        color="Churn Rate (%)", scope="europe", color_continuous_scale=["#fce8cc", "#bb3f3f"],
                        hover_name="Geography", hover_data={"Geography": False, "Churn Rate (%)": ':.2f'})
                    
                    country_coords = {'France': {'lat': 46.2276, 'lon': 2.2137},
                        'Germany': {'lat': 51.1657, 'lon': 10.4515}, 'Spain': {'lat': 40.4637, 'lon': -3.7492}}
                    
                    for idx, row in geo_churn.iterrows():
                        country = row['Geography']
                        rate = row['Churn Rate (%)']
                        if country in country_coords:
                            fig_geo.add_scattergeo(lat=[country_coords[country]['lat']], lon=[country_coords[country]['lon']],
                                text=[f"<b>{country}</b><br>{rate:.2f}%"], mode="text",
                                textfont=dict(color="#333333", size=14), showlegend=False, hoverinfo="skip")
                    
                fig_geo.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#d9d9d9", showland=True, landcolor="#ececec")
                fig_geo.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, coloraxis_showscale=False, geo=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_geo, width="stretch", theme="streamlit")

            st.markdown("---")

            c4, c5, c6 = st.columns(3)

            with c4:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY CREDIT SCORE</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Churn risk higher for customers with poor credit scores</p>", unsafe_allow_html=True)
                bins_cs = [300, 500, 600, 700, 800, 850]
                labels_cs = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
                df_cs = df_filtered.copy()
                df_cs['ScoreGroup'] = pd.cut(df_cs['CreditScore'], bins=bins_cs, labels=labels_cs)
                score_data = df_cs.groupby('ScoreGroup', observed=False).agg(Total=('Exited', 'count'), Exited=('Exited', 'sum')).reset_index()
                score_data['Retained'] = score_data['Total'] - score_data['Exited']

                fig_score = go.Figure()
                fig_score.add_trace(go.Bar(y=score_data['ScoreGroup'], x=score_data['Retained'], name='Customer', orientation='h', marker_color='#cccccc'))
                fig_score.add_trace(go.Bar(y=score_data['ScoreGroup'], x=score_data['Exited'], name='Exited', orientation='h', marker_color='#bb3f3f'))
                fig_score.update_layout(barmode='group', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), margin=dict(t=30, b=30, l=10, r=10), height=300, xaxis_title="Customer vs Exited", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(categoryorder='array', categoryarray=['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']))
                st.plotly_chart(fig_score, width="stretch", theme="streamlit")

            with c5:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY NUMBER OF PRODUCTS</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>High churn among those buying 3+ products</p>", unsafe_allow_html=True)
                prod_data = df_filtered.groupby('NumOfProducts').agg(Total=('Exited', 'count'), Exited=('Exited', 'sum')).reset_index()
                prod_data['Retained'] = prod_data['Total'] - prod_data['Exited']
                prod_data['Churn_Pct'] = (prod_data['Exited'] / prod_data['Total'] * 100).fillna(0).round(0)
                prod_data['Retain_Pct'] = (100 - prod_data['Churn_Pct']).fillna(0).round(0)

                fig_prod = go.Figure()
                fig_prod.add_trace(go.Bar(x=prod_data['NumOfProducts'], y=prod_data['Retain_Pct'], name='Retained', marker_color='#cccccc', text=[f"{v:.0f}%" for v in prod_data['Retain_Pct']], textposition='inside', textfont=dict(color='#333', size=11)))
                fig_prod.add_trace(go.Bar(x=prod_data['NumOfProducts'], y=prod_data['Churn_Pct'], name='Churned', marker_color='#bb3f3f', text=[f"{v:.0f}%" for v in prod_data['Churn_Pct']], textposition='inside', textfont=dict(color='white', size=11)))
                fig_prod.update_layout(barmode='stack', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), margin=dict(t=30, b=30, l=10, r=10), height=300, xaxis_title="NumOfProducts", yaxis_title="Customer %", yaxis=dict(range=[0, 100], ticksuffix='%'), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_prod, width="stretch", theme="streamlit")

            with c6:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>Driver Analysis</span></h4>", unsafe_allow_html=True)
                st.markdown("""<div style='font-size:12px; color:var(--text-color); opacity:0.8; line-height:1.6; margin-bottom:10px;'>
                Age is a critical demographic factor. Churn risk peaks during middle age (around 46-55) where customers are highly financially active, but significantly decreases for senior customers (65+) due to banking inertia. Inactive members are more likely to churn. Germany has a higher churn rate. Higher credit score correlates with lower churn. More product purchases increase churn likelihood.
                </div>""", unsafe_allow_html=True)
                st.markdown("<p style='font-size:13px; color:#bb3f3f; font-weight:bold;'>Coefficient by Driver Name</p>", unsafe_allow_html=True)

                if model is not None:
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({'Feature': expected_features, 'Importance': importances})
                    feat_imp = feat_imp.sort_values('Importance', ascending=True)
                    colors_imp = ['#bb3f3f' if v > feat_imp['Importance'].median() else '#cccccc' for v in feat_imp['Importance']]
                    fig_driver = go.Figure(go.Bar(x=feat_imp['Importance'], y=feat_imp['Feature'], orientation='h', marker_color=colors_imp))
                    fig_driver.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=250, xaxis=dict(showgrid=True, gridcolor='#eee'), yaxis=dict(tickfont=dict(size=9)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_driver, width="stretch", theme="streamlit")

            st.markdown("---")

            c7, c8 = st.columns(2)

            with c7:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY BALANCE IN ACCOUNT</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>High-churn risk observed among customers with larger balances</p>", unsafe_allow_html=True)
                bal_bins = [0, 30000, 60000, 90000, 120000, 150000, float('inf')]
                bal_labels = ['0-30K', '30K-60K', '60K-90K', '90K-120K', '120K-150K', '150K+']
                df_bal = df_filtered.copy()
                df_bal['BalanceBin'] = pd.cut(df_bal['Balance'], bins=bal_bins, labels=bal_labels, include_lowest=True)
                bal_data = df_bal.groupby('BalanceBin', observed=False).agg(Total=('Exited', 'count'), Exited=('Exited', 'sum')).reset_index()
                bal_data['Retained'] = bal_data['Total'] - bal_data['Exited']

                fig_bal = go.Figure()
                fig_bal.add_trace(go.Bar(x=bal_data['BalanceBin'], y=bal_data['Retained'], name='Customer', marker_color='#cccccc'))
                fig_bal.add_trace(go.Bar(x=bal_data['BalanceBin'], y=bal_data['Exited'], name='Exited', marker_color='#bb3f3f'))
                fig_bal.update_layout(barmode='group', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), margin=dict(t=30, b=30, l=10, r=10), height=350, xaxis_title="Balance (Binned)", yaxis_title="Customer", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bal, width="stretch", theme="streamlit")

            with c8:
                st.markdown("<h4 style='color:var(--text-color);'><span style='color:#bb3f3f;'>CHURN</span> BY AGE</h4>", unsafe_allow_html=True)
                
                if total_cust == 0:
                    st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>No data to display</p>", unsafe_allow_html=True)
                    fig_age = go.Figure()
                    fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig_age, width="stretch", theme="streamlit")
                else:
                    age_bins = [0, 35, 45, 55, 100]
                    age_labels = ['<35 yrs', '36-45 yrs', '46-55 yrs', '55+']
                    df_age = df_filtered.copy()
                    df_age['AgeGroup'] = pd.cut(df_age['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
                    age_data = df_age.groupby('AgeGroup', observed=False).agg(Total=('Exited', 'count'), Exited=('Exited', 'sum')).reset_index()
                    age_data['Retained'] = age_data['Total'] - age_data['Exited']
                    age_data['Churn_Pct'] = (age_data['Exited'] / age_data['Total'] * 100).fillna(0).round(0)
                    age_data['Retain_Pct'] = (100 - age_data['Churn_Pct']).fillna(0).round(0)

                    if age_data['Total'].sum() > 0 and not age_data['Churn_Pct'].isna().all():
                        max_age_grp = age_data.loc[age_data['Churn_Pct'].idxmax()]
                        st.markdown(f"<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>Churn rate is <b>{max_age_grp['Churn_Pct']:.0f}%</b> in the age group of <b>{max_age_grp['AgeGroup']}</b></p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='font-size:13px; color:var(--text-color); opacity:0.8;'>No data available for the selected filters.</p>", unsafe_allow_html=True)

                    age_data['Label'] = age_data.apply(lambda r: f"{r['AgeGroup']}\n{r['Total']:,.0f} ({r['Total']/total_cust*100:.0f}%)", axis=1)

                    fig_age = go.Figure()
                    fig_age.add_trace(go.Bar(x=age_data['Label'], y=age_data['Retain_Pct'], name='Retained', marker_color='#cccccc', text=[f"{v:.0f}%" for v in age_data['Retain_Pct']], textposition='inside', textfont=dict(color='#333', size=11)))
                    fig_age.add_trace(go.Bar(x=age_data['Label'], y=age_data['Churn_Pct'], name='Churned', marker_color='#bb3f3f', text=[f"{v:.0f}%" for v in age_data['Churn_Pct']], textposition='inside', textfont=dict(color='white', size=11)))
                    fig_age.update_layout(barmode='stack', showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), margin=dict(t=30, b=30, l=10, r=10), height=350, yaxis=dict(range=[0, 100], ticksuffix='%'), yaxis_title="Customer %", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_age, width="stretch", theme="streamlit")

    # --- TAB 1: SINGLE CUSTOMER ANALYSIS ---
    with tab1:
        st.subheader("Customer Parameters")
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
        with col_form2:
            balance = st.number_input("Account Balance (€)", min_value=0.0, value=50000.0, step=1000.0)
            est_salary = st.number_input("Estimated Salary (€)", min_value=0.0, value=60000.0, step=1000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        with col_form3:
            geography = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        customer_data = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": has_cr_card, "IsActiveMember": is_active, "EstimatedSalary": est_salary
        }

        if st.button("🔍 Perform Risk Analysis", width="stretch"):
            st.session_state.run_analysis = True

        if st.session_state.get('run_analysis', False):
            with st.spinner("AI Calculating..."):
                result = get_api_prediction(customer_data)
                
                if result:
                    churn_probability = result["churn_probability"] * 100
                    st.session_state.current_customer = customer_data
                    st.session_state.base_risk = churn_probability

                    customer_value = balance + (est_salary * 0.20)
                    expected_loss = customer_value * (churn_probability / 100)

                    st.write("---")
                    st.subheader("💰 Financial Impact Analysis (CLTV)")
                    fin_col1, fin_col2, fin_col3 = st.columns(3)
                    fin_col1.metric(label="Customer Value to Bank", value=f"€{customer_value:,.2f}")
                    fin_col2.metric(label="Churn Probability", value=f"%{churn_probability:.1f}")
                    fin_col3.metric(label="Expected Financial Loss", value=f"€{expected_loss:,.2f}", delta="- Risk Amount", delta_color="inverse")

                    st.write("---")
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.subheader("📊 Model Output")
                        st.markdown("<h5 style='color:var(--text-color); opacity:0.9;'>SHAP (SHapley Additive exPlanations)</h5>", unsafe_allow_html=True)
                        if "shap_values" in result and "features" in result:
                            shap_vals = np.array(result["shap_values"])
                            expected_feats = np.array(result["features"])
                            
                            sort_inds = np.argsort(np.abs(shap_vals))
                            sorted_features = expected_feats[sort_inds]
                            sorted_shap = shap_vals[sort_inds]
                            colors = ['salmon' if float(val) > 0 else 'lightgreen' for val in sorted_shap]

                            fig_shap = go.Figure(go.Bar(x=sorted_shap, y=sorted_features, orientation='h', marker_color=colors))
                            fig_shap.update_layout(xaxis_title="<- Factors that Reduce Risk | Factors that Increase Risk ->", margin=dict(l=0, r=0, t=0, b=0), height=300)
                            st.plotly_chart(fig_shap, width="stretch", theme="streamlit")

                    with res_col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=churn_probability,
                            title={'text': "Churn Probability (%)", 'font': {'size': 24}},
                            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "black"},
                                   'steps': [{'range': [0, 40], 'color': "lightgreen"}, {'range': [40, 70], 'color': "gold"}, {'range': [70, 100], 'color': "salmon"}]}))
                        st.plotly_chart(fig_gauge, width="stretch", theme="streamlit")

                    # --- AI AGENT KISMI ---
                    st.divider()
                    st.subheader("🤖 Otonom AI Agent: Pazarlama Asistanı")
                    
                    if churn_probability >= 50.0 and balance > 75000.0:
                        st.error("🚨 KRİTİK ALARM: Yüksek değerli bir müşteriyi kaybetmek üzereyiz!")
                        st.markdown("""**🧠 Agent'ın Düşüncesi:** Bu müşterinin bankamızda 75.000 € üzerinde mevduatı var ve modeli ayrılma riskini çok yüksek (%50+) olarak belirledi. Rakiplerden yüksek faiz teklifi almış olabilir. Acil olarak VIP elde tutma (Retention) politikası uygulanmalı.""")
                        st.warning("⚡ Otonom Aksiyon: Şube müdürünün ekranına acil arama görevi düşürüldü. Müşteriye +%2 ek mevduat faizi ve özel portföy yöneticisi atanması teklif edilecek.")
                        if st.button("📞 VIP Müşteri Temsilcisini Tetikle"):
                            st.success("✅ Görev oluşturuldu: Müşteri temsilcisi 15 dakika içinde müşteriyi arayacak.")

                    elif churn_probability >= 50.0 and balance <= 75000.0:
                        st.warning("⚠️ UYARI: Müşteri hareketliliği azalıyor, terk riski yüksek.")
                        st.markdown("""**🧠 Agent'ın Düşüncesi:** Müşteri bankayı bırakma eğiliminde ancak bakiyesi banka ortalamasının altında. Yüksek maliyetli bir kampanya (özel faiz vb.) yapmak bankanın kar/zarar dengesini bozar. Düşük maliyetli ve otomatik bir tutundurma politikası izlenmeli.""")
                        st.info("✉️ Otonom Aksiyon: Müşteriye kredi kartı aidat iadesi ve ücretsiz EFT/Havale kampanyası otomatik SMS ile iletiliyor.")
                        if st.button("📩 Otomatik SMS Kampanyasını Başlat"):
                            st.success("✅ SMS sistemi tetiklendi. Kampanya teklifi müşteriye gönderildi.")

                    else:
                        st.success("✅ GÜVENLİ: Müşteri bankamıza sadık görünüyor.")
                        st.markdown("""**🧠 Agent'ın Düşüncesi:** Müşteri sadakati yüksek (Terk riski %50'nin altında). Onu elde tutmak için ekstra maliyete veya faiz artırımına girmeye gerek yok. Bu güven ilişkisi, çapraz satış (Cross-sell) fırsatına çevrilmeli.""")
                        if age < 35:
                            urun_teklifi = "Yeni Nesil Teknoloji Fonları ve Düşük Faizli Tüketici Kredisi"
                        else:
                            urun_teklifi = "Bireysel Emeklilik Sistemi (BES) ve Sabit Getirili Mevduat"
                        st.info(f"🎯 Otonom Aksiyon: Müşterinin yaş profiline göre hedeflenmiş ürünler önerilecek: **{urun_teklifi}**")
                        if st.button("📈 Çapraz Satış (Cross-Sell) Fırsatlarını Onayla"):
                            st.success(f"✅ Pazarlama algoritması güncellendi. Müşteri mobil şubeye girdiğinde '{urun_teklifi}' reklamını görecek.")

    # --- TAB 2: WHAT-IF SIMULATOR ---
    with tab2:
        if st.session_state.current_customer is not None:
            cust = st.session_state.current_customer
            base_risk = st.session_state.base_risk
            st.metric(label="Current Churn Risk", value=f"%{base_risk:.1f}")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                new_balance = st.number_input("New Account Balance (€)", value=float(cust["Balance"]), step=1000.0, key="sim_bal")
                new_active = st.selectbox("Set Customer to Active?", [1, 0], index=0 if cust["IsActiveMember"] == 1 else 1, key="sim_act")
            with sim_col2:
                new_crcard = st.selectbox("Define Credit Card Campaign?", [1, 0], index=0 if cust["HasCrCard"] == 1 else 1, key="sim_cr")
                new_products = st.slider("Change Number of Products", 1, 4, value=cust["NumOfProducts"], key="sim_prod")

            if st.button("🔄 Simulate Change Scenario", type="primary", width="stretch"):
                sim_data = cust.copy()
                sim_data.update({"Balance": new_balance, "IsActiveMember": new_active, "HasCrCard": new_crcard, "NumOfProducts": new_products})

                res_sim = get_api_prediction(sim_data)
                
                if res_sim:
                    new_risk = res_sim["churn_probability"] * 100
                    diff = new_risk - base_risk

                    if diff < 0:
                        st.metric(label="New Risk", value=f"%{new_risk:.1f}", delta=f"{diff:.1f} Points", delta_color="normal")
                    else:
                        st.metric(label="New Risk", value=f"%{new_risk:.1f}", delta=f"+{diff:.1f} Points", delta_color="inverse")
        else:
            st.warning("Please perform an analysis from the 'Single Customer Analysis' tab first.")

    # --- TAB 3: BATCH CUSTOMER UPLOAD ---
    with tab3:
        st.subheader("📁 Financially Focused Batch Analysis")
        uploaded_file = st.file_uploader("Select File", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if st.button("🚀 Analyze and Prioritize Entire List", width="stretch"):
                progress_bar = st.progress(0)
                results_list = []
                total_rows = len(df)
                
                for index, row in df.iterrows():
                    cust_row = {
                        "CreditScore": int(row["CreditScore"]), "Geography": str(row["Geography"]),
                        "Gender": str(row["Gender"]), "Age": int(row["Age"]), "Tenure": int(row["Tenure"]), 
                        "Balance": float(row["Balance"]), "NumOfProducts": int(row["NumOfProducts"]), 
                        "HasCrCard": int(row["HasCrCard"]), "IsActiveMember": int(row["IsActiveMember"]), 
                        "EstimatedSalary": float(row["EstimatedSalary"])
                    }
                    
                    res_batch = get_api_prediction(cust_row)
                    
                    if res_batch:
                        churn_prob = res_batch["churn_probability"]
                        c_value = cust_row["Balance"] + (cust_row["EstimatedSalary"] * 0.20)
                        exp_loss = c_value * churn_prob
                        risk_level = "High Risk" if churn_prob >= 0.50 else "Low Risk"
                        
                        results_list.append({
                            "Customer ID": row.get("CustomerId", index), "Risk (%)": round(churn_prob * 100, 2),
                            "Risk Level": risk_level, "Customer Value (€)": round(c_value, 2),
                            "Expected Loss (€)": round(exp_loss, 2)
                        })
                    progress_bar.progress((index + 1) / total_rows)

                if results_list:
                    results_df = pd.DataFrame(results_list).sort_values(by="Expected Loss (€)", ascending=False).reset_index(drop=True)
                    def color_risk(val):
                        return f"color: {'red' if 'High' in str(val) else 'green'}"
                    styled_df = results_df.style.map(color_risk, subset=['Risk Level']).format(
                        {"Customer Value (€)": "{:,.2f}", "Expected Loss (€)": "{:,.2f}"})
                    st.success("✅ Analysis completed!")
                    st.dataframe(styled_df, width="stretch")
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download", data=csv, file_name='financial_churn_report.csv', mime='text/csv')

if not st.session_state.logged_in:
    login_screen()
else:
    main_dashboard()
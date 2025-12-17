import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from backend import MyLinearRegression, DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="ML Algo Studio", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1 {color: #4F8BF9;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: bold;}
    .metric-container {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("Machine Learning Algorithm Studio")

if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = DatasetManager()
if 'df_main' not in st.session_state:
    st.session_state.df_main = None

st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio("Source", ["Sklearn Dataset", "Create Synthetic (Non-Linear)", "Upload CSV"])

if data_source == "Sklearn Dataset":
    d_name = st.sidebar.selectbox("Select", ["Diabetes", "California Housing"])
    if st.sidebar.button("Load"):
        st.session_state.df_main = st.session_state.dataset_manager.load_sklearn_dataset(d_name)
elif data_source == "Create Synthetic (Non-Linear)":
    n = st.sidebar.slider("Samples", 50, 500, 200)
    noise = st.sidebar.slider("Noise", 0, 5, 1)
    if st.sidebar.button("Generate"):
        st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(n, noise)
elif data_source == "Upload CSV":
    up_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up_file:
        st.session_state.df_main = pd.read_csv(up_file)

if st.session_state.df_main is not None:
    df = st.session_state.df_main
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.subheader("Variables")
        target = st.selectbox("Target (y)", df.columns, index=len(df.columns)-1)
        features = st.multiselect("Features (X)", [c for c in df.columns if c != target], default=[df.columns[0]])
    
    with col_d2:
        st.subheader("Preview")
        st.dataframe(df.head(3), use_container_width=True)

    if features:
        X = df[features].values
        y = df[target].values
        
        test_pct = st.sidebar.slider("Test Data %", 10, 90, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
        
        st.markdown("---")
        
        st.header("Algorithm Selection")
        
        algo_type = st.radio(
            "Choose Approach:", 
            ["Linear Regression (Normal Equation - Exact)", "Gradient Descent (Iterative - Animation)"],
            horizontal=True
        )
        
        c_poly1, c_poly2 = st.columns(2)
        reg_type = c_poly1.radio("Regression Type", ["Simple/Multiple (Linear)", "Polynomial (Curve)"])
        degree = 1
        if reg_type == "Polynomial (Curve)":
            degree = c_poly2.number_input("Polynomial Degree", 2, 10, 2)
            is_poly = True
        else:
            is_poly = False

        model = MyLinearRegression()

        if "Normal Equation" in algo_type:
            st.info("The Normal Equation solves for weights directly:  $W = (X^T X)^{-1} X^T y$")
            
            if st.button("Solve & Calculate Metrics"):
                matrices = model.fit_normal_equation(X_train, y_train, is_polynomial=is_poly, degree=degree)
                
                y_pred = model.predict(X_test, is_polynomial=is_poly)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                st.subheader("Model Performance (Test Set)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R2 Score (Accuracy)", f"{r2:.4f}")
                m2.metric("MSE (Mean Sq Error)", f"{mse:.4f}")
                m3.metric("RMSE (Root MSE)", f"{rmse:.4f}")
                m4.metric("MAE (Mean Abs Error)", f"{mae:.4f}")
                
                st.markdown("---")
                st.subheader("Fit Visualization")
                
                if len(features) == 1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    ax.scatter(X_train, y_train, color='orange', alpha=0.5, label='Training Data', s=30)
                    ax.scatter(X_test, y_test, color='gray', alpha=0.7, label='Testing Data', marker='x', s=40)
                    
                    x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                    y_range = model.predict(x_range, is_polynomial=is_poly)
                    
                    ax.plot(x_range, y_range, color='#4F8BF9', linewidth=3, label='Model Prediction')
                    
                    ax.set_title(f"Regression Fit (Degree {degree})")
                    ax.set_xlabel("Feature X")
                    ax.set_ylabel("Target y")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(y_test, y_pred, color='#4F8BF9', alpha=0.6)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

        else:
            c1, c2, c3, c4 = st.columns(4)
            gd_variant = c1.selectbox("GD Variant", ["Batch", "Mini-Batch", "Stochastic"])
            lr = c2.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
            epochs = c3.number_input("Epochs", 10, 1000, 50)
            batch_s = 32
            if gd_variant == "Mini-Batch":
                batch_s = c4.slider("Batch Size", 1, len(X_train), 32)
            
            st.markdown("### Execution Controls")
            
            c_anim1, c_anim2 = st.columns(2)
            show_anim = c_anim1.checkbox("Show Live Animation", value=True)
            speed = c_anim2.slider("Animation Speed (Delay)", 0.0, 0.5, 0.01, disabled=not show_anim)
            
            if st.button("Start Training"):
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                
                if show_anim:
                    col_anim1, col_anim2 = st.columns(2)
                    with col_anim1:
                        plot_placeholder = st.empty()
                    with col_anim2:
                        loss_placeholder = st.empty()
                
                progress = st.progress(0)
                
                final_history = []
                final_weights = None
                
                for ep, loss, history, w in model.fit_gd_stream(
                    X_train_s, y_train, lr, epochs, gd_variant, batch_s, is_poly, degree
                ):
                    final_history = history
                    final_weights = w
                    
                    if show_anim:
                        fig_l, ax_l = plt.subplots(figsize=(5, 3))
                        ax_l.plot(history, 'r-')
                        ax_l.set_title(f"Loss vs Epochs (Ep: {ep})")
                        loss_placeholder.pyplot(fig_l)
                        plt.close(fig_l)
                        
                        if len(features) == 1:
                            fig_f, ax_f = plt.subplots(figsize=(5, 3))
                            ax_f.scatter(X_train, y_train, color='blue', alpha=0.3, s=10)
                            
                            x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                            if is_poly and model.poly:
                                 x_line_poly = model.poly.transform(x_line)
                                 x_line_poly_b = np.c_[np.ones((100, 1)), x_line_poly]
                                 y_line = x_line_poly_b.dot(w)
                            else:
                                 x_line_b = np.c_[np.ones((100, 1)), x_line]
                                 y_line = x_line_b.dot(w)
                                 
                            x_line_orig = scaler.inverse_transform(x_line)
                            ax_f.plot(x_line_orig, y_line, 'k-', lw=2)
                            ax_f.set_title("Live Fit Adjustment")
                            plot_placeholder.pyplot(fig_f)
                            plt.close(fig_f)

                        time.sleep(speed)
                        
                    progress.progress((ep+1)/epochs)

                st.success("Training Finished!")
                
                if not show_anim:
                    st.subheader("Training Summary")
                    c_static1, c_static2 = st.columns(2)
                    
                    with c_static1:
                        st.markdown("**Training Loss**")
                        fig_l, ax_l = plt.subplots(figsize=(5, 3))
                        ax_l.plot(final_history, 'r-')
                        ax_l.set_xlabel("Epochs")
                        ax_l.set_ylabel("Loss")
                        st.pyplot(fig_l)
                        
                    with c_static2:
                        st.markdown("**Final Fit Visualization**")
                        if len(features) == 1:
                            fig_f, ax_f = plt.subplots(figsize=(5, 3))
                            ax_f.scatter(X_train, y_train, color='blue', alpha=0.3, s=10, label="Train Data")
                            
                            x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                            if is_poly and model.poly:
                                 x_line_poly = model.poly.transform(x_line)
                                 x_line_poly_b = np.c_[np.ones((100, 1)), x_line_poly]
                                 y_line = x_line_poly_b.dot(final_weights)
                            else:
                                 x_line_b = np.c_[np.ones((100, 1)), x_line]
                                 y_line = x_line_b.dot(final_weights)
                                 
                            x_line_orig = scaler.inverse_transform(x_line)
                            ax_f.plot(x_line_orig, y_line, 'k-', lw=2, label="Model")
                            ax_f.legend()
                            st.pyplot(fig_f)
                        else:
                            st.info("Fit visualization available for 1D data only.")

                X_test_scaled = scaler.transform(X_test)
                y_pred_gd = model.predict(X_test_scaled, is_polynomial=is_poly)
                
                r2 = r2_score(y_test, y_pred_gd)
                mse = mean_squared_error(y_test, y_pred_gd)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_gd)
                
                st.markdown("---")
                st.subheader("Final Model Performance (Test Set)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R2 Score", f"{r2:.4f}")
                m2.metric("MSE", f"{mse:.4f}")
                m3.metric("RMSE", f"{rmse:.4f}")
                m4.metric("MAE", f"{mae:.4f}")

else:
    st.info("Please load a dataset to begin.")
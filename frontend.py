import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from backend import MyLinearRegression, MyLogisticRegression, DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

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

tab1, tab2 = st.tabs(["Dashboard", "Know the Mathematics"])

with tab1:
    st.sidebar.header("Data Configuration")
    data_source = st.sidebar.radio("Source", ["Sklearn Dataset", "Create Synthetic", "Upload CSV"])

    if data_source == "Sklearn Dataset":
        d_name = st.sidebar.selectbox("Select", ["Diabetes (Reg)", "California Housing (Reg)", "Iris (Binary Class)", "Breast Cancer (Class)"])
        if st.sidebar.button("Load"):
            st.session_state.df_main = st.session_state.dataset_manager.load_sklearn_dataset(d_name)
            
    elif data_source == "Create Synthetic":
        d_type = st.sidebar.selectbox("Type", ["Regression", "Classification"])
        n = st.sidebar.slider("Samples", 50, 500, 200)
        n_feats = st.sidebar.slider("Total Features", 1, 10, 2)
        n_info = st.sidebar.slider("Informative Features (Signal)", 1, n_feats, 1)
        
        if d_type == "Regression":
            noise = st.sidebar.slider("Noise Level", 0, 10, 2)
        else:
            noise = st.sidebar.slider("Difficulty (Overlap)", 0, 10, 2)
        
        if st.sidebar.button("Generate"):
            type_key = 'regression' if d_type == "Regression" else 'classification'
            st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                n_samples=n, noise=noise, type=type_key, n_features=n_feats, n_informative=n_info
            )

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
            all_feats = [c for c in df.columns if c != target]
            features = st.multiselect("Features (X)", all_feats, default=all_feats)
        
        with col_d2:
            st.subheader("Preview")
            st.dataframe(df.head(3), use_container_width=True)

        if features:
            X = df[features].values
            y = df[target].values
            
            unique_targets = len(np.unique(y))
            problem_type = "Classification" if unique_targets <= 5 else "Regression"
            
            st.info(f"Detected Problem Type: {problem_type} (Features: {len(features)})")
            
            test_pct = st.sidebar.slider("Test Data %", 10, 90, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
            
            st.markdown("---")
            st.header("Algorithm Selection")
            
            if problem_type == "Regression":
                algo_type = st.radio(
                    "Choose Approach:", 
                    ["Linear Regression (Normal Equation)", "Gradient Descent (Iterative)"],
                    horizontal=True
                )
                
                c_poly1, c_poly2 = st.columns(2)
                reg_type = c_poly1.radio("Feature Type", ["Linear", "Polynomial"])
                degree = 1
                if reg_type == "Polynomial":
                    degree = c_poly2.number_input("Degree", 2, 10, 2)
                    is_poly = True
                else:
                    is_poly = False
                
                model = MyLinearRegression()
                
                if "Normal Equation" in algo_type:
                    reg_mode = st.selectbox("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"])
                    
                    if "Lasso" in reg_mode:
                        st.error("🚫 Lasso (L1) cannot be solved using the Normal Equation (Exact Matrix Solution).")
                        st.info("👉 Please switch the 'Choose Approach' option above to **Gradient Descent** to use Lasso.")
                    else:
                        alpha = 0.0
                        if "Ridge" in reg_mode:
                            alpha = st.number_input("Alpha (Lambda)", 0.01, 100.0, 1.0)

                        if st.button("Solve"):
                            model.fit_normal_equation(
                                X_train, y_train, is_polynomial=is_poly, degree=degree, 
                                reg_type=reg_mode, alpha=alpha
                            )
                            y_pred = model.predict(X_test, is_polynomial=is_poly)
                            
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            st.subheader("Performance Metrics (Test Set)")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("R2 Score", f"{r2:.4f}")
                            m2.metric("MSE", f"{mse:.4f}")
                            m3.metric("RMSE", f"{rmse:.4f}")
                            m4.metric("MAE", f"{mae:.4f}")
                            
                            if len(features) == 1:
                                st.subheader("Model Fit Visualization")
                                c_train, c_test = st.columns(2)
                                
                                x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                                y_range = model.predict(x_range, is_polynomial=is_poly)
                                
                                with c_train:
                                    st.markdown("**Training Data Fit**")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.scatter(X_train, y_train, color='orange', alpha=0.5, label='Training')
                                    ax.plot(x_range, y_range, color='black', linewidth=2, label='Model')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    
                                with c_test:
                                    st.markdown("**Testing Data Fit**")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.scatter(X_test, y_test, color='green', alpha=0.6, label='Testing')
                                    ax.plot(x_range, y_range, color='black', linewidth=2, label='Model')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                            else:
                                st.subheader("Actual vs Predicted")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
                                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
                                ax.set_xlabel("Actual Values")
                                ax.set_ylabel("Predicted Values")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            
                else:
                    c1, c2, c3 = st.columns(3)
                    lr = c1.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
                    epochs = c2.number_input("Epochs", 10, 2000, 100)
                    gd_variant = c3.selectbox("GD Type", ["Batch", "Mini-Batch", "Stochastic"])
                    
                    c_reg1, c_reg2 = st.columns(2)
                    reg_mode_gd = c_reg1.selectbox("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"])
                    alpha_gd = 0.0
                    reg_key_gd = 'None'
                    
                    if "Ridge" in reg_mode_gd:
                        reg_key_gd = 'Ridge (L2)'
                        alpha_gd = c_reg2.number_input("Alpha (Lambda)", 0.01, 100.0, 1.0)
                    elif "Lasso" in reg_mode_gd:
                        reg_key_gd = 'Lasso (L1)'
                        alpha_gd = c_reg2.number_input("Alpha (Lambda)", 0.01, 100.0, 1.0)
                    
                    show_anim = st.checkbox("Show Animation", value=True)
                    speed = st.slider("Speed", 0.0, 0.5, 0.01, disabled=not show_anim)
                    
                    if st.button("Start Training"):
                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train)
                        
                        col_anim1, col_anim2 = st.columns(2)
                        with col_anim1: plot_placeholder = st.empty()
                        with col_anim2: loss_placeholder = st.empty()
                        progress = st.progress(0)
                        
                        final_w = None
                        final_history = []
                        
                        for ep, loss, history, w in model.fit_gd_stream(
                            X_train_s, y_train, lr, epochs, gd_variant, 32, is_poly, degree,
                            reg_type=reg_key_gd, alpha=alpha_gd
                        ):
                            final_w = w
                            final_history = history
                            if show_anim:
                                fig_l, ax_l = plt.subplots(figsize=(5, 3))
                                ax_l.plot(history, 'r-', label='Loss')
                                ax_l.set_title("Training Loss")
                                ax_l.legend()
                                ax_l.grid(True, alpha=0.3)
                                loss_placeholder.pyplot(fig_l)
                                plt.close(fig_l)
                                
                                if len(features) == 1:
                                    fig_f, ax_f = plt.subplots(figsize=(5, 3))
                                    ax_f.scatter(X_train, y_train, color='blue', alpha=0.3)
                                    x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                                    if is_poly and model.poly:
                                        x_line_poly = model.poly.transform(x_line)
                                        x_line_poly_b = np.c_[np.ones((100, 1)), x_line_poly]
                                        y_line = x_line_poly_b.dot(w)
                                    else:
                                        x_line_b = np.c_[np.ones((100, 1)), x_line]
                                        y_line = x_line_b.dot(w)
                                    x_orig = scaler.inverse_transform(x_line)
                                    ax_f.plot(x_orig, y_line, 'k-')
                                    plot_placeholder.pyplot(fig_f)
                                    plt.close(fig_f)
                                time.sleep(speed)
                            progress.progress((ep+1)/epochs)
                        
                        X_test_s = scaler.transform(X_test)
                        y_pred = model.predict(X_test_s, is_polynomial=is_poly)
                        
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        st.subheader("Final Performance Metrics")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("R2", f"{r2:.4f}")
                        c2.metric("MSE", f"{mse:.4f}")
                        c3.metric("RMSE", f"{rmse:.4f}")
                        c4.metric("MAE", f"{mae:.4f}")

                        st.subheader("Training Summary")
                        st.markdown("**Loss History**")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(final_history, 'r-', label='Training Loss')
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        if len(features) == 1:
                            st.subheader("Model Fit Visualization")
                            c_train, c_test = st.columns(2)
                            
                            x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                            if is_poly and model.poly:
                                 x_line_poly = model.poly.transform(x_line)
                                 x_line_poly_b = np.c_[np.ones((100, 1)), x_line_poly]
                                 y_line = x_line_poly_b.dot(final_w)
                            else:
                                 x_line_b = np.c_[np.ones((100, 1)), x_line]
                                 y_line = x_line_b.dot(final_w)
                            x_orig = scaler.inverse_transform(x_line)
                            
                            with c_train:
                                st.markdown("**Training Data Fit**")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.scatter(X_train, y_train, color='orange', alpha=0.3, label="Train Data")
                                ax.plot(x_orig, y_line, 'k-', lw=2, label="Model")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                            with c_test:
                                st.markdown("**Testing Data Fit**")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.scatter(X_test, y_test, color='green', alpha=0.6, label="Test Data")
                                ax.plot(x_orig, y_line, 'k-', lw=2, label="Model")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                        else:
                            st.markdown("**Actual vs Predicted**")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
                            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
                            ax.set_xlabel("Actual")
                            ax.set_ylabel("Predicted")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)

            else:
                class_algo = st.radio("Algorithm", 
                    ["Perceptron Trick", "Sigmoid Perceptron (SGD)", "Logistic Regression (Batch)"]
                )
                
                c1, c2 = st.columns(2)
                lr = c1.number_input("Learning Rate", 0.0001, 10.0, 0.1, format="%.4f")
                epochs = c2.number_input("Epochs", 10, 5000, 500)
                
                show_anim = st.checkbox("Show Animation", value=True)
                speed = st.slider("Speed", 0.0, 0.5, 0.01, disabled=not show_anim)
                
                if st.button("Start Training"):
                    model = MyLogisticRegression()
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    col_anim1, col_anim2 = st.columns(2)
                    with col_anim1: plot_placeholder = st.empty()
                    with col_anim2: loss_placeholder = st.empty()
                    progress = st.progress(0)
                    
                    stream = None
                    if "Perceptron Trick" in class_algo:
                        stream = model.fit_perceptron_trick(X_train_s, y_train, lr, epochs)
                    elif "Sigmoid" in class_algo:
                        stream = model.fit_sigmoid_perceptron(X_train_s, y_train, lr, epochs)
                    else:
                        stream = model.fit_batch_logistic(X_train_s, y_train, lr, epochs)
                    
                    final_w = None
                    final_hist = []
                    
                    for ep, loss, history, w in stream:
                        final_w = w
                        final_hist = history
                        
                        if show_anim and ep % 5 == 0:
                            fig_l, ax_l = plt.subplots(figsize=(5, 3))
                            ax_l.plot(history, 'r-', label='Loss')
                            ax_l.set_title("Loss")
                            ax_l.legend()
                            ax_l.grid(True, alpha=0.3)
                            loss_placeholder.pyplot(fig_l)
                            plt.close(fig_l)
                            
                            if len(features) == 1:
                                fig_f, ax_f = plt.subplots(figsize=(5, 3))
                                ax_f.scatter(X_train_s, y_train, c=y_train, cmap='bwr', alpha=0.6)
                                x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                                z = w[0] + w[1] * x_line
                                y_line = 1 / (1 + np.exp(-z))
                                ax_f.plot(x_line, y_line, 'k-', lw=2)
                                plot_placeholder.pyplot(fig_f)
                                plt.close(fig_f)

                            elif len(features) == 2:
                                fig_f, ax_f = plt.subplots(figsize=(5, 3))
                                ax_f.scatter(X_train_s[:,0], X_train_s[:,1], c=y_train, cmap='bwr', alpha=0.6)
                                x_min, x_max = X_train_s[:,0].min()-0.5, X_train_s[:,0].max()+0.5
                                if w[2] != 0:
                                    x_vals = np.array([x_min, x_max])
                                    y_vals = -(w[1] * x_vals + w[0]) / w[2]
                                    ax_f.plot(x_vals, y_vals, 'k--', lw=2)
                                    ax_f.set_ylim(X_train_s[:,1].min()-1, X_train_s[:,1].max()+1)
                                plot_placeholder.pyplot(fig_f)
                                plt.close(fig_f)
                                
                            time.sleep(speed)
                        progress.progress((ep+1)/epochs)

                    y_pred = model.predict(X_test_s)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    st.success("Training Finished")
                    st.subheader("Classification Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.4f}")
                    m2.metric("Precision", f"{prec:.4f}")
                    m3.metric("Recall", f"{rec:.4f}")
                    m4.metric("F1 Score", f"{f1:.4f}")
                    
                    st.subheader("Training Summary")
                    st.markdown("**Loss History**")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(final_hist, label='Training Loss')
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Loss")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    st.subheader("Model Decision Boundary (Train vs Test)")
                    
                    if len(features) == 2:
                        c_train, c_test = st.columns(2)
                        
                        with c_train:
                            st.markdown("**Training Data**")
                            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
                            ax_tr.scatter(X_train_s[:,0], X_train_s[:,1], c=y_train, cmap='bwr', label='Train Points')
                            
                            x_min, x_max = X_train_s[:,0].min()-1, X_train_s[:,0].max()+1
                            if final_w[2] != 0:
                                x_vals = np.array([x_min, x_max])
                                y_vals = -(final_w[1] * x_vals + final_w[0]) / final_w[2]
                                ax_tr.plot(x_vals, y_vals, 'k--', lw=2, label='Boundary')
                            
                            ax_tr.set_ylim(X_train_s[:,1].min()-1, X_train_s[:,1].max()+1)
                            ax_tr.set_title("Train Fit")
                            ax_tr.legend()
                            st.pyplot(fig_tr)
                            
                        with c_test:
                            st.markdown("**Testing Data**")
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                            ax_ts.scatter(X_test_s[:,0], X_test_s[:,1], c=y_test, cmap='bwr', marker='x', label='Test Points')
                            
                            x_min_t, x_max_t = X_test_s[:,0].min()-1, X_test_s[:,0].max()+1
                            if final_w[2] != 0:
                                x_vals_t = np.array([x_min_t, x_max_t])
                                y_vals_t = -(final_w[1] * x_vals_t + final_w[0]) / final_w[2]
                                ax_ts.plot(x_vals_t, y_vals_t, 'k--', lw=2, label='Boundary')
                                
                            ax_ts.set_ylim(X_test_s[:,1].min()-1, X_test_s[:,1].max()+1)
                            ax_ts.set_title("Test Fit")
                            ax_ts.legend()
                            st.pyplot(fig_ts)

                    elif len(features) == 1:
                        c_train, c_test = st.columns(2)
                        
                        with c_train:
                            st.markdown("**Training Data**")
                            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
                            ax_tr.scatter(X_train_s, y_train, c=y_train, cmap='bwr', label='Train Points')
                            
                            x_line = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                            z = final_w[0] + final_w[1] * x_line
                            y_line = 1 / (1 + np.exp(-z))
                            ax_tr.plot(x_line, y_line, 'k-', lw=2, label='Sigmoid')
                            ax_tr.legend()
                            st.pyplot(fig_tr)
                            
                        with c_test:
                            st.markdown("**Testing Data**")
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                            ax_ts.scatter(X_test_s, y_test, c=y_test, cmap='bwr', marker='x', label='Test Points')
                            
                            x_line_t = np.linspace(X_test_s.min(), X_test_s.max(), 100).reshape(-1, 1)
                            z_t = final_w[0] + final_w[1] * x_line_t
                            y_line_t = 1 / (1 + np.exp(-z_t))
                            ax_ts.plot(x_line_t, y_line_t, 'k-', lw=2, label='Sigmoid')
                            ax_ts.legend()
                            st.pyplot(fig_ts)
                            
                    else:
                        st.markdown("**Confusion Matrix**")
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap='Blues', colorbar=False)
                        st.pyplot(fig_cm)

    else:
        st.info("Please load a dataset to begin.")

with tab2:
    st.title("Know the Mathematics")
    
    st.header("1. Simple Linear Regression")
    st.markdown("""
    **Goal:** Find the best fit line $\hat{y} = wx + b$ that minimizes error.
    
    **Hypothesis Formulation:**
    $$ h_\\theta(x) = \\theta_0 + \\theta_1 x $$
    Where:
    * $\\theta_0$ is the bias (intercept) $b$.
    * $\\theta_1$ is the weight (slope) $w$.
    
    **Cost Function (Mean Squared Error - MSE):**
    $$ J(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 $$
    The goal is to minimize $J(\\theta)$.
    """)
    

    st.divider()
    
    st.header("2. Normal Equation (Closed Form Solution)")
    st.markdown("""
    Instead of iterating, we can solve for $\\theta$ mathematically in one step by setting the derivative of the cost function to zero.
    
    **Derivation:**
    1. Matrix notation for hypothesis: $Y = X\\theta$
    2. Cost function in matrix form:
       $$ J(\\theta) = (X\\theta - y)^T (X\\theta - y) $$
    3. Take derivative with respect to $\\theta$ and set to 0:
       $$ \\nabla_\\theta J(\\theta) = 2X^T X \\theta - 2X^T y = 0 $$
    4. Solve for $\\theta$:
       $$ X^T X \\theta = X^T y $$
       $$ \\theta = (X^T X)^{-1} X^T y $$
    
    *Note: This works best for smaller datasets where calculating the inverse of $(X^T X)$ is computationally feasible.*
    """)
    

    st.divider()
    
    st.header("3. Gradient Descent (Iterative Solution)")
    st.markdown("""
    When data is huge, finding the inverse matrix is slow. Gradient Descent "walks" down the error hill step-by-step.
    
    **Update Rule:**
    Repeat until convergence:
    $$ \\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta) $$
    
    **Derivative Calculation:**
    $$ \\frac{\\partial}{\\partial \\theta_j} J(\\theta) = \\frac{2}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} $$
    
    So the update step becomes:
    $$ \\theta = \\theta - \\alpha \\cdot \\frac{2}{m} X^T (X\\theta - y) $$
    """)
    

    st.divider()
    
    st.header("4. Ridge Regression (L2 Regularization)")
    st.markdown("""
    **Goal:** Prevent overfitting by punishing large weights. We add a "penalty" term to the cost function.
    
    **Cost Function:**
    $$ J(\\theta) = \\text{MSE} + \\lambda \\sum_{j=1}^{n} \\theta_j^2 $$
    (Note: We do not penalize the bias term $\\theta_0$).
    
    **Gradient Update Rule:**
    $$ \\theta_j := \\theta_j - \\alpha [ \\text{Original Gradient} + 2\\lambda \\theta_j ] $$
    
    **Normal Equation for Ridge:**
    $$ \\theta = (X^T X + \\lambda I)^{-1} X^T y $$
    Where $I$ is the Identity Matrix (with 0 at index [0,0] to ignore bias).
    """)
    

    st.divider()
    
    st.header("5. Lasso Regression (L1 Regularization)")
    st.markdown("""
    **Goal:** Shrink coefficients to exactly zero (Feature Selection).
    
    **Cost Function:**
    $$ J(\\theta) = \\text{MSE} + \\lambda \\sum_{j=1}^{n} |\\theta_j| $$
    
    **Gradient Issue:**
    The absolute value function $|x|$ is essentially a V-shape. It is not differentiable at exactly 0.
    
    **Sub-Gradient Update Rule:**
    $$ \\theta_j := \\theta_j - \\alpha [ \\text{Original Gradient} + \\lambda \\cdot \\text{sign}(\\theta_j) ] $$
    """)
    

    st.divider()
    
    st.header("6. Logistic Regression")
    st.markdown("""
    **Goal:** Classification (0 or 1). We squash the linear output into a probability between 0 and 1.
    
    **Sigmoid Function:**
    $$ g(z) = \\frac{1}{1 + e^{-z}} $$
    
    **Hypothesis:**
    $$ h_\\theta(x) = g(\\theta^T x) $$
    
    **Cost Function (Log Loss):**
    We cannot use MSE (it makes the curve "wavy" and non-convex). We use Log Loss:
    $$ J(\\theta) = -\\frac{1}{m} \\sum [ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1 - h_\\theta(x^{(i)})) ] $$
    
    **Gradient Descent Update:**
    Surprisingly, the derivative ends up looking exactly like Linear Regression:
    $$ \\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m} X^T (h_\\theta(x) - y) $$
    """)
    

    st.divider()
    
    st.header("7. Perceptron Trick")
    st.markdown("""
    The simplest classification algorithm. It doesn't use probabilities, just a hard cutoff.
    
    **Decision Boundary:**
    $$ \\hat{y} = 1 \\quad \\text{if } \\theta^T x \\ge 0 \\quad \\text{else } 0 $$
    
    **Update Rule (Only on Misclassification):**
    1. Pick a random point $(x, y)$.
    2. If prediction is correct, do nothing.
    3. If prediction is wrong:
       * If actual is 1 but predicted 0: **Add** vector $x$ to weights.
       * If actual is 0 but predicted 1: **Subtract** vector $x$ from weights.
       
    $$ \\theta := \\theta + \\alpha (y - \\hat{y}) x $$
    """)
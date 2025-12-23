import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from backend import MyLinearRegression, MyLogisticRegression, DatasetManager, MyDecisionTree, MyVotingEnsemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

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
        d_name = st.sidebar.selectbox("Select", ["Diabetes (Reg)", "California Housing (Reg)", "Iris (Binary Class)", "Breast Cancer (Class)", "Iris (Multi-Class)", "Wine (Multi-Class)"])
        if st.sidebar.button("Load"):
            st.session_state.df_main = st.session_state.dataset_manager.load_sklearn_dataset(d_name)
            
    elif data_source == "Create Synthetic":
        d_type = st.sidebar.selectbox("Type", ["Regression", "Classification"])
        n = st.sidebar.slider("Samples", 50, 500, 200)
        n_feats = st.sidebar.slider("Total Features", 1, 10, 2)
        n_info = st.sidebar.slider("Informative Features (Signal)", 1, n_feats, 1)
        
        if d_type == "Regression":
            noise = st.sidebar.slider("Noise Level", 0, 10, 2)
            if st.sidebar.button("Generate"):
                st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                    n_samples=n, noise=noise, type='regression', n_features=n_feats, n_informative=n_info
                )
        else:
            noise = st.sidebar.slider("Difficulty (Overlap)", 0, 10, 2)
            n_cls = st.sidebar.slider("Number of Classes", 2, 4, 2)
            if st.sidebar.button("Generate"):
                st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                    n_samples=n, noise=noise, type='classification', n_features=n_feats, n_informative=n_info, n_classes=n_cls
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
            
            st.info(f"Detected Problem Type: {problem_type} (Classes: {unique_targets}, Features: {len(features)})")
            
            test_pct = st.sidebar.slider("Test Data %", 10, 90, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
            
            st.markdown("---")
            st.header("Algorithm Selection")
            

            if problem_type == "Regression":
                algo_type = st.radio(
                    "Choose Approach:", 
                    ["Linear Regression (Normal Equation)", "Gradient Descent (Iterative)", "Decision Tree Regression", "Voting Regressor (Ensemble)"],
                    horizontal=True
                )
                
                model = None
                is_poly = False
                
            
                if "Decision Tree" in algo_type:
                    c1, c2, c3 = st.columns(3)
                    crit = c1.selectbox("Criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])
                    md = c2.number_input("Max Depth", 1, 20, 5)
                    mss = c3.number_input("Min Samples Split", 2, 20, 2)
                    
                    if st.button("Train Decision Tree"):
                        model = MyDecisionTree(mode='regression', max_depth=md, min_samples_split=mss, criterion=crit)
                        model.fit(X_train, y_train)
                        st.success("Decision Tree Trained")
                        st.text("Tree Structure Preview:")
                        st.text(model.get_text_tree(feature_names=features))

     
                elif "Voting" in algo_type:
                    st.info("Combines Linear Regression and Decision Tree")
                    if st.button("Train Ensemble"):
                        est = [('lr', LinearRegression()), ('dt', DecisionTreeRegressor(max_depth=5))]
                        model = MyVotingEnsemble(estimators=est)
                        model.fit(X_train, y_train, mode='regression')
                        st.success("Ensemble Trained")

       
                elif "Normal Equation" in algo_type:
                    model = MyLinearRegression()
                    c_poly1, c_poly2 = st.columns(2)
                    reg_type = c_poly1.radio("Feature Type", ["Linear", "Polynomial"])
                    degree = 1
                    if reg_type == "Polynomial":
                        degree = c_poly2.number_input("Degree", 2, 10, 2)
                        is_poly = True
                    
                    reg_mode = st.selectbox("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"])
                    alpha = 0.0
                    if "Lasso" in reg_mode:
                        st.error("🚫 Lasso (L1) cannot be solved using the Normal Equation.")
                        st.stop()
                    if "Ridge" in reg_mode: 
                        alpha = st.number_input("Alpha", 0.01, 100.0, 1.0)

                    if st.button("Solve"):
                        model.fit_normal_equation(X_train, y_train, is_polynomial=is_poly, degree=degree, reg_type=reg_mode, alpha=alpha)

               
                else:
                    model = MyLinearRegression()
                    c_poly1, c_poly2 = st.columns(2)
                    reg_type = c_poly1.radio("Feature Type", ["Linear", "Polynomial"])
                    degree = 1
                    if reg_type == "Polynomial":
                        degree = c_poly2.number_input("Degree", 2, 10, 2)
                        is_poly = True
                        
                    c1, c2, c3 = st.columns(3)
                    lr = c1.number_input("Learning Rate", 0.00001, 1.0, 0.001, format="%.5f")
                    epochs = c2.number_input("Epochs", 10, 2000, 100)
                    gd_variant = c3.selectbox("GD Type", ["Batch", "Mini-Batch", "Stochastic"])
                    
                    reg_mode_gd = st.selectbox("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"])
                    alpha_gd = 0.0
                    reg_key_gd = 'None'
                    if "Ridge" in reg_mode_gd: 
                        reg_key_gd = 'Ridge (L2)'
                        alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                    elif "Lasso" in reg_mode_gd: 
                        reg_key_gd = 'Lasso (L1)'
                        alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                    
                    if st.button("Start Training"):
                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train)
                        final_hist = []
                        prog = st.progress(0)
                        
                        for ep, loss, history, w in model.fit_gd_stream(
                            X_train_s, y_train, lr, epochs, gd_variant, 32, is_poly, degree, reg_type=reg_key_gd, alpha=alpha_gd
                        ):
                            final_hist = history
                            if ep % 10 == 0: prog.progress((ep+1)/epochs)
                        
                        st.subheader("Training Loss")
                        st.line_chart(final_hist)
                
            
                if model:
                    
                    y_pred = None
                    if "Gradient" in algo_type:
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                        X_test_in = scaler.transform(X_test)
                        if is_poly:
                            y_pred = model.predict(X_test_in, is_polynomial=True)
                        else:
                            y_pred = model.predict(X_test_in)
                    else:
                        if is_poly and isinstance(model, MyLinearRegression):
                            y_pred = model.predict(X_test, is_polynomial=True)
                        else:
                            y_pred = model.predict(X_test)
                    
                   
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.subheader("Performance Metrics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R2 Score", f"{r2:.4f}")
                    c2.metric("MSE", f"{mse:.4f}")
                    c3.metric("RMSE", f"{rmse:.4f}")
                    c4.metric("MAE", f"{mae:.4f}")
                    
                    if len(features) == 1:
                        st.subheader("Model Fit Visualization")
                        c_train, c_test = st.columns(2)
                        
                     
                        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                        y_range = None
                        
                        
                        if "Gradient" in algo_type:
                            x_range_s = scaler.transform(x_range)
                            if is_poly: y_range = model.predict(x_range_s, is_polynomial=True)
                            else: y_range = model.predict(x_range_s)
                        elif is_poly and isinstance(model, MyLinearRegression):
                            y_range = model.predict(x_range, is_polynomial=True)
                        else:
                            y_range = model.predict(x_range)

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
                        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                        ax.set_xlabel("Actual Values")
                        ax.set_ylabel("Predicted Values")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)


            else:
                
                if unique_targets > 2:
                    st.info("Multi-class detected.")
                    algo_list = ["Softmax Regression", "Decision Tree Classifier", "Voting Classifier"]
                else:
                    algo_list = ["Perceptron", "Logistic Regression", "Decision Tree Classifier", "Voting Classifier"]
                    
                class_algo = st.radio("Algorithm", algo_list)
                
                model = None
                is_poly = False
                
               
                if "Decision Tree" in class_algo:
                    c1, c2, c3 = st.columns(3)
                    crit = c1.selectbox("Criterion", ["gini", "entropy", "log_loss"])
                    md = c2.number_input("Max Depth", 1, 20, 5)
                    mss = c3.number_input("Min Samples Split", 2, 20, 2)
                    
                    if st.button("Train Tree"):
                        model = MyDecisionTree(mode='classification', max_depth=md, min_samples_split=mss, criterion=crit)
                        model.fit(X_train, y_train)
                        st.success("Tree Trained")
                        st.text("Tree Rules:")
                        st.text(model.get_text_tree(feature_names=features))

              
                elif "Voting" in class_algo:
                    v_type = st.radio("Voting Type", ["hard", "soft"])
                    if st.button("Train Ensemble"):
                        est = [('log', LogisticRegression()), ('dt', DecisionTreeClassifier(max_depth=5))]
                        model = MyVotingEnsemble(estimators=est, voting_type=v_type)
                        model.fit(X_train, y_train, mode='classification')
                        st.success("Ensemble Trained")

               
                else:
                    c_poly1, c_poly2 = st.columns(2)
                    reg_type_feat = c_poly1.radio("Feature Type", ["Linear", "Polynomial"])
                    degree = 1
                    if reg_type_feat == "Polynomial":
                        degree = c_poly2.number_input("Degree", 2, 10, 2)
                        is_poly = True

                    c1, c2, c3 = st.columns(3)
                    lr = c1.number_input("Learning Rate", 0.0001, 10.0, 0.1, format="%.4f")
                    epochs = c2.number_input("Epochs", 10, 5000, 500)
                    
                   
                    reg_mode_gd = c3.selectbox("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"])
                    alpha_gd = 0.0
                    reg_key_gd = 'None'
                    if "Ridge" in reg_mode_gd: 
                        reg_key_gd = 'Ridge (L2)'
                        alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                    elif "Lasso" in reg_mode_gd: 
                        reg_key_gd = 'Lasso (L1)'
                        alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                    
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
                        if "Perceptron" in class_algo:
                            stream = model.fit_perceptron_trick(X_train_s, y_train, lr, epochs)
                        elif "Logistic" in class_algo:
                            stream = model.fit_batch_logistic(X_train_s, y_train, lr, epochs, is_polynomial=is_poly, degree=degree, reg_type=reg_key_gd, alpha=alpha_gd)
                        elif "Softmax" in class_algo:
                            stream = model.fit_softmax(X_train_s, y_train, lr, epochs, is_polynomial=is_poly, degree=degree, reg_type=reg_key_gd, alpha=alpha_gd)
                            
                        final_hist = []
                        for ep, loss, history, w in stream:
                            final_hist = history
                            if show_anim and ep % 5 == 0:
                                fig_l, ax_l = plt.subplots(figsize=(5, 3))
                                ax_l.plot(history, 'r-')
                                ax_l.set_title("Loss")
                                ax_l.grid(True, alpha=0.3)
                                loss_placeholder.pyplot(fig_l)
                                plt.close(fig_l)
                                
                                # Live Boundary Plot (2D only)
                                if len(features) == 2:
                                    fig_f, ax_f = plt.subplots(figsize=(5, 3))
                                    x_min, x_max = X_train_s[:,0].min()-1, X_train_s[:,0].max()+1
                                    y_min, y_max = X_train_s[:,1].min()-1, X_train_s[:,1].max()+1
                                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                                    mesh_input = np.c_[xx.ravel(), yy.ravel()]
                                    
                                    Z = model.predict(mesh_input, is_polynomial=is_poly)
                                    Z = Z.reshape(xx.shape)
                                    
                                    ax_f.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                                    ax_f.scatter(X_train_s[:,0], X_train_s[:,1], c=y_train, cmap='viridis', edgecolors='k', alpha=0.6)
                                    ax_f.set_title("Decision Boundary")
                                    plot_placeholder.pyplot(fig_f)
                                    plt.close(fig_f)
                                time.sleep(speed)
                            progress.progress((ep+1)/epochs)
                        
                        st.line_chart(final_hist)

               
                if model:
                    y_pred = None
                    
                    if "Tree" in class_algo or "Voting" in class_algo:
                        y_pred = model.predict(X_test)
                    else:
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                        X_test_s = scaler.transform(X_test)
                        y_pred = model.predict(X_test_s, is_polynomial=is_poly)

                   
                    acc = accuracy_score(y_test, y_pred)
                    avg_method = 'weighted' if unique_targets > 2 else 'binary'
                    prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    st.subheader("Classification Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.4f}")
                    m2.metric("Precision", f"{prec:.4f}")
                    m3.metric("Recall", f"{rec:.4f}")
                    m4.metric("F1 Score", f"{f1:.4f}")
                    
                    st.subheader("Model Decision Boundary (Train vs Test)")
                    
                    
                    if len(features) == 2:
                        c_train, c_test = st.columns(2)
                        
                        def plot_boundary(ax, X_data, y_data, title, is_scaled=False):
                            # Create mesh
                            x_min, x_max = X_data[:,0].min()-1, X_data[:,0].max()+1
                            y_min, y_max = X_data[:,1].min()-1, X_data[:,1].max()+1
                            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                            mesh_input = np.c_[xx.ravel(), yy.ravel()]
                            
                            # Prediction logic
                            Z = None
                            if "Tree" in class_algo or "Voting" in class_algo:
                                Z = model.predict(mesh_input)
                            else:
                                if not is_scaled:
                                    # If data passed is raw, scale it for prediction
                                    scaler = StandardScaler()
                                    scaler.fit(X_train) # Use train stats
                                    mesh_input = scaler.transform(mesh_input)
                                Z = model.predict(mesh_input, is_polynomial=is_poly)
                                
                            Z = Z.reshape(xx.shape)
                            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                            ax.scatter(X_data[:,0], X_data[:,1], c=y_data, cmap='viridis', edgecolors='k')
                            ax.set_title(title)

                        with c_train:
                            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
                            # For Linear models, if we trained on Scaled data, plot Scaled data
                            if "Tree" in class_algo or "Voting" in class_algo:
                                plot_boundary(ax_tr, X_train, y_train, "Train Fit")
                            else:
                                scaler = StandardScaler()
                                X_train_s = scaler.fit_transform(X_train)
                                plot_boundary(ax_tr, X_train_s, y_train, "Train Fit (Scaled)", is_scaled=True)
                            st.pyplot(fig_tr)
                            
                        with c_test:
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                            if "Tree" in class_algo or "Voting" in class_algo:
                                plot_boundary(ax_ts, X_test, y_test, "Test Fit")
                            else:
                                scaler = StandardScaler()
                                scaler.fit(X_train)
                                X_test_s = scaler.transform(X_test)
                                plot_boundary(ax_ts, X_test_s, y_test, "Test Fit (Scaled)", is_scaled=True)
                            st.pyplot(fig_ts)

                    # 1D VISUALIZATION (Only for simple Binary models)
                    elif len(features) == 1 and unique_targets == 2 and not is_poly and "Tree" not in class_algo and "Voting" not in class_algo:
                        c_train, c_test = st.columns(2)
                        
                        with c_train:
                            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
                            scaler = StandardScaler()
                            X_train_s = scaler.fit_transform(X_train)
                            ax_tr.scatter(X_train_s, y_train, c=y_train, cmap='viridis', label='Train')
                            x_l = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
                            y_l = model.predict_proba(x_l, is_polynomial=False)
                            ax_tr.plot(x_l, y_l, 'k-', lw=2)
                            st.pyplot(fig_tr)
                            
                        with c_test:
                            fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                            scaler.fit(X_train)
                            X_test_s = scaler.transform(X_test)
                            ax_ts.scatter(X_test_s, y_test, c=y_test, cmap='viridis', label='Test')
                            x_l = np.linspace(X_test_s.min(), X_test_s.max(), 100).reshape(-1, 1)
                            y_l = model.predict_proba(x_l, is_polynomial=False)
                            ax_ts.plot(x_l, y_l, 'k-', lw=2)
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
    **Goal:** Find $\hat{y} = wx + b$ that minimizes MSE.
    $$ J(\\theta) = \\frac{1}{m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 $$
    """)
    st.divider()
    
    st.header("2. Normal Equation")
    st.markdown("""
    **Closed Form Solution:**
    $$ \\theta = (X^T X)^{-1} X^T y $$
    """)
    st.divider()
    
    st.header("3. Gradient Descent")
    st.markdown("""
    **Iterative Update:**
    $$ \\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta) $$
    """)
    st.divider()
    
    st.header("4. Ridge Regression (L2)")
    st.markdown("""
    **Cost Function:**
    $$ J(\\theta) = \\text{MSE} + \\lambda \\sum \\theta_j^2 $$
    **Update Rule:**
    $$ \\theta_j := \\theta_j - \\alpha [ \\text{Gradient} + 2\\lambda \\theta_j ] $$
    """)
    st.divider()
    
    st.header("5. Lasso Regression (L1)")
    st.markdown("""
    **Cost Function:**
    $$ J(\\theta) = \\text{MSE} + \\lambda \\sum |\\theta_j| $$
    **Update Rule:**
    $$ \\theta_j := \\theta_j - \\alpha [ \\text{Gradient} + \\lambda \\cdot \\text{sign}(\\theta_j) ] $$
    """)
    st.divider()
    
    st.header("6. Logistic Regression")
    st.markdown("""
    **Sigmoid:**
    $$ g(z) = \\frac{1}{1 + e^{-z}} $$
    **Log Loss:**
    $$ J(\\theta) = -\\frac{1}{m} \\sum [ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1 - h_\\theta(x^{(i)})) ] $$
    """)
    st.divider()
    
    st.header("7. Softmax Regression (Multi-Class)")
    st.markdown("""
    **Goal:** Classify into $K$ classes.
    $$ P(y=k|x) = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}} $$
    """)
    st.divider()
    
    st.header("8. Decision Trees")
    st.markdown("""
    **Concept:** Recursively splits data based on features to create homogeneous groups.
    **Splitting Criteria (Classification):**
    * **Gini Impurity:** $1 - \\sum p_i^2$
    * **Entropy:** $- \\sum p_i \\log_2(p_i)$
    """)
    

    st.header("9. Ensemble Voting")
    st.markdown("""
    **Hard Voting:** Majority rule.
    **Soft Voting:** Average probabilities.
    """)
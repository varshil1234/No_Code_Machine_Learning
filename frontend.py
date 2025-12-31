import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from backend import MyLinearRegression, MyLogisticRegression, DatasetManager, MyDecisionTree, MyVotingEnsemble, MyEnsembleWrapper, MyKMeans, ModelTuner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, silhouette_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)

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
if 'tuner' not in st.session_state:
    st.session_state.tuner = ModelTuner()

# --- HELPER TO PARSE LIST INPUTS ---
def parse_range(input_str, type_func=int):
    try:
        return [type_func(x.strip()) for x in input_str.split(',')]
    except:
        return []

# --- TABS CONFIGURATION ---
tab1, tab2 = st.tabs(["Dashboard", "Know the Mathematics"])

with tab1:
    # --- SIDEBAR: DATA CONFIGURATION ---
    st.sidebar.header("Data Configuration")
    # Removed "Upload CSV" from the options list
    data_source = st.sidebar.radio("Source", ["Sklearn Dataset", "Create Synthetic"])

    if data_source == "Sklearn Dataset":
        d_name = st.sidebar.selectbox("Select", ["Diabetes (Reg)", "California Housing (Reg)", "Iris (Binary Class)", "Breast Cancer (Class)", "Iris (Multi-Class)", "Wine (Multi-Class)"])
        if st.sidebar.button("Load"):
            st.session_state.df_main = st.session_state.dataset_manager.load_sklearn_dataset(d_name)
            
    elif data_source == "Create Synthetic":
        d_type = st.sidebar.selectbox("Type", ["Regression", "Classification", "Clustering"])
        n = st.sidebar.slider("Samples", 50, 500, 200)
        n_feats = st.sidebar.slider("Total Features", 1, 10, 2)
        n_info = st.sidebar.slider("Informative Features (Signal)", 1, n_feats, 1)
        
        if d_type == "Regression":
            noise = st.sidebar.slider("Noise Level", 0, 10, 2)
            if st.sidebar.button("Generate"):
                st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                    n_samples=n, noise=noise, type='regression', n_features=n_feats, n_informative=n_info
                )
        elif d_type == "Classification":
            noise = st.sidebar.slider("Difficulty (Overlap)", 0, 10, 2)
            n_cls = st.sidebar.slider("Number of Classes", 2, 4, 2)
            if st.sidebar.button("Generate"):
                st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                    n_samples=n, noise=noise, type='classification', n_features=n_feats, n_informative=n_info, n_classes=n_cls
                )
        else: # Clustering
            noise = st.sidebar.slider("Cluster Spread (Std Dev)", 0, 10, 2)
            n_cls = st.sidebar.slider("Number of Clusters (Centers)", 2, 6, 3)
            if st.sidebar.button("Generate"):
                st.session_state.df_main = st.session_state.dataset_manager.create_synthetic_data(
                    n_samples=n, noise=noise, type='clustering', n_features=n_feats, n_informative=n_info, n_classes=n_cls
                )

    # --- MAIN DATA PROCESSING ---
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
            
            if data_source == "Create Synthetic" and d_type == "Clustering":
                problem_type = "Clustering"
            elif unique_targets <= 10:
                problem_type = "Classification"
            else:
                problem_type = "Regression"
            
            st.info(f"Detected Problem Type: {problem_type} (Classes: {unique_targets}, Features: {len(features)})")
            
            if problem_type != "Clustering":
                test_pct = st.sidebar.slider("Test Data %", 10, 90, 20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            st.markdown("---")
            st.header("Algorithm Selection")
            
            # ==========================
            # REGRESSION LOGIC
            # ==========================
            if problem_type == "Regression":
                algo_type = st.radio(
                    "Choose Approach:", 
                    [
                        "Linear Regression (Normal Equation)", "Gradient Descent (Iterative)", 
                        "Decision Tree Regression", "Voting Regressor (Ensemble)",
                        "Bagging Regressor", "Gradient Boosting Regressor", "AdaBoost Regressor"
                    ],
                    horizontal=True
                )
                
                model = None
                is_poly = False
                enable_tuning = st.checkbox("Enable Hyperparameter Tuning (GridSearch)")
                
                # --- TUNING LOGIC ---
                if enable_tuning:
                    st.subheader("Hyperparameter Configuration")
                    param_grid = {}
                    estimator = None
                    
                    if "Decision Tree" in algo_type:
                        md_str = st.text_input("Max Depth (comma sep)", "3, 5, 10")
                        param_grid['max_depth'] = parse_range(md_str)
                        mss_str = st.text_input("Min Samples Split", "2, 5, 10")
                        param_grid['min_samples_split'] = parse_range(mss_str)
                        estimator = DecisionTreeRegressor(random_state=42)
                        
                    elif "Bagging" in algo_type:
                        ne_str = st.text_input("N Estimators", "10, 50, 100")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        estimator = BaggingRegressor(random_state=42)
                        
                    elif "Gradient Boosting" in algo_type:
                        ne_str = st.text_input("N Estimators", "50, 100")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        lr_str = st.text_input("Learning Rate", "0.01, 0.1, 0.5")
                        param_grid['learning_rate'] = parse_range(lr_str, float)
                        estimator = GradientBoostingRegressor(random_state=42)
                        
                    elif "AdaBoost" in algo_type:
                        ne_str = st.text_input("N Estimators", "50, 100")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        lr_str = st.text_input("Learning Rate", "0.1, 1.0")
                        param_grid['learning_rate'] = parse_range(lr_str, float)
                        estimator = AdaBoostRegressor(random_state=42)
                        
                    elif "Normal Equation" in algo_type or "Gradient Descent" in algo_type:
                        alpha_str = st.text_input("Alpha (Ridge)", "0.1, 1.0, 10.0")
                        param_grid['alpha'] = parse_range(alpha_str, float)
                        estimator = Ridge()
                    
                    if st.button("Run Grid Search"):
                        if estimator is not None:
                            best_params, best_score, best_model = st.session_state.tuner.tune(estimator, param_grid, X_train, y_train)
                            st.success(f"Best Score (R2): {best_score:.4f}")
                            st.json(best_params)
                        else:
                            st.warning("Tuning not implemented for this specific selection yet.")

                # --- STANDARD TRAINING LOGIC ---
                else:
                    if "Bagging" in algo_type:
                        n_est = st.number_input("Number of Estimators", 10, 500, 50)
                        if st.button("Train Bagging"):
                            model = MyEnsembleWrapper(mode='regression', algo='bagging', n_estimators=n_est)
                            model.fit(X_train, y_train)
                            st.success("Bagging Model Trained")
                    
                    elif "Gradient Boosting" in algo_type:
                        c1, c2 = st.columns(2)
                        n_est = c1.number_input("Number of Estimators", 10, 500, 100)
                        lr_boost = c2.number_input("Learning Rate", 0.01, 1.0, 0.1)
                        if st.button("Train Gradient Boosting"):
                            model = MyEnsembleWrapper(mode='regression', algo='gradient_boosting', n_estimators=n_est, learning_rate=lr_boost)
                            model.fit(X_train, y_train)
                            st.success("GBM Model Trained")
                            
                    elif "AdaBoost" in algo_type:
                        c1, c2 = st.columns(2)
                        n_est = c1.number_input("Number of Estimators", 10, 500, 50)
                        lr_boost = c2.number_input("Learning Rate", 0.01, 2.0, 1.0)
                        if st.button("Train AdaBoost"):
                            model = MyEnsembleWrapper(mode='regression', algo='adaboost', n_estimators=n_est, learning_rate=lr_boost)
                            model.fit(X_train, y_train)
                            st.success("AdaBoost Model Trained")

                    elif "Decision Tree" in algo_type:
                        c1, c2, c3 = st.columns(3)
                        crit = c1.selectbox("Criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])
                        md = c2.number_input("Max Depth", 1, 20, 5)
                        mss = c3.number_input("Min Samples Split", 2, 20, 2)
                        
                        if st.button("Train Decision Tree"):
                            model = MyDecisionTree(mode='regression', max_depth=md, min_samples_split=mss, criterion=crit)
                            model.fit(X_train, y_train)
                            st.success("Decision Tree Trained")
                            st.text(model.get_text_tree(feature_names=features))

                    elif "Voting" in algo_type:
                        st.info("Combines Linear Regression and Decision Tree.")
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
                        if "Lasso" in reg_mode: st.error("🚫 Lasso needs GD."); st.stop()
                        if "Ridge" in reg_mode: alpha = st.number_input("Alpha", 0.01, 100.0, 1.0)

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
                        if "Ridge" in reg_mode_gd: reg_key_gd = 'Ridge (L2)'; alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                        elif "Lasso" in reg_mode_gd: reg_key_gd = 'Lasso (L1)'; alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                        
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
                            st.line_chart(final_hist)
                    
                    # --- RESULTS ---
                    if model:
                        y_pred = None
                        if "Gradient" in algo_type:
                            scaler = StandardScaler()
                            scaler.fit(X_train)
                            X_test_in = scaler.transform(X_test)
                            if is_poly: y_pred = model.predict(X_test_in, is_polynomial=True)
                            else: y_pred = model.predict(X_test_in)
                        elif is_poly and isinstance(model, MyLinearRegression):
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
                        
                        st.subheader(f"Actual {target} vs Predicted {target}")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
                        ax.set_xlabel(f"Actual {target}")
                        ax.set_ylabel(f"Predicted {target}")
                        ax.legend()
                        st.pyplot(fig)

            # ==========================
            # CLASSIFICATION LOGIC
            # ==========================
            elif problem_type == "Classification":
                if unique_targets > 2:
                    st.info("Multi-class detected.")
                    algo_list = [
                        "Softmax Regression", "Decision Tree Classifier", "Voting Classifier",
                        "Bagging Classifier", "Gradient Boosting Classifier", "AdaBoost Classifier"
                    ]
                else:
                    algo_list = [
                        "Perceptron", "Logistic Regression", "Decision Tree Classifier", "Voting Classifier",
                        "Bagging Classifier", "Gradient Boosting Classifier", "AdaBoost Classifier"
                    ]
                    
                class_algo = st.radio("Algorithm", algo_list)
                
                model = None
                is_poly = False
                enable_tuning = st.checkbox("Enable Hyperparameter Tuning (GridSearch)")

                if enable_tuning:
                    st.subheader("Hyperparameter Configuration")
                    param_grid = {}
                    estimator = None
                    
                    if "Decision Tree" in class_algo:
                        md_str = st.text_input("Max Depth", "3, 5, 10")
                        param_grid['max_depth'] = parse_range(md_str)
                        estimator = DecisionTreeClassifier(random_state=42)
                        
                    elif "Bagging" in class_algo:
                        ne_str = st.text_input("N Estimators", "10, 50")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        estimator = BaggingClassifier(random_state=42)
                    
                    elif "Gradient Boosting" in class_algo:
                        ne_str = st.text_input("N Estimators", "50, 100")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        lr_str = st.text_input("Learning Rate", "0.1, 1.0")
                        param_grid['learning_rate'] = parse_range(lr_str, float)
                        estimator = GradientBoostingClassifier(random_state=42)
                        
                    elif "AdaBoost" in class_algo:
                        ne_str = st.text_input("N Estimators", "50, 100")
                        param_grid['n_estimators'] = parse_range(ne_str)
                        lr_str = st.text_input("Learning Rate", "0.5, 1.0")
                        param_grid['learning_rate'] = parse_range(lr_str, float)
                        estimator = AdaBoostClassifier(random_state=42)
                        
                    elif "Logistic" in class_algo:
                        C_str = st.text_input("C (Inverse Regularization)", "0.1, 1.0, 10.0")
                        param_grid['C'] = parse_range(C_str, float)
                        estimator = LogisticRegression()
                    
                    if st.button("Run Grid Search"):
                        # FIX: Using 'if estimator is not None' avoids invoking __len__ on unfitted ensembles
                        if estimator is not None:
                            best_params, best_score, best_model = st.session_state.tuner.tune(estimator, param_grid, X_train, y_train)
                            st.success(f"Best Accuracy: {best_score:.4f}")
                            st.json(best_params)
                        else:
                            st.warning("Tuning not implemented for this algo.")

                else:
                    if "Bagging" in class_algo:
                        n_est = st.number_input("Number of Estimators", 10, 500, 50)
                        if st.button("Train Bagging"):
                            model = MyEnsembleWrapper(mode='classification', algo='bagging', n_estimators=n_est)
                            model.fit(X_train, y_train)
                            st.success("Bagging Classifier Trained")
                            
                    elif "Gradient Boosting" in class_algo:
                        c1, c2 = st.columns(2)
                        n_est = c1.number_input("Number of Estimators", 10, 500, 100)
                        lr_boost = c2.number_input("Learning Rate", 0.01, 1.0, 0.1)
                        if st.button("Train Gradient Boosting"):
                            model = MyEnsembleWrapper(mode='classification', algo='gradient_boosting', n_estimators=n_est, learning_rate=lr_boost)
                            model.fit(X_train, y_train)
                            st.success("GBM Classifier Trained")
                            
                    elif "AdaBoost" in class_algo:
                        c1, c2 = st.columns(2)
                        n_est = c1.number_input("Number of Estimators", 10, 500, 50)
                        lr_boost = c2.number_input("Learning Rate", 0.01, 2.0, 1.0)
                        if st.button("Train AdaBoost"):
                            model = MyEnsembleWrapper(mode='classification', algo='adaboost', n_estimators=n_est, learning_rate=lr_boost)
                            model.fit(X_train, y_train)
                            st.success("AdaBoost Classifier Trained")

                    elif "Decision Tree" in class_algo:
                        c1, c2, c3 = st.columns(3)
                        crit = c1.selectbox("Criterion", ["gini", "entropy", "log_loss"])
                        md = c2.number_input("Max Depth", 1, 20, 5)
                        mss = c3.number_input("Min Samples Split", 2, 20, 2)
                        
                        if st.button("Train Tree"):
                            model = MyDecisionTree(mode='classification', max_depth=md, min_samples_split=mss, criterion=crit)
                            model.fit(X_train, y_train)
                            st.success("Tree Trained")
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
                        if "Ridge" in reg_mode_gd: reg_key_gd = 'Ridge (L2)'; alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                        elif "Lasso" in reg_mode_gd: reg_key_gd = 'Lasso (L1)'; alpha_gd = st.number_input("Alpha", 0.01, 100.0, 1.0)
                        
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
                                    loss_placeholder.pyplot(fig_l)
                                    plt.close(fig_l)
                                    
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
                                        ax_f.set_xlabel(features[0])
                                        ax_f.set_ylabel(features[1])
                                        plot_placeholder.pyplot(fig_f)
                                        plt.close(fig_f)
                                    time.sleep(speed)
                                progress.progress((ep+1)/epochs)
                            
                            st.line_chart(final_hist)

                    # --- RESULTS ---
                    if model:
                        y_pred = None
                        if "Tree" in class_algo or "Voting" in class_algo or "Bagging" in class_algo or "Boosting" in class_algo or "AdaBoost" in class_algo:
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
                                x_min, x_max = X_data[:,0].min()-1, X_data[:,0].max()+1
                                y_min, y_max = X_data[:,1].min()-1, X_data[:,1].max()+1
                                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                                mesh_input = np.c_[xx.ravel(), yy.ravel()]
                                
                                Z = None
                                if "Tree" in class_algo or "Voting" in class_algo or "Bagging" in class_algo or "Boosting" in class_algo or "AdaBoost" in class_algo:
                                    Z = model.predict(mesh_input)
                                else:
                                    if not is_scaled:
                                        scaler = StandardScaler()
                                        scaler.fit(X_train) 
                                        mesh_input = scaler.transform(mesh_input)
                                    Z = model.predict(mesh_input, is_polynomial=is_poly)
                                    
                                Z = Z.reshape(xx.shape)
                                ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                                ax.scatter(X_data[:,0], X_data[:,1], c=y_data, cmap='viridis', edgecolors='k')
                                ax.set_title(title)
                                ax.set_xlabel(features[0])
                                ax.set_ylabel(features[1])

                            with c_train:
                                fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
                                is_sc = False
                                if "Tree" not in class_algo and "Voting" not in class_algo and "Bagging" not in class_algo and "Boosting" not in class_algo and "AdaBoost" not in class_algo:
                                    scaler = StandardScaler()
                                    X_train_s = scaler.fit_transform(X_train)
                                    X_in = X_train_s
                                    is_sc = True
                                else:
                                    X_in = X_train
                                plot_boundary(ax_tr, X_in, y_train, "Train Fit", is_scaled=is_sc)
                                st.pyplot(fig_tr)
                                
                            with c_test:
                                fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                                is_sc = False
                                if "Tree" not in class_algo and "Voting" not in class_algo and "Bagging" not in class_algo and "Boosting" not in class_algo and "AdaBoost" not in class_algo:
                                    scaler = StandardScaler()
                                    scaler.fit(X_train)
                                    X_test_s = scaler.transform(X_test)
                                    X_in_t = X_test_s
                                    is_sc = True
                                else:
                                    X_in_t = X_test
                                plot_boundary(ax_ts, X_in_t, y_test, "Test Fit", is_scaled=is_sc)
                                st.pyplot(fig_ts)
                        else:
                            st.markdown("**Confusion Matrix**")
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap='Blues', colorbar=False)
                            st.pyplot(fig_cm)

            # ==========================
            # CLUSTERING LOGIC
            # ==========================
            elif problem_type == "Clustering":
                st.info("Unsupervised Learning detected.")
                
                # Clustering Tuning (Silhouette Score)
                enable_tuning = st.checkbox("Tune Number of Clusters (K)")
                
                if enable_tuning:
                    k_min = st.number_input("Min K", 2, 5, 2)
                    k_max = st.number_input("Max K", 6, 15, 10)
                    if st.button("Run Silhouette Analysis"):
                        best_k, best_score, results = st.session_state.tuner.tune_kmeans(X, k_min, k_max)
                        st.success(f"Best K: {best_k} (Silhouette Score: {best_score:.4f})")
                        
                        ks = list(results.keys())
                        scores = list(results.values())
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(ks, scores, marker='o')
                        ax.set_xlabel("Number of Clusters (K)")
                        ax.set_ylabel("Silhouette Score")
                        ax.set_title("Elbow Method (Silhouette)")
                        ax.grid(True)
                        st.pyplot(fig)
                else:
                    n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
                    if st.button("Run K-Means"):
                        model = MyKMeans(n_clusters=n_clusters)
                        model.fit(X)
                        labels = model.predict(X)
                        
                        score = silhouette_score(X, labels)
                        st.metric("Silhouette Score", f"{score:.4f}")
                        
                        st.subheader("Cluster Visualization")
                        if len(features) == 2:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolors='k')
                            centers = model.model.cluster_centers_
                            ax.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, label='Centroids')
                            ax.legend()
                            ax.set_title(f"K-Means (K={n_clusters})")
                            ax.set_xlabel(features[0])
                            ax.set_ylabel(features[1])
                            st.pyplot(fig)
                        else:
                            st.warning("Visualization only available for 2D data.")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolors='k')
                            st.pyplot(fig)

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
    
    st.header("8. Polynomial Regression / Classification")
    st.markdown("""
    **Concept:** Transform input features into higher order terms to capture non-linear relationships.
    If input is $x$, degree 2 polynomial features are $[1, x, x^2]$.
    The algorithm (Linear or Logistic) remains the same, but acts on these transformed features.
    """)
    
    st.header("9. Decision Trees")
    st.markdown("""
    **Concept:** Recursively splits data based on features to create homogeneous groups.
    **Splitting Criteria (Classification):**
    * **Gini Impurity:** $1 - \\sum p_i^2$
    * **Entropy:** $- \\sum p_i \\log_2(p_i)$
    """)
    

    st.header("10. Ensemble Voting")
    st.markdown("""
    **Hard Voting:** Majority rule.
    **Soft Voting:** Average probabilities.
    """)
    
    
    st.header("11. Bagging (Bootstrap Aggregating)")
    st.markdown("""
    **Concept:** Train multiple base estimators (usually Decision Trees) on different random subsets of the training data (with replacement).
    **Aggregation:** Average predictions (Regression) or Vote (Classification).
    **Goal:** Reduces Variance (Overfitting).
    """)
    
    st.header("12. Boosting (Gradient Boosting & AdaBoost)")
    st.markdown("""
    **Concept:** Train estimators sequentially. Each new estimator corrects the errors of the previous ones.
    **AdaBoost:** Focuses on difficult data points by increasing their weights.
    **Gradient Boosting:** Fits the new estimator to the *residual errors* of the previous model.
    **Goal:** Reduces Bias (Underfitting).
    """)
    
    st.header("13. K-Means Clustering")
    st.markdown("""
    **Goal:** Partition data into $K$ clusters.
    **Algorithm:**
    1. Initialize $K$ centroids randomly.
    2. Assign each point to the nearest centroid.
    3. Update centroids to the mean of assigned points.
    4. Repeat until convergence.
    **Objective:** Minimize inertia (sum of squared distances to centroids).
    """)
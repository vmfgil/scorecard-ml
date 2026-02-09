import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import unicodedata
from scipy import stats
from scipy.stats import norm 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import sys
import time
import gc 
import re 
import io

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="ML Scorecard Pro", layout="wide")

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600, show_spinner="A ler dados...")
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, engine='pyarrow')
        else:
            return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Erro a ler ficheiro: {e}")
        return None

def log_message(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def calculate_ks(y_true, y_probs):
    df_ks = pd.DataFrame({'target': y_true, 'prob': y_probs})
    df_ks = df_ks.sort_values('prob', ascending=False)
    df_ks['cum_event'] = df_ks['target'].cumsum() / df_ks['target'].sum()
    df_ks['cum_non_event'] = (df_ks['target'] == 0).cumsum() / (df_ks['target'] == 0).sum()
    df_ks['ks_dist'] = abs(df_ks['cum_event'] - df_ks['cum_non_event'])
    return df_ks['ks_dist'].max() * 100, df_ks

def calculate_iv(df, feature, target):
    """Calcula Information Value (IV)"""
    lst = []
    clean_series = df[feature].astype(str)
    for val in clean_series.unique():
        all_cnt = clean_series[clean_series==val].count()
        bad_cnt = df[(clean_series==val) & (df[target]==1)].shape[0]
        good_cnt = all_cnt - bad_cnt
        lst.append([str(val), good_cnt, bad_cnt, all_cnt])
    
    dset = pd.DataFrame(lst, columns=['Value', 'Good', 'Bad', 'Total'])
    
    total_good = dset['Good'].sum()
    total_bad = dset['Bad'].sum()
    if total_good == 0: total_good = 1
    if total_bad == 0: total_bad = 1
    
    dset['Dist_Good'] = dset['Good'] / total_good
    dset['Dist_Bad'] = dset['Bad'] / total_bad
    dset['Dist_Good'] = dset['Dist_Good'].replace(0, 0.0001)
    dset['Dist_Bad'] = dset['Dist_Bad'].replace(0, 0.0001)
    
    dset['WoE'] = np.log(dset['Dist_Good'] / dset['Dist_Bad'])
    dset['IV'] = (dset['Dist_Good'] - dset['Dist_Bad']) * dset['WoE']
    
    return dset['IV'].sum(), dset

def monotonic_binning(df, var, target, max_bins=5):
    n_bins = max_bins
    while n_bins >= 2:
        try:
            df[f'{var}_bins'] = pd.qcut(df[var], n_bins, duplicates='drop')
            stats_bin = df.groupby(f'{var}_bins', observed=True)[target].agg(['count', 'sum', 'mean'])
            min_sample = len(df) * 0.02
            
            # Checks
            has_event = all(stats_bin['sum'] > 0)
            large_enough = all(stats_bin['count'] > min_sample)
            
            means = stats_bin['mean'].values
            is_increasing = np.all(np.diff(means) >= 0)
            is_decreasing = np.all(np.diff(means) <= 0)
            
            if has_event and large_enough and (is_increasing or is_decreasing):
                return df[f'{var}_bins']
            
            n_bins -= 1
        except Exception as e:
            n_bins -= 1
    return None

def sanitize_col_name(name):
    return re.sub(r'[^\w]', '_', str(name))

def generate_normal_bins(n_grades):
    x = np.linspace(-2.5, 2.5, n_grades)
    weights = norm.pdf(x)
    weights = weights / weights.sum()
    cum_weights = np.cumsum(weights)
    bins = [0.0] + list(cum_weights)
    bins[-1] = 1.0 
    return bins

def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')

def scale_scorecard(df_raw, score_min=300, score_max=850):
    agg_stats = df_raw.groupby('Variável')['Coef'].agg(['min', 'max']).reset_index()
    agg_stats.columns = ['Variável', 'MinCoef', 'MaxCoef']
    
    df_calc = pd.merge(df_raw, agg_stats, on='Variável')
    
    SCORE_DELTA = score_max - score_min
    N_VARS = df_calc['Variável'].nunique()
    BASE_SCORE_PER_VAR = score_min / N_VARS
    
    SUM_MIN_COEF = agg_stats['MinCoef'].sum()
    SUM_MAX_COEF = agg_stats['MaxCoef'].sum()
    
    DENOMINATOR = SUM_MIN_COEF - SUM_MAX_COEF
    if DENOMINATOR == 0: DENOMINATOR = -1 
    
    df_calc['Score Base'] = BASE_SCORE_PER_VAR
    df_calc['Pontos'] = df_calc['Score Base'] + ((df_calc['Coef'] - df_calc['MaxCoef']) / DENOMINATOR) * SCORE_DELTA
    df_calc['Pontos'] = df_calc['Pontos'].fillna(0).round(0).astype(int)
    
    agg_stats['Amplitude'] = agg_stats['MaxCoef'] - agg_stats['MinCoef']
    total_amplitude = agg_stats['Amplitude'].sum()
    if total_amplitude == 0: total_amplitude = 1
    
    agg_stats['Peso (%)'] = (agg_stats['Amplitude'] / total_amplitude) * 100
    weights_table = agg_stats[['Variável', 'Peso (%)']].sort_values('Peso (%)', ascending=False)
    
    cols_final = ['Variável', 'Categoria', 'WoE', 'Coef', 'Score Base', 'Pontos']
    if 'WoE' not in df_calc.columns: df_calc['WoE'] = np.nan
        
    scorecard_final = df_calc[cols_final].sort_values(['Variável', 'Pontos'])
    
    return scorecard_final, weights_table

# --- INICIALIZAÇÃO DO ESTADO ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'df' not in st.session_state: st.session_state.df = None
if 'features' not in st.session_state: st.session_state.features = []
if 'target' not in st.session_state: st.session_state.target = None
if 'model_ready' not in st.session_state: st.session_state.model_ready = False
if 'num_grades' not in st.session_state: st.session_state.num_grades = 5
if 'iv_tables' not in st.session_state: st.session_state.iv_tables = {} 

keys_report = ['report_dq', 'report_split', 'report_uni', 'report_multi', 'report_ml', 
               'final_model_data', 'report_dq_missings', 'report_multi_details', 'roc_curves', 
               'report_iv', 'report_binning_fail', 'ks_curves', 'uni_summary', 'X_test_final', 'y_test_final',
               'stats_dq', 'stats_uni', 'stats_multi', 'stats_cat']

for key in keys_report:
    if key not in st.session_state: st.session_state[key] = None

# --- INTERFACE DE LOGIN ---
if not st.session_state.logged_in:
    st.title("🔐 Acesso ao Sistema")
    with st.form("login_form"):
        user = st.text_input("Utilizador")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")
        if submit:
            if user == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Credenciais inválidas")
else:
    # --- SIDEBAR ---
    st.sidebar.title("🚀 Menu Principal")
    # ALTERAÇÃO: Nome do Menu
    menu = st.sidebar.radio("Ir para:", ["🏠 Início", "📁 Upload de Dados", "⚙️ Modelização", "📊 Resultados Finais", "📑 Relatório Técnico"])
    if st.sidebar.button("Sair"):
        st.session_state.logged_in = False
        st.rerun()

    # --- LÓGICA DAS PÁGINAS ---
    if menu == "🏠 Início":
        st.title("Bem-vindo ao ML Scorecard Pro")
        st.markdown("""Esta aplicação foi desenhada para automatizar o fluxo de criação de modelos preditivos.""")

    elif menu == "📁 Upload de Dados":
        st.title("Upload e Configuração de Variáveis")
        uploaded_file = st.file_uploader("Carregue o seu ficheiro CSV ou Excel", type=['csv', 'xlsx'])
        
        if uploaded_file:
            if st.session_state.df is None:
                with st.spinner('A carregar dados para a cache...'):
                    st.session_state.df = load_data(uploaded_file)
                    if not st.session_state.features:
                        st.session_state.features = st.session_state.df.columns.tolist()

            st.success(f"Dados prontos! {len(st.session_state.df)} linhas carregadas.")

            col1, col2 = st.columns(2)
            cols = st.session_state.df.columns.tolist()

            with col1:
                def update_features():
                    st.session_state.features = st.session_state.widget_features
                
                selected_feats = st.multiselect(
                    "Candidatas (Features):", 
                    options=cols, 
                    default=st.session_state.features,
                    key="widget_features",
                    on_change=update_features
                )
            
            with col2:
                current = st.session_state.target
                idx = cols.index(current) + 1 if current in cols else 0
                selected_target = st.selectbox("Target:", options=[None] + cols, index=idx)
                st.session_state.target = selected_target

            st.markdown("---")
            if st.button("✅ Confirmar Configuração de Variáveis"):
                if st.session_state.target is None:
                    st.error("Por favor, selecione uma variável Target.")
                else:
                    st.success(f"Configuração Guardada! Alvo: {st.session_state.target}")
                    
    elif menu == "⚙️ Modelização":
        st.title("Configuração do Modelo")
        if st.session_state.df is None or st.session_state.target is None:
            st.warning("Por favor, faça o upload dos dados e selecione a target primeiro.")
        else:
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1: 
                num_vars_limit = st.number_input("Número máximo de variáveis finais", min_value=1, value=10)
            with col_cfg2: 
                n_grades_input = st.slider("Número de Grades (Scorecard)", min_value=3, max_value=15, value=5)
            
            if st.button("🚀 Correr Modelo Completo"):
                st.session_state.num_grades = n_grades_input
                
                with st.status("A iniciar motor de cálculo...", expanded=True) as status:
                    log_message("--- INÍCIO DO PROCESSO ---")
                    st.write("📥 A preparar dados...")
                    
                    df_work = st.session_state.df.copy() 
                    target_col = st.session_state.target
                    feature_cols = sorted(list(set([c for c in st.session_state.features if c != target_col])))
                    
                    initial_count = len(feature_cols)
                    log_dq = []
                    
                    # --- a) DATA QUALITY ---
                    st.write("🧹 Executando Data Quality...")
                    log_message(f"Total Variáveis Iniciais: {initial_count}")
                    
                    df_work.replace(r'^\s*$', np.nan, regex=True, inplace=True)
                    
                    miss_series = df_work[feature_cols].isnull().mean()
                    missing_report = pd.DataFrame({
                        'Variável': feature_cols,
                        'Missing %': miss_series
                    })
                    missing_report['Status'] = missing_report['Missing %'].apply(lambda x: "Eliminar" if x > 0.2 else "Manter")
                    st.session_state.report_dq_missings = missing_report

                    for col in feature_cols:
                        if df_work[col].dtype == 'object':
                            # Tenta converter
                            conv = pd.to_numeric(df_work[col], errors='coerce')
                            if conv.notnull().mean() <= 0.5:
                                try:
                                    clean = df_work[col].astype(str).str.replace(r'[€%\s]', '', regex=True).str.replace(',', '')
                                    conv = pd.to_numeric(clean, errors='coerce')
                                except: pass
                            
                            if conv.notnull().mean() > 0.5:
                                df_work[col] = conv
                                log_message(f"DQ: Variável '{col}' convertida para Numérica.")

                    drop = missing_report[missing_report['Status'] == 'Eliminar']['Variável'].tolist()
                    if drop: log_message(f"DQ: Eliminadas por Missing > 20%: {drop}")
                    feature_cols = [c for c in feature_cols if c not in drop]
                    
                    # ALTERAÇÃO: Guardar estatísticas DQ
                    st.session_state.stats_dq = {
                        "Entrada": initial_count,
                        "Eliminadas": len(drop),
                        "Saída": len(feature_cols)
                    }
                    
                    num_cols = df_work[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df_work[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    if num_cols:
                        for idx, col in enumerate(num_cols):
                            low, high = df_work[col].quantile([0.001, 0.999])
                            df_work[col] = df_work[col].clip(low, high).fillna(-999999)

                    if cat_cols:
                        for idx, col in enumerate(cat_cols):
                            df_work[col] = df_work[col].fillna("DESCONHECIDO").astype(str)
                            cnt = df_work[col].value_counts(normalize=True)
                            rare = cnt[cnt < 0.01].index
                            if not rare.empty: df_work.loc[df_work[col].isin(rare), col] = "OUTROS"
                    
                    st.session_state.report_dq = log_dq
                    
                    # --- b) SPLIT ---
                    st.write("✂️ Split...")
                    train, test = train_test_split(df_work, test_size=0.3, stratify=df_work[target_col], random_state=42)
                    
                    split_stats = pd.DataFrame({
                        'Amostra': ['Treino', 'Teste', 'Total'],
                        'Registos': [len(train), len(test), len(df_work)],
                        'Eventos': [train[target_col].sum(), test[target_col].sum(), df_work[target_col].sum()],
                        'Taxa Evento': [train[target_col].mean(), test[target_col].mean(), df_work[target_col].mean()]
                    })
                    st.session_state.report_split = split_stats # Guardar no estado para usar depois

                    # --- c) UNIVARIADA ---
                    st.write("🔬 Univariada...")
                    log_message("A iniciar Análise Univariada...")
                    uni_res = []
                    keep_uni = []
                    samp = train.sample(20000, random_state=42) if len(train) > 20000 else train
                    
                    count_uni_in = len(feature_cols)
                    
                    for idx, col in enumerate(feature_cols):
                        try:
                            if col in cat_cols:
                                if samp[col].nunique() > 50:
                                    top = samp[col].value_counts().nlargest(20).index
                                    s = samp[col].where(samp[col].isin(top), "OUTROS")
                                    p = stats.chi2_contingency(pd.crosstab(s, samp[target_col]))[1]
                                else:
                                    p = stats.chi2_contingency(pd.crosstab(samp[col], samp[target_col]))[1]
                                metric, tipo = "Chi2", "Categórica"
                            else:
                                tau, p = stats.kendalltau(samp[col], samp[target_col])
                                metric, tipo = "Kendall", "Numérica"
                        except: p, metric, tipo = 1.0, "Erro", "Erro"
                        
                        dec = "Manter" if p <= 0.05 else "Eliminar"
                        uni_res.append({"Variável": col, "Tipo": tipo, "Teste": metric, "p-value": p, "Status": dec})
                        if p <= 0.05: 
                            keep_uni.append(col)
                        else:
                            log_message(f"Univariada: Eliminada '{col}' (p={p:.4f})")
                    
                    st.session_state.report_uni = pd.DataFrame(uni_res)
                    
                    # ALTERAÇÃO: Guardar estatísticas Univariada
                    st.session_state.stats_uni = {
                        "Entrada": count_uni_in,
                        "Eliminadas": count_uni_in - len(keep_uni),
                        "Saída": len(keep_uni)
                    }

                    # --- d) MULTIVARIADA ---
                    st.write("🕸️ Multivariada...")
                    num_k = [f for f in keep_uni if f in num_cols]
                    drop_multi = set()
                    multi_details = [] # Inicializar lista para guardar detalhes
                    
                    if len(num_k) > 1:
                        corr = train[num_k].sample(50000).corr().abs() if len(train)>50000 else train[num_k].corr().abs()
                        st.session_state.corr_matrix = corr 
                        
                        pairs = np.where(corr > 0.8)
                        for i, j in zip(*pairs):
                            if i < j:
                                c1, c2 = num_k[i], num_k[j]
                                try:
                                    tau1, _ = stats.kendalltau(train[c1].fillna(-999), train[target_col])
                                    tau2, _ = stats.kendalltau(train[c2].fillna(-999), train[target_col])
                                    val_corr = corr.iloc[i, j] # Guardar valor correlação
                                    
                                    if abs(tau1) < abs(tau2): 
                                        elim, keeper = c1, c2
                                        k_elim, k_keep = tau1, tau2
                                    else: 
                                        elim, keeper = c2, c1
                                        k_elim, k_keep = tau2, tau1
                                        
                                    drop_multi.add(elim)
                                    # ALTERAÇÃO: Preencher tabela de detalhes
                                    multi_details.append({
                                        "Eliminada": elim, 
                                        "Mantida (Killer)": keeper, 
                                        "Correlação": val_corr,
                                        "Kendall Eliminada": round(k_elim, 4),
                                        "Kendall Mantida": round(k_keep, 4)
                                    })
                                except: pass
                    
                    final_feats = [f for f in keep_uni if f not in drop_multi]
                    st.session_state.report_multi = list(drop_multi)
                    
                    # ALTERAÇÃO: Guardar DataFrame com detalhes da eliminação
                    if multi_details:
                        st.session_state.report_multi_details = pd.DataFrame(multi_details).drop_duplicates(subset=['Eliminada'])
                    else:
                        st.session_state.report_multi_details = pd.DataFrame(columns=["Eliminada", "Mantida (Killer)", "Correlação", "Kendall Eliminada", "Kendall Mantida"])

                    # ALTERAÇÃO: Guardar estatísticas Multivariada
                    st.session_state.stats_multi = {
                        "Entrada": len(keep_uni),
                        "Eliminadas": len(drop_multi),
                        "Saída": len(final_feats)
                    }

                    # --- e) BINNING & CATEGORIZATION ---
                    st.write("📦 Binning & IV...")
                    log_message("A iniciar Binning...")
                    train_b, test_b = train.copy(), test.copy()
                    proc_feats = []
                    binning_fail = []
                    iv_report = []
                    iv_tables_dict = {}
                    
                    count_cat_in = len(final_feats)
                    
                    for idx, col in enumerate(final_feats):
                        is_num = col in num_cols
                        
                        if is_num and train[col].nunique() <= 3:
                            is_num = False # Trata como categórica se tiver poucos valores

                        if is_num:
                            binned = monotonic_binning(train, col, target_col)
                            if binned is not None:
                                train_b[col] = binned.astype(str)
                                try:
                                    _, bins = pd.qcut(train[col], 5, duplicates='drop', retbins=True)
                                    test_b[col] = pd.cut(test[col], bins, include_lowest=True).astype(str)
                                except: test_b[col] = "Other"
                            else:
                                binning_fail.append(col)
                                log_message(f"Binning Falhou (Monotonicidade): {col}")
                                continue 
                        
                        train_b[col] = train_b[col].astype(str)
                        if col in test_b.columns: test_b[col] = test_b[col].astype(str)

                        val_iv, df_iv_table = calculate_iv(train_b, col, target_col)
                        n_bins = train_b[col].nunique()
                        
                        if n_bins > 1:
                            iv_report.append({"Variável": col, "Nº Bins": n_bins, "IV": val_iv})
                            iv_tables_dict[col] = df_iv_table 
                            proc_feats.append(col)
                        else:
                            binning_fail.append(col)
                            log_message(f"Binning Falhou (1 Bin): {col}")

                    if not iv_report:
                        st.error("ERRO CRÍTICO: Nenhuma variável sobreviveu ao Binning Monotónico.")
                        st.stop()

                    st.session_state.report_iv = pd.DataFrame(iv_report).sort_values('IV', ascending=False)
                    st.session_state.report_binning_fail = binning_fail
                    st.session_state.iv_tables = iv_tables_dict
                    
                    # ALTERAÇÃO: Guardar estatísticas Categorização
                    st.session_state.stats_cat = {
                        "Entrada": count_cat_in,
                        "Eliminadas": len(binning_fail),
                        "Saída": len(proc_feats)
                    }
                    
                    log_message(f"Fase Final: {len(proc_feats)} variáveis prontas para modelização.")

                    # --- f) PREP DUMMIES ---
                    st.write("🔢 Seleção de Variáveis (Feature Selection)...")
                    
                    for col in proc_feats:
                        stats_cat = train_b.groupby(col)[target_col].mean().sort_values()
                        ordered_cats = stats_cat.index.tolist()
                        train_b[col] = pd.Categorical(train_b[col], categories=ordered_cats, ordered=True)
                        test_b[col] = pd.Categorical(test_b[col], categories=ordered_cats, ordered=True)
                    
                    X_tr_all = pd.get_dummies(train_b[proc_feats], drop_first=True)
                    X_te_all = pd.get_dummies(test_b[proc_feats], drop_first=True)
                    for c in (set(X_tr_all.columns) - set(X_te_all.columns)): X_te_all[c] = 0
                    X_te_all = X_te_all[X_tr_all.columns]
                    
                    clean = [sanitize_col_name(c) for c in X_tr_all.columns]
                    X_tr_all.columns, X_te_all.columns = clean, clean
                    y_tr, y_te = train[target_col], test[target_col]
                    
                    total_b = pd.concat([train_b[proc_feats], test_b[proc_feats]])
                    y_total = pd.concat([y_tr, y_te])

                    # --- g) MODELOS E SCORECARD ---
                    st.write("🤖 Treino e Seleção de Features...")
                    
                    models_def = {
                        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
                        "XGBoost": xgb.XGBClassifier(n_estimators=50, eval_metric='logloss'),
                        "LightGBM": lgb.LGBMClassifier(n_estimators=50, verbosity=-1),
                        "CatBoost": CatBoostClassifier(n_estimators=50, verbose=0)
                    }
                    
                    res_ml, model_data_dict, roc_data, ks_data = [], {}, {}, {}
                    
                    for name, m in models_def.items():
                        log_message(f"Processando: {name}")
                        
                        if name == "LogReg":
                            scaler = StandardScaler()
                            X_train_model = pd.DataFrame(scaler.fit_transform(X_tr_all), columns=X_tr_all.columns, index=X_tr_all.index)
                            X_test_model = pd.DataFrame(scaler.transform(X_te_all), columns=X_te_all.columns, index=X_te_all.index)
                        else:
                            X_train_model = X_tr_all
                            X_test_model = X_te_all
                        
                        m.fit(X_train_model, y_tr)
                        
                        if name == "LogReg": raw_importances = np.abs(m.coef_[0])
                        elif name == "CatBoost": raw_importances = m.get_feature_importance()
                        else: raw_importances = m.feature_importances_
                        
                        feat_imp_map = dict(zip(X_train_model.columns, raw_importances))
                        parent_importances = {}
                        
                        for parent in proc_feats:
                            clean_parent = sanitize_col_name(parent)
                            relevant_dummies = [col for col in X_train_model.columns if col.startswith(clean_parent)]
                            if relevant_dummies:
                                vals = [feat_imp_map.get(d, 0) for d in relevant_dummies]
                                parent_importances[parent] = np.mean(vals)
                            else:
                                parent_importances[parent] = 0.0
                        
                        sorted_vars = sorted(parent_importances, key=parent_importances.get, reverse=True)
                        selected_vars = sorted_vars[:num_vars_limit]
                        log_message(f"Vars Finais ({name}): {selected_vars}")
                        
                        final_dummies = []
                        for col in X_train_model.columns:
                            for v in selected_vars:
                                clean_v = sanitize_col_name(v)
                                if col.startswith(clean_v):
                                    final_dummies.append(col)
                                    break
                        
                        X_tr_fin = X_train_model[final_dummies]
                        X_te_fin = X_test_model[final_dummies]
                        
                        m.fit(X_tr_fin, y_tr)
                        
                        pte = m.predict_proba(X_te_fin)[:,1]
                        auc = roc_auc_score(y_te, pte)
                        ks, df_ks = calculate_ks(y_te, pte)
                        fpr, tpr, _ = roc_curve(y_te, pte) 
                        
                        roc_data[name] = (fpr, tpr, float(auc)) 
                        ks_data[name] = df_ks
                        res_ml.append({"Algoritmo": name, "ROC AUC": round(auc,3), "KS": round(ks,1)})
                        
                        sc_rows = []
                        if name == "LogReg":
                            coef_df = pd.DataFrame({'Dummy': X_tr_fin.columns, 'Coef': m.coef_[0]})
                            for var in selected_vars:
                                if var in iv_tables_dict: 
                                    woe_map = iv_tables_dict[var].set_index('Value')['WoE'].to_dict()
                                else: woe_map = {}
                                
                                for cat in sorted(train_b[var].cat.categories):
                                    cat_str = str(cat)
                                    dpat = sanitize_col_name(f"{var}_{cat}")
                                    match_coef = 0.0
                                    for dummy_col in coef_df['Dummy'].values:
                                        if dummy_col == dpat or (dummy_col.startswith(sanitize_col_name(var)) and str(cat) in dummy_col):
                                            match_coef = coef_df.loc[coef_df['Dummy']==dummy_col, 'Coef'].values[0]
                                            break
                                    val_woe = woe_map.get(cat_str, np.nan)
                                    sc_rows.append({'Variável': var, 'Categoria': cat_str, 'WoE': val_woe, 'Coef': match_coef})
                        else:
                            for var in selected_vars:
                                if var in iv_tables_dict:
                                    df_woe = iv_tables_dict[var]
                                    wmin = df_woe['WoE'].min()
                                    wmap = df_woe.set_index('Value')['WoE'].to_dict()
                                    for cat in sorted(train_b[var].cat.categories):
                                        cat_str = str(cat)
                                        val = wmap.get(cat_str, 0)
                                        sc_rows.append({'Variável': var, 'Categoria': cat_str, 'WoE': val, 'Coef': wmin - val})
                        
                        df_sc, weights = scale_scorecard(pd.DataFrame(sc_rows))
                        
                        df_scored = total_b.copy()
                        df_scored['TotalScore'] = 0
                        pmap = df_sc.set_index(['Variável', 'Categoria'])['Pontos'].to_dict()
                        
                        for var in selected_vars:
                            df_scored[f'P_{var}'] = df_scored[var].astype(str).apply(lambda x: pmap.get((var, x), 0)).fillna(0).astype(int)
                            df_scored['TotalScore'] += df_scored[f'P_{var}']
                        
                        temp_dummies = pd.get_dummies(total_b[selected_vars], drop_first=True)
                        temp_clean = [sanitize_col_name(c) for c in temp_dummies.columns]
                        temp_dummies.columns = temp_clean
                        
                        X_total_model = pd.DataFrame(0, index=np.arange(len(total_b)), columns=final_dummies)
                        for c in final_dummies:
                            if c in temp_dummies.columns:
                                X_total_model[c] = temp_dummies[c].values

                        if name == "LogReg":
                            final_scaler_total = StandardScaler()
                            final_scaler_total.fit(X_tr_fin)
                            X_total_model = pd.DataFrame(final_scaler_total.transform(X_total_model), columns=final_dummies)
                        
                        ptotal = m.predict_proba(X_total_model)[:,1]
                        
                        model_data_dict[name] = {
                            "scorecard": df_sc,
                            "weights": weights,
                            "scores_total": df_scored['TotalScore'].values,
                            "y_total": y_total.values,
                            "preds_total": ptotal
                        }

                    st.session_state.report_ml = pd.DataFrame(res_ml)
                    st.session_state.final_model_data = model_data_dict
                    st.session_state.roc_curves = roc_data
                    st.session_state.ks_curves = ks_data
                    st.session_state.model_ready = True
                    
                    status.update(label="✅ Concluído!", state="complete", expanded=False)
                    st.success("Fim!")

    elif menu == "📊 Resultados Finais":
        st.title("Resultados e Scorecard")
        if not st.session_state.model_ready:
            st.error("Primeiro precisa de correr o modelo na aba de Modelização.")
        else:
            opts = list(st.session_state.final_model_data.keys())
            sel_model = st.selectbox("Selecione o Modelo para Análise:", opts)
            data_model = st.session_state.final_model_data.get(sel_model)
            
            if data_model:
                st.subheader("Importância das Variáveis (Pesos)")
                st.dataframe(data_model.get("weights"), use_container_width=True)

                st.subheader(f"Grelha de Pontuação - {sel_model}")
                st.dataframe(data_model.get("scorecard"), use_container_width=True)
                
                scores = data_model["scores_total"]
                targets = data_model["y_total"]
                
                st.subheader("Distribuição dos Scores")
                fig_hist = px.histogram(x=scores, nbins=50, title=f"Histograma de Scores - {sel_model}", labels={'x': 'Score', 'y': 'Contagem'})
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)

                st.divider()

                st.subheader("Performance por Grade")
                df_res = pd.DataFrame({"Score": scores, "Target": targets})
                df_res = df_res.sort_values("Score", ascending=False)
                
                n_grades = st.session_state.num_grades
                normal_bins_cuts = generate_normal_bins(n_grades)
                
                df_res['Grade'] = pd.qcut(df_res['Score'].rank(method='first', pct=True), q=normal_bins_cuts, labels=False)
                df_res['Grade'] = df_res['Grade'].apply(lambda x: chr(65 + (n_grades - 1 - int(x))))
                
                grade_stats = df_res.groupby("Grade").agg(
                    Min_Score=('Score', 'min'),
                    Max_Score=('Score', 'max'),
                    Count=('Target', 'count'),
                    Bad=('Target', 'sum')
                ).reset_index()
                
                grade_stats['Bad Rate'] = grade_stats['Bad'] / grade_stats['Count']
                grade_stats['Label'] = grade_stats.apply(lambda x: f"{x['Grade']} [{int(x['Min_Score'])}-{int(x['Max_Score'])}]", axis=1)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=grade_stats['Label'], y=grade_stats['Count'], name="Volume", marker_color='rgb(55, 83, 109)'), secondary_y=False)
                fig.add_trace(go.Scatter(x=grade_stats['Label'], y=grade_stats['Bad Rate'], name="Bad Rate", mode='lines+markers', line=dict(color='rgb(26, 118, 255)')), secondary_y=True)
                fig.update_layout(title_text=f"Scorecard Performance ({sel_model})")
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("📥 Exportar Resultados")
                export_df = pd.DataFrame({
                    "Index": df_res.index,
                    "Target": df_res["Target"],
                    "y_pred": data_model["preds_total"],
                    "Score": df_res["Score"],
                    "Grade": df_res["Grade"]
                })
                csv = convert_df(export_df)
                
                st.download_button(
                    label="Download Scores Totais (CSV)",
                    data=csv,
                    file_name=f'scorecard_results_{sel_model}.csv',
                    mime='text/csv',
                )

    elif menu == "📑 Relatório Técnico":
        st.title("Relatório Técnico")
        if not st.session_state.model_ready:
            st.info("Corra o modelo para gerar o relatório.")
        else:
            tab_dq, tab_split, tab_uni, tab_multi, tab_cat, tab_ml = st.tabs(
                ["Data Quality", "Split", "Univariada", "Multivariada", "Categorização & IV", "ML Performance"]
            )
            
            with tab_dq:
                # ALTERAÇÃO: Sumário DQ
                st_dq = st.session_state.stats_dq
                if st_dq:
                    col_d1, col_d2, col_d3 = st.columns(3)
                    col_d1.metric("Variáveis Iniciais", st_dq["Entrada"])
                    col_d2.metric("Eliminadas (>20% missings)", st_dq["Eliminadas"])
                    col_d3.metric("Restantes", st_dq["Saída"])
                st.write("Estado das variáveis quanto a Missings (>20%):")
                st.dataframe(st.session_state.report_dq_missings.style.apply(lambda x: ['background: #ffcccc' if v == 'Eliminar' else '' for v in x], axis=1), use_container_width=True)
            
            with tab_split:
                st.table(st.session_state.report_split)
            
            with tab_uni:
                # ALTERAÇÃO: Sumário Univariada
                st_uni = st.session_state.stats_uni
                if st_uni:
                    col_u1, col_u2, col_u3 = st.columns(3)
                    col_u1.metric("Variáveis Testadas", st_uni["Entrada"])
                    col_u2.metric("Eliminadas (p-value > 0.05)", st_uni["Eliminadas"])
                    col_u3.metric("Restantes", st_uni["Saída"])
                st.dataframe(st.session_state.report_uni.style.apply(lambda x: ['background: #ffcccc' if v == 'Eliminar' else '' for v in x], axis=1), use_container_width=True)
            
            with tab_multi:
                # ALTERAÇÃO: Sumário Multivariada
                st_mul = st.session_state.stats_multi
                if st_mul:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Candidatas", st_mul["Entrada"])
                    col_m2.metric("Eliminadas (Correlação > 0.8)", st_mul["Eliminadas"])
                    col_m3.metric("Finais", st_mul["Saída"])
                
                st.write("Matriz de Correlação:")
                if 'corr_matrix' in st.session_state and st.session_state.corr_matrix is not None:
                    # ALTERAÇÃO: Altura da Matriz
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=st.session_state.corr_matrix.values,
                        x=st.session_state.corr_matrix.columns,
                        y=st.session_state.corr_matrix.index,
                        colorscale='Viridis'))
                    fig_corr.update_layout(height=800)
                    st.plotly_chart(fig_corr)
                st.divider()
                st.write("Variáveis Eliminadas por Correlação (Critério: Kendall Tau):")
                st.dataframe(st.session_state.report_multi_details, use_container_width=True)
            
            with tab_cat:
                # ALTERAÇÃO: Sumário Categorização
                st_cat = st.session_state.stats_cat
                if st_cat:
                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Tentativa Binning", st_cat["Entrada"])
                    col_c2.metric("Falharam (Monotonicidade/Size)", st_cat["Eliminadas"])
                    col_c3.metric("Variáveis Finais", st_cat["Saída"])
                
                st.header("Variáveis Finais (IV e Bins)")
                st.dataframe(st.session_state.report_iv, use_container_width=True)
                st.divider()
                st.subheader("Análise Gráfica de Bins e WoE")
                iv_vars = st.session_state.report_iv['Variável'].tolist()
                selected_iv_var = st.selectbox("Selecione a Variável para ver detalhe:", iv_vars)
                
                if selected_iv_var and selected_iv_var in st.session_state.iv_tables:
                    # CORREÇÃO: SORT BY WOE
                    df_viz = st.session_state.iv_tables[selected_iv_var].copy()
                    df_viz = df_viz.sort_values(by='WoE', ascending=True)
                    
                    fig_woe = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_woe.add_trace(go.Bar(x=df_viz['Value'], y=df_viz['Total'], name='Volume', marker_color='rgb(158,202,225)'), secondary_y=False)
                    fig_woe.add_trace(go.Scatter(x=df_viz['Value'], y=df_viz['WoE'], name='WoE', mode='lines+markers', line=dict(color='rgb(227,26,28)', width=3)), secondary_y=True)
                    fig_woe.update_layout(title=f"Binning & WoE: {selected_iv_var}")
                    fig_woe.update_yaxes(title_text="Volume", secondary_y=False)
                    fig_woe.update_yaxes(title_text="Weight of Evidence (WoE)", secondary_y=True)
                    st.plotly_chart(fig_woe, use_container_width=True)
                    st.dataframe(df_viz, use_container_width=True)

            with tab_ml:
                st.table(st.session_state.report_ml)
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.subheader("Curvas ROC")
                    fig_roc = go.Figure()
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    for name, (fpr, tpr, auc) in st.session_state.roc_curves.items():
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.2f})", mode='lines'))
                    fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR')
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                with col_g2:
                    st.subheader("Curvas KS (Comparativo)")
                    fig_ks = go.Figure()
                    for name, ks_data in st.session_state.ks_curves.items():
                        x_axis = np.linspace(0, 100, len(ks_data))
                        fig_ks.add_trace(go.Scatter(x=x_axis, y=ks_data['ks_dist'], name=f"{name}", mode='lines'))
                    
                    fig_ks.update_layout(title="Distância KS", xaxis_title="% População", yaxis_title="KS")
                    st.plotly_chart(fig_ks, use_container_width=True)
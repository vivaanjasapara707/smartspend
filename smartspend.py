import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SmartSpend Mumbai", page_icon="💰", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "SmartSpend_Mumbai_Datasets.xlsx")
MODEL_PATH = "models"

st.markdown("""
<style>
.stApp { background: #0a0a0f; color: #e8e8f0; }
.big-title { font-size: 3rem; font-weight: 900; color: #ffffff; margin-bottom: 0.5rem; }
.sub { font-size: 1rem; color: #d8d8e8; margin-bottom: 2rem; }
.cat-btn { background: #13131f; border: 1px solid #2a2a3e; border-radius: 16px; padding: 2rem; text-align: center; cursor: pointer; margin-bottom: 1rem; }
.cat-btn:hover { border-color: #6c63ff; }
.cat-title { font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-top: 0.5rem; }
.cat-sub { font-size: 0.85rem; color: #c0c0d8; margin-top: 0.3rem; }
.metric-box { background: #13131f; border: 1px solid #1e1e32; border-radius: 12px; padding: 1rem; text-align: center; }
.m-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; color: #c8c8d8; margin-bottom: 0.3rem; }
.m-value { font-size: 1.4rem; font-weight: 700; color: #ffffff; }
.signal-card { background: #13131f; border: 1px solid #1e1e32; border-radius: 16px; padding: 1.4rem; }
.badge-buy { background: #0d2e1a; color: #2ecc71; border: 1px solid #2ecc71; border-radius: 8px; padding: 0.3rem 0.9rem; font-weight: 700; font-size: 0.85rem; }
.badge-wait { background: #2e1a0d; color: #e67e22; border: 1px solid #e67e22; border-radius: 8px; padding: 0.3rem 0.9rem; font-weight: 700; font-size: 0.85rem; }
.badge-caution { background: #1a1a0d; color: #f1c40f; border: 1px solid #f1c40f; border-radius: 8px; padding: 0.3rem 0.9rem; font-weight: 700; font-size: 0.85rem; }
.section-hdr { font-size: 1.2rem; font-weight: 700; color: #ffffff; border-left: 3px solid #6c63ff; padding-left: 0.8rem; margin: 1.5rem 0 1rem 0; }
.insight-box { background: #13131f; border: 1px solid #1e1e32; border-radius: 16px; padding: 1.4rem; color: #d8d8e8; font-size: 0.95rem; line-height: 1.9; }
.insight-box b { color: #ffffff; }
.insight-box .highlight-green { color: #2ecc71; font-weight: 700; }
.insight-box .highlight-red { color: #e74c3c; font-weight: 700; }
.insight-box .highlight-yellow { color: #f1c40f; font-weight: 700; }
div.stButton > button { background: #13131f; color: #ffffff; border: 1px solid #2a2a3e; border-radius: 12px; padding: 0.6rem 1.5rem; font-size: 0.9rem; width: 100%; }
div.stButton > button:hover { border-color: #6c63ff; color: #6c63ff; }
label { color: #ffffff !important; }
.stSelectbox label p { color: #ffffff !important; }
.stRadio label p { color: #ffffff !important; }
div[data-baseweb="select"] span { color: #ffffff !important; }
div[data-baseweb="select"] { background: #13131f !important; border-color: #2a2a3e !important; }
.stRadio div[role="radiogroup"] label { color: #e8e8f0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_veg = pd.read_excel(DATA_PATH, sheet_name='1_Vegetables_Fruits')
    df_staples = pd.read_excel(DATA_PATH, sheet_name='2_Kitchen_Staples')
    df_qcomm = pd.read_excel(DATA_PATH, sheet_name='3_Quick_Commerce')
    df_flights = pd.read_excel(DATA_PATH, sheet_name='4_Flights')
    df_hotels = pd.read_excel(DATA_PATH, sheet_name='5_Hotels')
    df_petrol = pd.read_excel(DATA_PATH, sheet_name='6_Petrol_CNG')
    df_medicine = pd.read_excel(DATA_PATH, sheet_name='7_Medicines')
    df_realestate = pd.read_excel(DATA_PATH, sheet_name='8_Real_Estate')
    df_veg['Date'] = pd.to_datetime(df_veg['Date'])
    df_staples['Date'] = pd.to_datetime(df_staples['Date'])
    df_petrol['Date'] = pd.to_datetime(df_petrol['Date'])
    df_medicine['Date'] = pd.to_datetime(df_medicine['Date'])
    df_flights['Search_Date'] = pd.to_datetime(df_flights['Search_Date'])
    df_qcomm['Week'] = pd.to_datetime(df_qcomm['Week'])
    df_hotels['Week'] = pd.to_datetime(df_hotels['Week'])
    return df_veg, df_staples, df_qcomm, df_flights, df_hotels, df_petrol, df_medicine, df_realestate

df_veg, df_staples, df_qcomm, df_flights, df_hotels, df_petrol, df_medicine, df_realestate = load_data()

@st.cache_resource
def load_model(name):
    m = os.path.join(MODEL_PATH, f'{name}_model.pkl')
    s = os.path.join(MODEL_PATH, f'{name}_scaler.pkl')
    if os.path.exists(m) and os.path.exists(s):
        return joblib.load(m), joblib.load(s)
    return None, None

def smart_signal(current, mean_p, min_p, max_p, model=None, scaler=None, features=None):
    """
    Multi-factor smart buy signal. Uses model if available,
    falls back to statistical heuristics. Returns (prob, rec, rationale).
    """
    price_range = max_p - min_p if max_p != min_p else 1
    percentile_score = (current - min_p) / price_range  # 0 = at min, 1 = at max
    pct_vs_mean = (current - mean_p) / mean_p * 100

    # Try model first
    if model is not None and scaler is not None and features is not None:
        try:
            raw_prob = model.predict_proba(scaler.transform([features]))[0][1]
            # Blend model with statistical heuristics for more balanced output
            stat_prob = 1 - percentile_score  # higher = cheaper = more reason to buy
            blended = 0.55 * raw_prob + 0.45 * stat_prob
            prob = float(np.clip(blended, 0.05, 0.97))
        except:
            prob = float(1 - percentile_score)
    else:
        # Pure heuristic: combine percentile + distance from mean
        mean_factor = 1 - (pct_vs_mean / 100 + 1) / 2  # normalized
        prob = float(np.clip((1 - percentile_score) * 0.6 + max(0, -pct_vs_mean / 50) * 0.4, 0.05, 0.97))

    # Decision thresholds — more nuanced than a flat 0.6
    if prob >= 0.62:
        rec = 'BUY NOW'
    elif prob >= 0.42:
        rec = 'WAIT'
    else:
        rec = 'WAIT'

    return prob, rec, percentile_score, pct_vs_mean

def badge(rec):
    if rec == 'BUY NOW':
        return '<span class="badge-buy">✦ BUY NOW</span>'
    elif rec == 'CAUTION':
        return '<span class="badge-caution">◆ CAUTION</span>'
    return '<span class="badge-wait">◈ WAIT</span>'

def verdict_color(rec):
    if rec == 'BUY NOW': return '#2ecc71'
    if rec == 'CAUTION': return '#f1c40f'
    return '#e67e22'

def dark_chart():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#13131f')
    ax.set_facecolor('#13131f')
    ax.tick_params(colors='#9090b0', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2a2a3e')
    ax.grid(color='#1e1e32', linewidth=0.5, alpha=0.7)
    return fig, ax

def format_pct(val):
    sign = '+' if val >= 0 else ''
    return f"{sign}{val:.1f}%"

def percentile_label(p):
    if p <= 0.2: return "near all-time low 🟢"
    if p <= 0.4: return "below average range 🟢"
    if p <= 0.6: return "around average 🟡"
    if p <= 0.8: return "above average range 🟠"
    return "near all-time high 🔴"

def build_insight(name, unit, current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra_lines=None):
    """Build a rich multi-line insight HTML block."""
    pl = percentile_label(percentile_score)
    diff_abs = abs(current - mean_p)
    direction = "above" if pct_vs_mean >= 0 else "below"
    color = verdict_color(rec)

    saving_hint = ""
    if pct_vs_mean < -5:
        saving_hint = f"Buying today instead of waiting for peak saves approximately <b>₹{diff_abs:.0f}{unit}</b> per unit compared to the historical average."
    elif pct_vs_mean > 10:
        saving_hint = f"Waiting for prices to fall toward the ₹{mean_p:.0f}{unit} average could save you <b>₹{diff_abs:.0f}{unit}</b> per unit."
    else:
        saving_hint = f"Price is within <b>₹{diff_abs:.0f}{unit}</b> of the long-run average — a reasonable time to buy if you need it now."

    lines = [
        f"<b>Current price:</b> ₹{current:,.0f}{unit} — currently <b style='color:{color}'>{pl}</b>.",
        f"<b>vs. Historical average:</b> <b>{format_pct(pct_vs_mean)}</b> ({direction} the ₹{mean_p:,.0f}{unit} long-run mean).",
        f"<b>Price range context:</b> All-time low ₹{min_p:,.0f}{unit} → All-time high ₹{max_p:,.0f}{unit}.",
        saving_hint,
    ]
    if extra_lines:
        lines += extra_lines
    lines.append(f"<b>Model confidence:</b> {prob:.0%} — <b style='color:{color}'>{rec}</b>.")

    html = "<br>".join(lines)
    return f'<div class="insight-box">{html}</div>'

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ── HOME PAGE ──
if st.session_state.page == 'home':
    st.markdown('<div class="big-title">SmartSpend Mumbai</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Know exactly when to buy and when to wait — statistically proven.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Choose a Category</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">🥬</div><div class="cat-title">Vegetables</div><div class="cat-sub">Tomato · Onion · Potato</div></div>', unsafe_allow_html=True)
        if st.button('Open Vegetables', key='btn_veg'):
            st.session_state.page = 'vegetables'
            st.rerun()
    with col2:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">✈️</div><div class="cat-title">Flights</div><div class="cat-sub">Mumbai routes</div></div>', unsafe_allow_html=True)
        if st.button('Open Flights', key='btn_flights'):
            st.session_state.page = 'flights'
            st.rerun()
    with col3:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">📦</div><div class="cat-title">Quick Commerce</div><div class="cat-sub">Blinkit · Zepto · Instamart</div></div>', unsafe_allow_html=True)
        if st.button('Open Quick Commerce', key='btn_qc'):
            st.session_state.page = 'qcomm'
            st.rerun()
    with col4:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">⛽</div><div class="cat-title">Petrol & CNG</div><div class="cat-sub">Mumbai daily prices</div></div>', unsafe_allow_html=True)
        if st.button('Open Petrol', key='btn_petrol'):
            st.session_state.page = 'petrol'
            st.rerun()

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">🌾</div><div class="cat-title">Kitchen Staples</div><div class="cat-sub">Dal · Atta · Oil</div></div>', unsafe_allow_html=True)
        if st.button('Open Kitchen Staples', key='btn_staples'):
            st.session_state.page = 'staples'
            st.rerun()
    with col6:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">🏨</div><div class="cat-title">Hotels</div><div class="cat-sub">Goa · Lonavala · Alibaug</div></div>', unsafe_allow_html=True)
        if st.button('Open Hotels', key='btn_hotels'):
            st.session_state.page = 'hotels'
            st.rerun()
    with col7:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">💊</div><div class="cat-title">Medicines</div><div class="cat-sub">Stock before peak season</div></div>', unsafe_allow_html=True)
        if st.button('Open Medicines', key='btn_med'):
            st.session_state.page = 'medicines'
            st.rerun()
    with col8:
        st.markdown('<div class="cat-btn"><div style="font-size:2rem">🏠</div><div class="cat-title">Real Estate</div><div class="cat-sub">Mumbai micro markets</div></div>', unsafe_allow_html=True)
        if st.button('Open Real Estate', key='btn_re'):
            st.session_state.page = 'realestate'
            st.rerun()
# ── ASK ANYTHING ──
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">🤖 Ask Anything About Mumbai Prices</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#c8c8d8;font-size:0.9rem;margin-bottom:1rem;">Ask in plain English — e.g. "which vegetable is cheapest right now", "compare petrol prices month by month", "which quick commerce platform is cheapest for milk"</div>', unsafe_allow_html=True)

    user_question = st.text_input('', placeholder='Type your question here...', key='ask_anything')

    if user_question:
        with st.spinner('Thinking...'):
            try:
                from groq import Groq
                groq_client = Groq(api_key="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")

                # ── pre-process all dataframes to remove datetime issues ──
                def safe_df(df):
                    d = df.copy()
                    for col in d.columns:
                        if pd.api.types.is_datetime64_any_dtype(d[col]):
                            d[col] = d[col].astype(str)
                    return d

                safe_vars = {
                    'df_veg': safe_df(df_veg),
                    'df_staples': safe_df(df_staples),
                    'df_qcomm': safe_df(df_qcomm),
                    'df_flights': safe_df(df_flights),
                    'df_hotels': safe_df(df_hotels),
                    'df_petrol': safe_df(df_petrol),
                    'df_medicine': safe_df(df_medicine),
                    'df_realestate': safe_df(df_realestate),
                    'pd': pd,
                    'np': np,
                }

                schema_prompt = f"""You have access to these pandas dataframes already loaded in memory.
IMPORTANT: All date/week columns are already converted to strings like '2024-01' or '2024-01-15'. Do NOT use pd.to_datetime(). Do NOT compare dates. Just use string operations or ignore date columns entirely.

1. df_veg — columns: Date(str), Tomato_per_kg, Onion_per_kg, Potato_per_kg, Mango_per_kg, Orange_per_kg
2. df_staples — columns: Date(str), Tur_Dal_per_kg, Moong_Dal_per_kg, Urad_Dal_per_kg, Atta_per_kg, Rice_per_kg, Sugar_per_kg, Mustard_Oil_per_litre
3. df_qcomm — columns: Week(str), Platform(Blinkit/Zepto/Instamart), Amul_Milk_1L, Tata_Salt_1kg, Aashirvaad_Atta_5kg, Fortune_Mustard_Oil_1L, Parle_G_800g, Maggi_Noodles_12pack, Dettol_Soap_4pack, Colgate_200g
4. df_flights — columns: Search_Date(str), Route, Price_INR, Day_of_Week, Days_to_Departure, Is_Peak_Season, Is_Monsoon
5. df_hotels — columns: Week(str), Destination, Category(budget/mid/luxury), Price_per_Night_INR, Is_Peak_Season, Is_Off_Season, Weekend_Checkin
6. df_petrol — columns: Date(str), Petrol_per_litre_INR, CNG_per_kg_INR
7. df_medicine — columns: Date(str), Crocin_650mg_15tabs, Vicks_Vaporub_50g, Cetirizine_10mg_10tabs, ORS_Electral_21sachets, VitaminD3_60kIU_4tabs, Antifungal_cream_30g, Allegra_120mg_10tabs
8. df_realestate — columns: Micro_Market, Price_per_sqft_INR, RBI_Repo_Rate_pct, Unsold_Inventory_Index, Demand_Multiplier

User question: {user_question}

Write ONLY pandas code to answer this. Rules:
- Store final answer in variable called `result`
- result must be a number, string, Series, or DataFrame
- Only use: .mean() .min() .max() .sort_values() .head() .tail() .groupby() .iloc[] .loc[] .idxmin() .idxmax() .value_counts()
- Do NOT use pd.to_datetime(), do NOT compare dates, do NOT use rolling(), do NOT import anything, do NOT use plt or st
- For "cheapest right now" use .iloc[-1] to get latest row or .mean() across all rows
- Keep it simple — one or two lines max
- Write ONLY the code, no explanation, no backticks"""

                code_response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": schema_prompt}],
                    max_tokens=300,
                    temperature=0.1,
                )

                pandas_code = code_response.choices[0].message.content.strip()
                pandas_code = pandas_code.replace("```python", "").replace("```", "").strip()

                # try running generated code
                result = None
                exec_error = None
                try:
                    exec(pandas_code, safe_vars)
                    result = safe_vars.get('result', None)
                except Exception as e1:
                    exec_error = str(e1)
                    # fallback — ask for even simpler code
                    fallback_prompt = f"""Write ONE line of pandas code only. No dates. No rolling. Store in `result`.
Available dataframes: df_veg(Tomato_per_kg,Onion_per_kg,Potato_per_kg), df_staples(Tur_Dal_per_kg,Atta_per_kg,Rice_per_kg), df_petrol(Petrol_per_litre_INR,CNG_per_kg_INR), df_qcomm(Platform,Amul_Milk_1L), df_flights(Route,Price_INR), df_hotels(Destination,Price_per_Night_INR), df_medicine(Crocin_650mg_15tabs), df_realestate(Micro_Market,Price_per_sqft_INR)
Question: {user_question}
ONE LINE only, store in result, no backticks:"""
                    try:
                        fallback_response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": fallback_prompt}],
                            max_tokens=150,
                            temperature=0.1,
                        )
                        fallback_code = fallback_response.choices[0].message.content.strip().replace("```python","").replace("```","").strip()
                        exec(fallback_code, safe_vars)
                        result = safe_vars.get('result', None)
                    except Exception as e2:
                        result = None

                # if still no result, answer from knowledge alone
                if result is None:
                    conclusion_prompt = f"""You are SmartSpend Mumbai, a price intelligence assistant for Mumbai families.
The user asked: {user_question}
You could not run a data query but answer from general Mumbai price knowledge.
Give a helpful 3-sentence answer with realistic price ranges for Mumbai. End with one tip. Be conversational."""
                else:
                    result_str = str(result)
                    if len(result_str) > 1500:
                        result_str = result_str[:1500] + '...'
                    conclusion_prompt = f"""You are SmartSpend Mumbai, a price intelligence assistant for Mumbai families.
User asked: {user_question}
Data result: {result_str}
Write a clear helpful answer in 3-4 sentences. Start with the answer directly. Include specific numbers. End with one actionable tip. Use simple language, Hinglish is fine."""

                conclusion_response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": conclusion_prompt}],
                    max_tokens=250,
                    temperature=0.7,
                )
                conclusion = conclusion_response.choices[0].message.content.strip()

                st.markdown(f'''
                <div style="background:#0d1f2d;border:1px solid #1a3a5c;border-radius:16px;padding:1.6rem;margin-top:1rem;">
                    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;color:#5b9bd5;margin-bottom:0.8rem;">🤖 SmartSpend AI — Powered by LLaMA 3</div>
                    <div style="color:#e8f4ff;font-size:1rem;line-height:1.8;">{conclusion}</div>
                </div>
                ''', unsafe_allow_html=True)

                if result is not None and (isinstance(result, pd.DataFrame) or isinstance(result, pd.Series)):
                    with st.expander("See raw data"):
                        st.dataframe(result)

            except Exception as e:
                st.markdown(f'''
                <div style="background:#2e0d0d;border:1px solid #5c1a1a;border-radius:16px;padding:1.2rem;margin-top:1rem;">
                    <div style="color:#ff6b6b;font-size:0.9rem;">Could not process this question. Try rephrasing it. Error: {str(e)}</div>
                </div>
                ''', unsafe_allow_html=True)
# ── VEGETABLES PAGE ──
elif st.session_state.page == 'vegetables':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Vegetables & Fruits</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Mumbai APMC Vashi market prices — should you buy today or wait?</div>', unsafe_allow_html=True)

    veg_map = {
        'Tomato': ('Tomato_per_kg', 'veg_Tomato_per_kg', '#e74c3c'),
        'Onion': ('Onion_per_kg', 'veg_Onion_per_kg', '#e67e22'),
        'Potato': ('Potato_per_kg', 'veg_Potato_per_kg', '#f1c40f'),
        'Mango': ('Mango_per_kg', None, '#f39c12'),
        'Orange': ('Orange_per_kg', None, '#e67e22'),
    }
    selected = st.selectbox('Select vegetable / fruit', list(veg_map.keys()))
    col_name, model_key, color = veg_map[selected]

    veg = df_veg.copy()
    veg['Month'] = veg['Date'].dt.month
    veg['Is_Monsoon'] = veg['Month'].isin([6,7,8,9]).astype(int)
    veg['Rolling'] = veg[col_name].rolling(3).mean()
    veg['Vs_Mean'] = veg[col_name] - veg['Rolling']

    current = float(veg[col_name].dropna().iloc[-1])
    mean_p = float(veg[col_name].mean())
    min_p = float(veg[col_name].min())
    max_p = float(veg[col_name].max())
    std_p = float(veg[col_name].std())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Current Price</div><div class="m-value">₹{current:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">Historical Mean</div><div class="m-value">₹{mean_p:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">All-time Low</div><div class="m-value">₹{min_p:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">All-time High</div><div class="m-value">₹{max_p:.0f}/kg</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    fig, ax = dark_chart()
    series = veg.set_index('Date')[col_name].dropna()
    ax.plot(series.index, series.values, color=color, linewidth=2)
    ax.fill_between(series.index, series.values, alpha=0.08, color=color)
    ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:.0f}')
    ax.plot(series.rolling(3).mean().index, series.rolling(3).mean().values, color='#9b9bcc', linewidth=1.2, linestyle=':', alpha=0.7, label='3-period avg')
    ax.set_title(f'{selected} Price Trend — Mumbai APMC', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price (INR/kg)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    model_features = None
    m, s = (None, None)
    if model_key:
        m, s = load_model(model_key)
        if m:
            last = veg[['Month','Is_Monsoon','Vs_Mean',col_name]].dropna().iloc[-1]
            model_features = [float(last['Month']), float(last['Is_Monsoon']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    # Extra vegetable-specific lines
    zscore = (current - mean_p) / std_p if std_p > 0 else 0
    monsoon_now = veg['Month'].iloc[-1] in [6,7,8,9]
    extra = []
    extra.append(f"<b>Z-score vs mean:</b> {zscore:+.2f} standard deviations — {'unusually high, demand caution' if zscore > 1.5 else 'unusually low, strong buy signal' if zscore < -1.5 else 'within normal range'}.")
    if monsoon_now:
        extra.append("<b>Seasonal note:</b> Monsoon season (Jun–Sep) historically drives vegetable prices up 15–30% due to supply disruptions. Stock up before the peak if price is currently low.")
    else:
        extra.append("<b>Seasonal note:</b> Non-monsoon season typically means stable to lower prices at Mumbai APMC. Ideal window for bulk buying.")
    if pct_vs_mean < -10:
        extra.append(f"<b>Opportunity alert:</b> {selected} is currently more than 10% below its historical average. This is statistically a strong buying window — prices at this level have historically reverted upward within 2–4 weeks.")
    elif pct_vs_mean > 15:
        extra.append(f"<b>Caution:</b> {selected} is elevated above its normal range. Historically, prices this high have corrected downward within 2–3 weeks. Consider buying smaller quantities for now.")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected} — APMC Vashi</div>
            <div class="m-value">₹{current:.0f}/kg</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected, '/kg', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── PETROL PAGE ──
elif st.session_state.page == 'petrol':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Petrol & CNG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Mumbai daily fuel prices — PPAC data. Fill up now or wait for a revision?</div>', unsafe_allow_html=True)

    fuel = st.radio('Select fuel', ['Petrol', 'CNG'], horizontal=True)
    col_name = 'Petrol_per_litre_INR' if fuel == 'Petrol' else 'CNG_per_kg_INR'
    model_key = 'petrol' if fuel == 'Petrol' else 'cng'
    color = '#e74c3c' if fuel == 'Petrol' else '#3498db'
    unit = '/L' if fuel == 'Petrol' else '/kg'

    p = df_petrol.copy().sort_values('Date')
    current = float(p[col_name].iloc[-1])
    mean_p = float(p[col_name].mean())
    min_p = float(p[col_name].min())
    max_p = float(p[col_name].max())
    std_p = float(p[col_name].std())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Today</div><div class="m-value">₹{current:.2f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">3yr Mean</div><div class="m-value">₹{mean_p:.2f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">All-time Low</div><div class="m-value">₹{min_p:.2f}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">All-time High</div><div class="m-value">₹{max_p:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    series = p.set_index('Date')[col_name].resample('W').mean().dropna()
    fig, ax = dark_chart()
    ax.plot(series.index, series.values, color=color, linewidth=2)
    ax.fill_between(series.index, series.values, alpha=0.08, color=color)
    ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:.2f}')
    ax.set_title(f'Mumbai {fuel} Price — Weekly Average', color='#ffffff', fontsize=11)
    ax.set_ylabel(f'Price (INR{unit})', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    model_features = None
    m, s = load_model(model_key)
    if m:
        p2 = p.copy()
        p2['Rolling'] = p2[col_name].rolling(30).mean()
        p2['Vs_Mean'] = p2[col_name] - p2['Rolling']
        p2['Day_of_Week'] = p2['Date'].dt.dayofweek
        p2['Month'] = p2['Date'].dt.month
        last = p2[['Day_of_Week','Month','Vs_Mean',col_name]].dropna().iloc[-1]
        model_features = [float(last['Day_of_Week']), float(last['Month']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    zscore = (current - mean_p) / std_p if std_p > 0 else 0
    extra = []
    extra.append(f"<b>Z-score vs mean:</b> {zscore:+.2f} — {'significantly above long-run average, government revision may push prices lower' if zscore > 1 else 'near or below long-run average — relatively favorable pricing' if zscore < 0 else 'within 1 standard deviation of average'}.")
    extra.append(f"<b>Government revision pattern:</b> India's central government typically reviews fuel prices every 6–8 weeks. CNG prices in Mumbai also depend on MFGL's quarterly tariff revisions. If prices are above average and crude oil has softened globally, a downward revision is plausible.")
    if fuel == 'Petrol':
        extra.append("<b>Practical tip:</b> Fill a full tank rather than partial fills during favorable windows. For a 40L tank, even a ₹2–3/L saving equals ₹80–120 per fill — meaningful over a year.")
    else:
        extra.append("<b>CNG insight:</b> CNG prices in Mumbai have historically been more stable than petrol. High CNG prices relative to history often correct within 4–8 weeks following city gas distribution reviews.")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{fuel} — Mumbai</div>
            <div class="m-value">₹{current:.2f}{unit}</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(fuel, unit, current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── FLIGHTS PAGE ──
elif st.session_state.page == 'flights':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Flights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Mumbai departure routes — book now or wait for a lower fare?</div>', unsafe_allow_html=True)

    routes = sorted(df_flights['Route'].unique().tolist())
    selected_route = st.selectbox('Select route', routes)

    route_data = df_flights[df_flights['Route'] == selected_route].copy()
    current = float(route_data['Price_INR'].iloc[-30:].mean())
    mean_p = float(route_data['Price_INR'].mean())
    min_p = float(route_data['Price_INR'].min())
    max_p = float(route_data['Price_INR'].max())
    std_p = float(route_data['Price_INR'].std())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Recent Avg (30d)</div><div class="m-value">₹{current:,.0f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">Historical Mean</div><div class="m-value">₹{mean_p:,.0f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">Lowest Seen</div><div class="m-value">₹{min_p:,.0f}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">Highest Seen</div><div class="m-value">₹{max_p:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    route_weekly = route_data.groupby('Search_Date')['Price_INR'].mean()
    fig, ax = dark_chart()
    ax.plot(route_weekly.index, route_weekly.values, color='#6c63ff', linewidth=1.5, alpha=0.7)
    ax.plot(route_weekly.rolling(7).mean().index, route_weekly.rolling(7).mean().values, color='#ffffff', linewidth=2, label='7-day avg')
    ax.axhline(mean_p, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.5, label=f'All-time mean ₹{mean_p:,.0f}')
    ax.fill_between(route_weekly.index, route_weekly.values, alpha=0.06, color='#6c63ff')
    ax.set_title(f'{selected_route} — Price Trend', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price (INR)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Cheapest Day to Book</div>', unsafe_allow_html=True)
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_avg = route_data.groupby('Day_of_Week')['Price_INR'].mean()
    day_avg = day_avg.reindex([d for d in day_order if d in day_avg.index])
    cheapest_day = day_avg.idxmin()
    most_expensive_day = day_avg.idxmax()
    saving_by_day = float(day_avg[most_expensive_day] - day_avg[cheapest_day])

    fig2, ax2 = dark_chart()
    bar_colors = ['#2ecc71' if d == cheapest_day else '#e74c3c' if d == most_expensive_day else '#2a2a3e' for d in day_avg.index]
    ax2.bar(day_avg.index, day_avg.values, color=bar_colors, edgecolor='#1e1e32')
    ax2.set_title('Average Price by Day of Week', color='#ffffff', fontsize=10)
    ax2.set_ylabel('Avg Price (INR)', color='#9090b0', fontsize=9)
    plt.xticks(rotation=30, color='#9090b0')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
    st.markdown(f'<div style="color:#d0d0e0;font-size:0.9rem;margin:0.5rem 0 1rem;">💡 Cheapest day to book: <strong style="color:#2ecc71">{cheapest_day}</strong> — Most expensive: <strong style="color:#e74c3c">{most_expensive_day}</strong> — Day-of-week saving: <strong style="color:#ffffff">₹{saving_by_day:,.0f}</strong></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    m, s = load_model('flights_buy')
    if not m:
        m, s = load_model('flights')

    model_features = None
    if m:
        try:
            last = route_data[['Days_to_Departure','Is_Peak_Season','Is_Monsoon']].dropna().iloc[-1]
            day_dummies = pd.get_dummies(route_data['Day_of_Week'], prefix='day', drop_first=True)
            X = pd.concat([route_data[['Days_to_Departure','Is_Peak_Season','Is_Monsoon']].reset_index(drop=True), day_dummies.reset_index(drop=True)], axis=1).dropna()
            if hasattr(s, 'feature_names_in_'):
                for col in s.feature_names_in_:
                    if col not in X.columns:
                        X[col] = 0
                X = X[s.feature_names_in_]
            model_features = X.iloc[-1].tolist()
        except:
            model_features = None

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    zscore = (current - mean_p) / std_p if std_p > 0 else 0
    extra = []
    extra.append(f"<b>Booking window insight:</b> Airlines typically price lowest 3–6 weeks before departure. Booking on a <b style='color:#2ecc71'>{cheapest_day}</b> can save up to <b>₹{saving_by_day:,.0f}</b> vs the most expensive day ({most_expensive_day}).")
    extra.append(f"<b>Z-score:</b> {zscore:+.2f} — current fares are {'significantly elevated, a pullback is likely if you can wait 1–2 weeks' if zscore > 1.2 else 'near or below average — good time to lock in a ticket' if zscore < -0.3 else 'within normal range'}.")
    if 'Goa' in selected_route or 'goa' in selected_route.lower():
        extra.append("<b>Route note:</b> Mumbai–Goa is a high-demand leisure route. Prices spike during long weekends, December–January, and Holi. If travelling in those windows, book at least 21 days in advance.")
    extra.append(f"<b>Price protection tip:</b> If current fares are near the all-time low of ₹{min_p:,.0f}, book immediately — prices at this level rarely persist beyond 48–72 hours on competitive routes.")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected_route}</div>
            <div class="m-value">₹{current:,.0f}</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Best Day to Book</div>
            <div style="color:#2ecc71;font-size:1rem;font-weight:600;">{cheapest_day}</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected_route, '', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── QUICK COMMERCE PAGE ──
elif st.session_state.page == 'qcomm':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Quick Commerce</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Blinkit · Zepto · Instamart — which platform is cheapest today and should you order now?</div>', unsafe_allow_html=True)

    qc_products = {
        'Amul Milk 1L': 'Amul_Milk_1L',
        'Tata Salt 1kg': 'Tata_Salt_1kg',
        'Aashirvaad Atta 5kg': 'Aashirvaad_Atta_5kg',
        'Fortune Mustard Oil 1L': 'Fortune_Mustard_Oil_1L',
        'Parle G 800g': 'Parle_G_800g',
        'Maggi Noodles 12pack': 'Maggi_Noodles_12pack',
        'Dettol Soap 4pack': 'Dettol_Soap_4pack',
        'Colgate 200g': 'Colgate_200g',
    }
    selected = st.selectbox('Select product', list(qc_products.keys()))
    col_name = qc_products[selected]
    model_key = f'qc_{col_name}'

    latest_week = df_qcomm['Week'].max()
    latest = df_qcomm[df_qcomm['Week'] == latest_week]
    platform_prices = {}
    for plat in ['Blinkit','Zepto','Instamart']:
        row = latest[latest['Platform'] == plat]
        if len(row) > 0:
            platform_prices[plat] = float(row[col_name].values[0])

    cheapest = None
    saving = 0
    if platform_prices:
        cheapest = min(platform_prices, key=platform_prices.get)
        saving = max(platform_prices.values()) - min(platform_prices.values())
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        plat_colors = {'Blinkit': '#2ecc71', 'Zepto': '#e74c3c', 'Instamart': '#3498db'}
        for i, (plat, price) in enumerate(platform_prices.items()):
            is_cheap = plat == cheapest
            border = '#2ecc71' if is_cheap else '#1e1e32'
            with cols[i]:
                st.markdown(f'''
                <div style="background:#13131f;border:2px solid {border};border-radius:16px;padding:1.2rem;text-align:center;">
                    <div class="m-label">{plat}</div>
                    <div class="m-value" style="color:{plat_colors[plat]}">₹{price:.2f}</div>
                    {"<div style='color:#2ecc71;font-size:0.85rem;margin-top:0.5rem;font-weight:700;'>✦ CHEAPEST TODAY</div>" if is_cheap else ""}
                </div>''', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#d0d0e0;font-size:0.9rem;margin:1rem 0;">💡 Order from <strong style="color:#ffffff">{cheapest}</strong> — save <strong style="color:#2ecc71">₹{saving:.2f}</strong> vs most expensive platform</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Price Trend Across Platforms</div>', unsafe_allow_html=True)
    fig, ax = dark_chart()
    plat_colors_chart = {'Blinkit': '#2ecc71', 'Zepto': '#e74c3c', 'Instamart': '#3498db'}
    all_prices = []
    for plat in ['Blinkit','Zepto','Instamart']:
        subset = df_qcomm[df_qcomm['Platform'] == plat].groupby('Week')[col_name].mean()
        ax.plot(subset.index, subset.values, label=plat, color=plat_colors_chart[plat], linewidth=1.8)
        all_prices.extend(subset.values.tolist())
    overall_mean = float(np.mean(all_prices)) if all_prices else 0
    overall_min = float(np.min(all_prices)) if all_prices else 0
    overall_max = float(np.max(all_prices)) if all_prices else 0
    ax.axhline(overall_mean, color='#ffffff', linewidth=1, linestyle='--', alpha=0.3, label=f'Overall mean ₹{overall_mean:.2f}')
    ax.set_title(f'{selected} — Weekly Price by Platform', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price (INR)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    current_price = float(platform_prices.get(cheapest, overall_mean)) if cheapest else overall_mean
    m, s = load_model(model_key)

    model_features = None
    if m:
        qc = df_qcomm.copy()
        qc['Month'] = qc['Week'].dt.month
        qc['Is_Monsoon'] = qc['Month'].isin([6,7,8,9]).astype(int)
        qc_avg = qc.groupby('Week').agg(Price=(col_name,'mean'), Month=('Month','first'), Is_Monsoon=('Is_Monsoon','first'), Weekend_Sale=('Weekend_Sale_Active','first')).reset_index()
        qc_avg['Rolling'] = qc_avg['Price'].rolling(4).mean()
        qc_avg['Vs_Mean'] = qc_avg['Price'] - qc_avg['Rolling']
        last = qc_avg[['Month','Is_Monsoon','Weekend_Sale','Vs_Mean','Price']].dropna().iloc[-1]
        model_features = [float(last['Month']), float(last['Is_Monsoon']), float(last['Weekend_Sale']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current_price, overall_mean, overall_min, overall_max, m, s, model_features)

    extra = []
    if saving > 0 and cheapest:
        extra.append(f"<b>Platform arbitrage:</b> {cheapest} is currently <b>₹{saving:.2f} cheaper</b> than the most expensive platform for this product. Always verify delivery time and minimum order before switching.")
    extra.append(f"<b>Quick commerce pricing pattern:</b> Blinkit, Zepto and Instamart frequently run platform-exclusive sales on weekends (Sat–Sun) and mid-month. If the product is not urgent, checking back on a Friday night often reveals flash discounts of 5–15%.")
    extra.append(f"<b>Bulk buying signal:</b> {'At current pricing near the lower end of the historical range, stocking up for 2–4 weeks is a smart move — prices are unlikely to fall significantly further.' if percentile_score < 0.4 else 'Current price is above the lower range. Buy what you need now but avoid bulk stockpiling — better prices are likely in the next 1–2 weeks.'}")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected}</div>
            <div class="m-value">₹{current_price:.2f} <span style="font-size:0.75rem;color:#9090b0;">on {cheapest if cheapest else 'avg'}</span></div>
            <br>
            <div class="m-label">Platform Saving</div>
            <div style="color:#2ecc71;font-size:1rem;font-weight:600;">₹{saving:.2f}</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected, '', current_price, overall_mean, overall_min, overall_max, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── KITCHEN STAPLES PAGE ──
elif st.session_state.page == 'staples':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Kitchen Staples</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Monthly wholesale prices Mumbai — stock your pantry at the right time.</div>', unsafe_allow_html=True)

    staple_map = {
        'Tur Dal': ('Tur_Dal_per_kg', 'staple_Tur_Dal_per_kg', '#e74c3c'),
        'Moong Dal': ('Moong_Dal_per_kg', 'staple_Moong_Dal_per_kg', '#2ecc71'),
        'Urad Dal': ('Urad_Dal_per_kg', 'staple_Urad_Dal_per_kg', '#9b59b6'),
        'Atta': ('Atta_per_kg', 'staple_Atta_per_kg', '#f39c12'),
        'Rice': ('Rice_per_kg', 'staple_Rice_per_kg', '#3498db'),
        'Sugar': ('Sugar_per_kg', 'staple_Sugar_per_kg', '#1abc9c'),
        'Mustard Oil': ('Mustard_Oil_per_litre', 'staple_Mustard_Oil_per_litre', '#e67e22'),
    }
    selected = st.selectbox('Select staple', list(staple_map.keys()))
    col_name, model_key, color = staple_map[selected]

    st2 = df_staples.copy()
    st2['Month'] = st2['Date'].dt.month
    st2['Is_Monsoon'] = st2['Month'].isin([6,7,8,9]).astype(int)
    st2['Rolling'] = st2[col_name].rolling(3).mean()
    st2['Vs_Mean'] = st2[col_name] - st2['Rolling']

    current = float(st2[col_name].dropna().iloc[-1])
    mean_p = float(st2[col_name].mean())
    min_p = float(st2[col_name].min())
    max_p = float(st2[col_name].max())
    std_p = float(st2[col_name].std())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Current</div><div class="m-value">₹{current:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">3yr Mean</div><div class="m-value">₹{mean_p:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">All-time Low</div><div class="m-value">₹{min_p:.0f}/kg</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">All-time High</div><div class="m-value">₹{max_p:.0f}/kg</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    fig, ax = dark_chart()
    series = st2.set_index('Date')[col_name].dropna()
    ax.plot(series.index, series.values, color=color, linewidth=2)
    ax.fill_between(series.index, series.values, alpha=0.08, color=color)
    ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:.0f}')
    ax.set_title(f'{selected} Price Trend', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price (INR/kg)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    m, s = load_model(model_key)
    model_features = None
    if m:
        last = st2[['Month','Is_Monsoon','Vs_Mean',col_name]].dropna().iloc[-1]
        model_features = [float(last['Month']), float(last['Is_Monsoon']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    zscore = (current - mean_p) / std_p if std_p > 0 else 0
    extra = []
    extra.append(f"<b>Z-score vs mean:</b> {zscore:+.2f} — {'above normal range, government price control or import duty change could bring prices down — consider buying only near-term requirements' if zscore > 1.2 else 'at or below average — excellent window to stock up for 1–2 months' if zscore < -0.5 else 'within normal trading range'}.")

    staple_notes = {
        'Tur Dal': "Tur Dal prices are heavily influenced by kharif harvest (Oct–Nov) and government MSP decisions. Post-harvest months (Nov–Jan) are historically the cheapest window to buy.",
        'Moong Dal': "Moong Dal has two harvest seasons (summer & kharif). Prices typically ease in March–April and again in October. Monsoon period sees elevated prices.",
        'Urad Dal': "Urad Dal is a kharif crop — prices tend to peak pre-harvest (Aug–Sep) and ease in Nov–Dec. Stock up during October–November.",
        'Atta': "Wheat atta prices track rabi wheat harvest (Apr–Jun). Post-harvest months of June–August are historically cheapest. Pre-monsoon stocking is a well-established household strategy.",
        'Rice': "Rice prices ease after kharif harvest (Oct–Dec). Monsoon months can see supply disruptions. October–November is the ideal bulk buying window for rice.",
        'Sugar': "Sugar prices in India are largely government-regulated (FRP/SAP). Price spikes are usually short-lived. Buying during June–September when off-season supply tightens isn't ideal.",
        'Mustard Oil': "Mustard oil prices track the rabi harvest of mustard (Feb–Apr). Prices typically soften from May onward. Post-May is historically the best window to buy in bulk.",
    }
    if selected in staple_notes:
        extra.append(f"<b>Seasonal harvest pattern:</b> {staple_notes[selected]}")

    bulk_saving = (mean_p - current) * 10 if current < mean_p else 0
    if bulk_saving > 0:
        extra.append(f"<b>Bulk buying opportunity:</b> Buying 10kg at today's price vs. waiting for the average saves approximately <b>₹{bulk_saving:.0f}</b>. Given the shelf life of {selected}, stocking 2–3 months ahead is practical.")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected}</div>
            <div class="m-value">₹{current:.0f}/kg</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected, '/kg', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── HOTELS PAGE ──
elif st.session_state.page == 'hotels':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Hotels</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Goa · Lonavala · Mahabaleshwar · Alibaug — book now or wait for better rates?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: selected_dest = st.selectbox('Destination', sorted(df_hotels['Destination'].unique()))
    with col2: selected_cat = st.selectbox('Category', ['budget','mid','luxury'])

    h = df_hotels[(df_hotels['Destination']==selected_dest) & (df_hotels['Category']==selected_cat)].copy()

    if len(h) == 0:
        st.warning("No data for this combination.")
    else:
        current = float(h['Price_per_Night_INR'].iloc[-1])
        mean_p = float(h['Price_per_Night_INR'].mean())
        min_p = float(h['Price_per_Night_INR'].min())
        max_p = float(h['Price_per_Night_INR'].max())
        std_p = float(h['Price_per_Night_INR'].std())
        peak_avg = float(h[h['Is_Peak_Season']==1]['Price_per_Night_INR'].mean()) if h['Is_Peak_Season'].sum() > 0 else mean_p
        offpeak_avg = float(h[h['Is_Peak_Season']==0]['Price_per_Night_INR'].mean()) if (h['Is_Peak_Season']==0).sum() > 0 else mean_p

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Current</div><div class="m-value">₹{current:,.0f}/night</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><div class="m-label">Average</div><div class="m-value">₹{mean_p:,.0f}/night</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><div class="m-label">Peak Season</div><div class="m-value">₹{peak_avg:,.0f}/night</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-box"><div class="m-label">Off Season</div><div class="m-value">₹{offpeak_avg:,.0f}/night</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        series = h.set_index('Week')['Price_per_Night_INR']
        fig, ax = dark_chart()
        ax.plot(series.index, series.values, color='#9b59b6', linewidth=2)
        ax.fill_between(series.index, series.values, alpha=0.08, color='#9b59b6')
        ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:,.0f}')
        ax.axhline(peak_avg, color='#e74c3c', linewidth=1, linestyle=':', alpha=0.5, label=f'Peak avg ₹{peak_avg:,.0f}')
        ax.axhline(offpeak_avg, color='#2ecc71', linewidth=1, linestyle=':', alpha=0.5, label=f'Off-peak avg ₹{offpeak_avg:,.0f}')
        ax.set_title(f'{selected_dest} {selected_cat.title()} — Price Trend', color='#ffffff', fontsize=11)
        ax.set_ylabel('Price per night (INR)', color='#9090b0', fontsize=9)
        ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

        m, s = load_model(f'hotel_{selected_cat}')
        model_features = None
        if m:
            h2 = df_hotels[df_hotels['Category']==selected_cat].copy()
            h2['Month'] = h2['Week'].dt.month
            h2['Rolling'] = h2['Price_per_Night_INR'].rolling(4).mean()
            h2['Vs_Mean'] = h2['Price_per_Night_INR'] - h2['Rolling']
            last = h2[['Month','Is_Peak_Season','Is_Off_Season','Weekend_Checkin','Vs_Mean','Price_per_Night_INR']].dropna().iloc[-1]
            model_features = [float(last['Month']), float(last['Is_Peak_Season']), float(last['Is_Off_Season']), float(last['Weekend_Checkin']), float(last['Vs_Mean'])]

        prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

        peak_saving = peak_avg - offpeak_avg
        zscore = (current - mean_p) / std_p if std_p > 0 else 0
        extra = []
        extra.append(f"<b>Peak vs off-season spread:</b> ₹{peak_saving:,.0f}/night difference — planning a {selected_dest} trip in the off-season saves approximately <b>₹{peak_saving*2:,.0f}+</b> on a 2-night stay.")
        extra.append(f"<b>Z-score:</b> {zscore:+.2f} — current rates are {'near peak pricing, consider shifting dates to a weekday or off-season month for significant savings' if zscore > 1 else 'near the lowest range we have seen — an excellent time to book, especially for peak-season dates' if zscore < -0.5 else 'within the normal trading range'}.")
        dest_notes = {
            'Goa': "Goa peaks Dec–Jan (Christmas/NYE) and long weekends. Shoulder season (Feb–Mar and Oct–Nov) offers near-peak experience at 30–40% lower rates.",
            'Lonavala': "Lonavala prices peak during monsoon (Jul–Sep) for the waterfall/fog experience and long weekends. Weekday bookings in Jan–Feb offer the best value.",
            'Mahabaleshwar': "Mahabaleshwar peaks May–Jun (summer escape) and Dec–Jan. March and September are sweet-spot months — pleasant weather, lower crowds, better rates.",
            'Alibaug': "Alibaug is extremely weekend-dependent. Midweek stays can be 40–60% cheaper. Avoid Republic Day, Holi, and Diwali weekends for budget travel.",
        }
        if selected_dest in dest_notes:
            extra.append(f"<b>Destination intelligence:</b> {dest_notes[selected_dest]}")
        if selected_cat == 'luxury':
            extra.append("<b>Luxury booking tip:</b> Luxury properties often release last-minute discounts 5–7 days before check-in if occupancy is low. For flexible travellers, waiting until that window can yield 20–35% savings.")

        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f'''<div class="signal-card">
                <div class="m-label">{selected_dest} — {selected_cat.title()}</div>
                <div class="m-value">₹{current:,.0f}/night</div>
                <br>
                <div class="m-label">Historical Percentile</div>
                <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
                <br>
                <div class="m-label">Peak vs Off-season</div>
                <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">₹{peak_saving:,.0f}/night</div>
                <br>
                <div class="m-label">Confidence</div>
                <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
                <br>{badge(rec)}
            </div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(build_insight(f'{selected_dest} {selected_cat.title()}', '/night', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── MEDICINES PAGE ──
elif st.session_state.page == 'medicines':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Medicines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">NPPA regulated prices — know when to stock up before seasonal demand spikes.</div>', unsafe_allow_html=True)

    med_map = {
        'Crocin 650mg': ('Crocin_650mg_15tabs', 'med_Crocin_650mg_15tabs', 'Crocin_650mg_15tabs_Is_Peak_Season'),
        'Vicks Vaporub 50g': ('Vicks_Vaporub_50g', 'med_Vicks_Vaporub_50g', 'Vicks_Vaporub_50g_Is_Peak_Season'),
        'Cetirizine 10mg': ('Cetirizine_10mg_10tabs', 'med_Cetirizine_10mg_10tabs', 'Cetirizine_10mg_10tabs_Is_Peak_Season'),
        'ORS Electral': ('ORS_Electral_21sachets', 'med_ORS_Electral_21sachets', 'ORS_Electral_21sachets_Is_Peak_Season'),
        'Vitamin D3': ('VitaminD3_60kIU_4tabs', 'med_VitaminD3_60kIU_4tabs', 'VitaminD3_60kIU_4tabs_Is_Peak_Season'),
        'Antifungal Cream': ('Antifungal_cream_30g', 'med_Antifungal_cream_30g', 'Antifungal_cream_30g_Is_Peak_Season'),
        'Allegra 120mg': ('Allegra_120mg_10tabs', 'med_Allegra_120mg_10tabs', 'Allegra_120mg_10tabs_Is_Peak_Season'),
    }
    selected = st.selectbox('Select medicine', list(med_map.keys()))
    col_name, model_key, peak_col = med_map[selected]

    med = df_medicine.copy()
    med['Month'] = med['Date'].dt.month
    med['Rolling'] = med[col_name].rolling(3).mean()
    med['Vs_Mean'] = med[col_name] - med['Rolling']
    med['Is_Peak'] = med[peak_col].astype(int)

    current = float(med[col_name].dropna().iloc[-1])
    mean_p = float(med[col_name].mean())
    min_p = float(med[col_name].min())
    max_p = float(med[col_name].max())
    std_p = float(med[col_name].std())
    peak_avg = float(med[med['Is_Peak']==1][col_name].mean()) if med['Is_Peak'].sum() > 0 else mean_p
    offpeak_avg = float(med[med['Is_Peak']==0][col_name].mean()) if (med['Is_Peak']==0).sum() > 0 else mean_p
    is_peak_now = bool(med['Is_Peak'].iloc[-1])

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Current</div><div class="m-value">₹{current:.2f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">Average</div><div class="m-value">₹{mean_p:.2f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">Peak Season</div><div class="m-value">₹{peak_avg:.2f}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">Off Season</div><div class="m-value">₹{offpeak_avg:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    fig, ax = dark_chart()
    series = med.set_index('Date')[col_name].dropna()
    ax.plot(series.index, series.values, color='#2ecc71', linewidth=2)
    ax.fill_between(series.index, series.values, alpha=0.08, color='#2ecc71')
    ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:.2f}')
    ax.axhline(peak_avg, color='#e74c3c', linewidth=1, linestyle=':', alpha=0.5, label=f'Peak avg ₹{peak_avg:.2f}')
    ax.set_title(f'{selected} — Price Trend', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price (INR)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    m, s = load_model(model_key)
    model_features = None
    if m:
        last = med[['Month','Is_Peak','Vs_Mean',col_name]].dropna().iloc[-1]
        model_features = [float(last['Month']), float(last['Is_Peak']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    peak_saving = peak_avg - offpeak_avg
    extra = []
    extra.append(f"<b>Currently in peak season:</b> {'Yes ⚠️ — demand is elevated and prices may be at or near their seasonal high. Stock what you need now but avoid over-buying.' if is_peak_now else 'No ✅ — off-season pricing is in effect. This is the ideal window to stock up before demand rises.'}")
    extra.append(f"<b>Peak vs off-season cost difference:</b> ₹{peak_saving:.2f} per pack. Buying 3 packs today during off-season vs peak season saves approximately <b>₹{peak_saving*3:.2f}</b>.")

    med_notes = {
        'Crocin 650mg': "Crocin demand peaks during monsoon (Jun–Sep) due to fever/flu season. Stock up in April–May before the monsoon onset.",
        'Vicks Vaporub 50g': "Vicks Vaporub peaks in Oct–Feb during cold/winter season. August–September is the smart buy window.",
        'Cetirizine 10mg': "Cetirizine (antihistamine) peaks during spring pollination (Feb–Mar) and monsoon (Jul–Aug). January is the ideal stocking month.",
        'ORS Electral': "ORS demand peaks heavily during summer (Apr–Jun) and gastroenteritis season (Jul–Aug). Buy in February–March to avoid peak pricing.",
        'Vitamin D3': "Vitamin D3 demand is year-round but spikes in winter (Nov–Jan) when sun exposure drops. October is the smart pre-winter stockpile month.",
        'Antifungal Cream': "Antifungal creams peak in monsoon (Jun–Sep) due to humidity-driven infections. April–May is the optimal pre-monsoon stocking window.",
        'Allegra 120mg': "Allegra (fexofenadine) peaks during allergy season (Feb–Apr and Oct). December–January is the ideal pre-season buying window.",
    }
    if selected in med_notes:
        extra.append(f"<b>Seasonal intelligence:</b> {med_notes[selected]}")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected}</div>
            <div class="m-value">₹{current:.2f}</div>
            <br>
            <div class="m-label">Season Status</div>
            <div style="color:{'#e74c3c' if is_peak_now else '#2ecc71'};font-size:0.9rem;font-weight:600;">{'🔴 Peak Season' if is_peak_now else '🟢 Off Season'}</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected, '', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

# ── REAL ESTATE PAGE ──
elif st.session_state.page == 'realestate':
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('<div class="big-title">Real Estate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Mumbai micro-market price per sqft — macro signals and market timing analysis.</div>', unsafe_allow_html=True)

    markets = sorted(df_realestate['Micro_Market'].unique())
    selected_market = st.selectbox('Select micro market', markets)

    re = df_realestate[df_realestate['Micro_Market']==selected_market].copy()
    current = float(re['Price_per_sqft_INR'].iloc[-1])
    mean_p = float(re['Price_per_sqft_INR'].mean())
    min_p = float(re['Price_per_sqft_INR'].min())
    max_p = float(re['Price_per_sqft_INR'].max())
    std_p = float(re['Price_per_sqft_INR'].std())
    current_repo = float(re['RBI_Repo_Rate_pct'].iloc[-1])
    prev_repo = float(re['RBI_Repo_Rate_pct'].iloc[-2]) if len(re) > 1 else current_repo

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="m-label">Current/sqft</div><div class="m-value">₹{current:,.0f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="m-label">Average</div><div class="m-value">₹{mean_p:,.0f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="m-label">All-time Low</div><div class="m-value">₹{min_p:,.0f}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="m-label">RBI Repo Rate</div><div class="m-value">{current_repo:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    fig, ax = dark_chart()
    ax.bar(range(len(re)), re['Price_per_sqft_INR'].values, color='#6c63ff', alpha=0.8, edgecolor='#1e1e32')
    ax.axhline(mean_p, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4, label=f'Mean ₹{mean_p:,.0f}')
    ax.set_title(f'{selected_market} — Price per sqft Over Time', color='#ffffff', fontsize=11)
    ax.set_ylabel('Price per sqft (INR)', color='#9090b0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#13131f', labelcolor='#c8c8e8')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Market Comparison</div>', unsafe_allow_html=True)
    market_latest = df_realestate.groupby('Micro_Market')['Price_per_sqft_INR'].last().sort_values()
    fig2, ax2 = dark_chart()
    bar_colors = ['#6c63ff' if m == selected_market else '#2a2a3e' for m in market_latest.index]
    ax2.barh(market_latest.index, market_latest.values, color=bar_colors, edgecolor='#1e1e32')
    ax2.set_title('Latest Price per sqft by Market', color='#ffffff', fontsize=10)
    ax2.set_xlabel('Price per sqft (INR)', color='#9090b0', fontsize=9)
    ax2.tick_params(colors='#c0c0d8', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-hdr">Buy Signal & Analysis</div>', unsafe_allow_html=True)

    m, s = load_model('realestate')
    model_features = None
    if m:
        re_all = df_realestate.copy()
        re_all['Rolling'] = re_all['Price_per_sqft_INR'].rolling(4).mean()
        re_all['Vs_Mean'] = re_all['Price_per_sqft_INR'] - re_all['Rolling']
        re_all['High_Demand'] = (re_all['Demand_Multiplier'] > re_all['Demand_Multiplier'].median()).astype(int)
        last = re_all[['RBI_Repo_Rate_pct','Unsold_Inventory_Index','High_Demand','Vs_Mean','Price_per_sqft_INR']].dropna().iloc[-1]
        model_features = [float(last['RBI_Repo_Rate_pct']), float(last['Unsold_Inventory_Index']), float(last['High_Demand']), float(last['Vs_Mean'])]

    prob, rec, percentile_score, pct_vs_mean = smart_signal(current, mean_p, min_p, max_p, m, s, model_features)

    repo_trend = "falling" if current_repo < prev_repo else "rising" if current_repo > prev_repo else "stable"
    zscore = (current - mean_p) / std_p if std_p > 0 else 0
    typical_flat_1bhk = current * 450
    typical_flat_2bhk = current * 650

    extra = []
    extra.append(f"<b>RBI Repo Rate:</b> {current_repo:.1f}% (trend: {repo_trend}). {'A falling repo rate typically stimulates demand and pushes real estate prices up over 6–12 months — buying now locks in pre-rally pricing.' if repo_trend == 'falling' else 'Rising rates increase EMI burden, which can soften demand and price growth — buyers have negotiating leverage.' if repo_trend == 'rising' else 'Stable rates mean the market is driven by local supply/demand fundamentals.'}")
    extra.append(f"<b>Affordability context:</b> At ₹{current:,.0f}/sqft, a typical 450 sqft 1BHK in {selected_market} costs approximately <b>₹{typical_flat_1bhk/1e7:.2f} Cr</b>, and a 650 sqft 2BHK approximately <b>₹{typical_flat_2bhk/1e7:.2f} Cr</b>.")
    extra.append(f"<b>Z-score vs history:</b> {zscore:+.2f} — {selected_market} is {'significantly above its historical mean — price discovery phase, higher risk for near-term buyers' if zscore > 1.2 else 'near or below its historical mean — historically, entry at this level has generated positive returns over a 3–5 year horizon' if zscore < 0 else 'within normal range — neither a clear bargain nor overpriced'}.")
    extra.append(f"<b>Market comparison:</b> {selected_market} ranks {'among the most expensive' if list(market_latest.index).index(selected_market) > len(market_latest)*0.7 else 'in the mid-range' if list(market_latest.index).index(selected_market) > len(market_latest)*0.3 else 'among the most affordable'} micro-markets in the dataset. Consider adjacent areas for better value if price is a constraint.")

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(f'''<div class="signal-card">
            <div class="m-label">{selected_market}</div>
            <div class="m-value">₹{current:,.0f}/sqft</div>
            <br>
            <div class="m-label">Repo Rate</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{current_repo:.1f}% ({repo_trend})</div>
            <br>
            <div class="m-label">Historical Percentile</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{percentile_score*100:.0f}th</div>
            <br>
            <div class="m-label">Confidence</div>
            <div style="color:#d0d0e8;font-size:1rem;font-weight:600;">{prob:.0%}</div>
            <br>{badge(rec)}
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(build_insight(selected_market, '/sqft', current, mean_p, min_p, max_p, prob, rec, percentile_score, pct_vs_mean, extra), unsafe_allow_html=True)

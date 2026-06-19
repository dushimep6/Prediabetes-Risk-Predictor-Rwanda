# =============================================================================
# tool_v30.py — Rwanda Prediabetes Risk Predictor
# Rwanda Prediabetes Risk Assessment Tool
# Rwanda WHO STEP Survey 2021-2022
# Model: Logistic Regression | AUC=0.9885
# Validated: Rwanda national population
# v28 changes: Single consistent threshold system (model = display)
#              Three zones only: Low / Moderate / High
#              Gauge arcs updated to match thresholds exactly
#              vhi zone removed and merged into High
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, joblib, time
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Rwanda Prediabetes Risk Predictor",
    page_icon="💚", layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
:root{
  --navy:#0B2545;--teal:#1A5276;--teal2:#1F618D;
  --green:#1A7A3C;--green-lt:#EAF7EF;--green-dk:#145A2E;
  --yellow:#B7770D;--yellow-lt:#FEF9E7;--yellow-bd:#F0D060;
  --red:#922B21;--red-lt:#FDEDEC;--red-bd:#E57373;
  --border:#D5DCE8;--text:#0B1E30;--text2:#3D5166;--text3:#7A8FA6;
  --bg:#F4F6FA;--card:white;--shadow:0 2px 12px rgba(11,37,69,0.08)
}
html,body,[class*="css"]{
  font-family:"DM Sans",sans-serif!important;
  background:var(--bg)!important;color:var(--text)!important}
.main .block-container{padding-top:0!important;padding-bottom:3rem;max-width:780px}
#MainMenu,footer,header,.stDeployButton{visibility:hidden;display:none}
.hdr-blue{background:#00A1DE;padding:20px 28px;text-align:center}
.hdr-title{font-family:"DM Serif Display",serif;font-size:22px;
  font-weight:400;color:white;letter-spacing:0.3px;
  line-height:1.3;margin:0;text-shadow:0 1px 4px rgba(0,0,0,0.15)}
.hdr-yellow{background:#FAD201;height:10px}
.hdr-green{background:#20603D;padding:8px 0;text-align:center}
.lang-row{display:flex;justify-content:center;gap:10px}
.lbtn{background:transparent!important;
  border:1.5px solid rgba(255,255,255,0.6)!important;
  color:white!important;padding:5px 22px;border-radius:4px;
  font-size:12px;font-weight:600;letter-spacing:0.5px;
  text-decoration:none!important;cursor:pointer;display:inline-block}
.lbtn:hover{background:rgba(255,255,255,0.15)!important}
.lbtn.on{background:rgba(255,255,255,0.22)!important;
  border-color:white!important;font-weight:700}
.slbl{font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;color:var(--teal2);
  margin:18px 0 10px;padding-bottom:5px;
  border-bottom:1px solid var(--border)}
.card{background:var(--card);border-radius:8px;
  border:1px solid var(--border);padding:16px 20px;
  margin:8px 0;box-shadow:var(--shadow)}
label,div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label{
  font-family:"DM Sans",sans-serif!important;font-weight:600!important;
  font-size:11px!important;color:var(--text2)!important;
  text-transform:uppercase!important;letter-spacing:0.8px!important}
div[data-testid="stNumberInput"] input{
  font-family:"DM Sans",sans-serif!important;font-size:20px!important;
  font-weight:700!important;color:var(--navy)!important;
  text-align:center!important;border:2px solid var(--border)!important;
  border-radius:6px!important}
.stButton>button{
  background:var(--navy)!important;color:white!important;
  border:none!important;border-radius:6px!important;
  font-family:"DM Sans",sans-serif!important;font-weight:700!important;
  font-size:13px!important;letter-spacing:1.5px!important;
  text-transform:uppercase!important;padding:14px 20px!important;
  width:100%!important;margin-top:6px!important;
  box-shadow:0 4px 14px rgba(11,37,69,0.25)!important}
.stButton>button:hover{background:var(--teal)!important}
div[data-testid="column"]:last-child .stButton>button{
  background:white!important;color:var(--navy)!important;
  border:2px solid var(--navy)!important;box-shadow:none!important}
div[data-testid="column"]:last-child .stButton>button:hover{
  background:var(--navy)!important;color:white!important}
.risk-banner{border-radius:8px;padding:20px 24px;
  margin:10px 0;border-left:5px solid}
.risk-low {background:var(--green-lt);border-color:var(--green)}
.risk-mod {background:var(--yellow-lt);border-color:var(--yellow)}
.risk-high{background:var(--red-lt);border-color:var(--red)}
.risk-pct {font-family:"DM Serif Display",serif;
  font-size:48px;line-height:1;margin-bottom:2px}
.risk-label{font-size:14px;font-weight:700;
  text-transform:uppercase;letter-spacing:1.5px}
.risk-onein{font-size:16px;font-weight:600;margin:4px 0;color:var(--text2)}
.risk-msg{font-size:13px;line-height:1.7;margin-top:8px;padding-top:8px;
  border-top:1px solid rgba(0,0,0,0.08)}
.rbar-bg{background:#E8ECF2;border-radius:4px;height:8px;margin:8px 0}
.rbar{height:100%;border-radius:4px;transition:width 0.8s ease}
.abadge{display:inline-block;padding:2px 8px;border-radius:3px;
  font-size:11px;font-weight:700;margin-left:6px;vertical-align:middle}
.abadge-alert{background:#FFE0D0;color:#7D3300;border:1px solid #FFB899}
.abadge-warn {background:#FFF3CD;color:#856404;border:1px solid #FFEAA7}
.abadge-ok   {background:#D4EDDA;color:#155724;border:1px solid #B8DFC9}
.abadge-info {background:#D1ECF1;color:#0C5460;border:1px solid #BEE5EB}
.alert-item{display:flex;align-items:flex-start;gap:10px;
  padding:8px 0;border-bottom:1px solid #F0F4F8}
.alert-item:last-child{border-bottom:none}
.alert-dot{width:8px;height:8px;border-radius:50%;
  flex-shrink:0;margin-top:5px}
.alert-name{font-size:13px;font-weight:700;color:var(--navy);margin-bottom:1px}
.alert-val{font-size:12px;color:var(--text2)}
.rec-item{padding:10px 0;border-bottom:1px solid #F0F4F8}
.rec-item:last-child{border-bottom:none}
.rec-num{display:inline-flex;align-items:center;justify-content:center;
  width:22px;height:22px;border-radius:50%;font-size:11px;
  font-weight:700;color:white;margin-right:8px;flex-shrink:0}
.rec-title{font-size:13px;font-weight:700;color:var(--navy);margin-bottom:2px}
.rec-desc{font-size:12px;color:var(--text2);line-height:1.6;margin-left:30px}
.cbar-wrap{margin:10px 0}
.cbar-row{display:flex;align-items:center;gap:8px;margin:4px 0}
.cbar-lbl{font-size:11px;font-weight:600;color:var(--text2);width:130px;flex-shrink:0}
.cbar-bg{background:#E8ECF2;border-radius:4px;height:12px;flex:1}
.cbar-fill{height:100%;border-radius:4px}
.cbar-pct{font-size:11px;font-weight:700;width:40px;text-align:right}
.qdots{display:flex;gap:5px;margin:6px 0 2px}
.qdot{height:5px;flex:1;border-radius:3px;background:var(--border);max-width:80px}
.qdot.on{background:var(--navy)}
.qlbl{font-size:10px;font-weight:600;letter-spacing:1px;
  text-transform:uppercase;color:var(--text3)}
.stTabs [data-baseweb="tab-list"]{
  background:var(--card)!important;border-radius:6px!important;
  padding:4px!important;gap:2px!important;
  border:1px solid var(--border)!important;
  box-shadow:var(--shadow)!important;
  justify-content:center!important;
  width:fit-content!important;margin:0 auto!important}
.stTabs [data-baseweb="tab"]{border-radius:4px!important;
  font-family:"DM Sans",sans-serif!important;font-weight:600!important;
  font-size:11px!important;padding:8px 28px!important;
  color:var(--text2)!important;letter-spacing:0.5px!important;
  text-transform:uppercase!important}
.stTabs [aria-selected="true"]{background:var(--navy)!important;
  color:white!important;box-shadow:0 2px 8px rgba(11,37,69,0.2)!important}
.disc{font-size:11px;color:var(--text3);text-align:center;
  padding:12px;border-top:1px solid var(--border);
  margin-top:16px;line-height:1.6}
@media print {
  * { visibility:hidden!important }
  .hdr-blue,.risk-banner,.risk-label,.risk-pct,
  .risk-onein,.risk-msg,.rbar-bg,.rbar,
  .card,.cbar-wrap,.cbar-row,.cbar-lbl,
  .cbar-bg,.cbar-fill,.cbar-pct,
  .alert-item,.alert-dot,.alert-name,.alert-val,
  .abadge,.rec-item,.rec-num,.rec-title,.rec-desc,
  .disc,.slbl,.hdr-title { visibility:visible!important }
  body { background:white!important }
  .main .block-container { padding:10px!important;max-width:100%!important }
  .hdr-blue { -webkit-print-color-adjust:exact!important;
    print-color-adjust:exact!important;position:fixed;top:0;width:100% }
  @page { margin:1cm;size:A4 portrait }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TRANSLATIONS
# =============================================================================
TX = {
"en": {
  "title":   "Prediabetes Risk Self-Assessment Tool",
  "meta":    "Rwanda WHO STEP Survey 2021-2022 · Nationally validated",
  "subtitle":"Rwanda Ministry of Health · Rwanda WHO STEP Survey",
  "tab1":    "ABOUT YOU",
  "tab2":    "ADD MEASUREMENTS",
  "l1cap":   "Answer all questions below — no equipment needed.",
  "l2cap":   "Optional measurements improve your result accuracy.",
  "personal":"Personal Information",
  "medical": "Medical History",
  "lifestyle":"Lifestyle",
  "bodysize":"Body Size",
  "age":     "How old are you?",
  "sex":     "Sex",
  "male":    "Male","female":"Female",
  "htn":     "Have you been told you have high blood pressure (hypertension)?",
  "sugar_tested":"Have you ever had your blood sugar tested?",
  "told_high":   "Were you told your blood sugar was high?",
  "fam_diab":    "Do any close family members have diabetes?",
  "fam_who":     "Who in your family?",
  "fam_close":   "Parent or sibling",
  "fam_other":   "Other relative (grandparent/aunt/uncle/cousin)",
  "gest":        "Have you had diabetes during a previous pregnancy?",
  "smoking":     "Do you currently smoke tobacco?",
  "alcohol":     "How often do you drink alcohol?",
  "alc0":    "Never",
  "alc1":    "In the past 12 months",
  "alc2":    "In the past 30 days",
  "activity":"How physically active are you?",
  "active":  "Active",
  "inactive":"Not active enough",
  "fruit":   "How many days per week do you eat fruit?",
  "veg":     "How many days per week do you eat vegetables?",
  "days":    ["0 days","1 day","2 days","3 days","4 days","5 days","6 days","7 days"],
  "proc":    "How often do you eat processed or salty food?",
  "proc0":   "Never",
  "proc1":   "Rarely (1-3 days/month)",
  "proc2":   "Sometimes (2-3 days/week)",
  "proc3":   "Often (4-5 days/week)",
  "proc4":   "Every day",
  "know_wh": "Do you know your weight and height?",
  "yes":     "Yes","no":"No",
  "weight":  "Weight (kg)",
  "height":  "Height (cm)",
  "bodycat": "Select your body size",
  "bc0":     "Underweight (very thin)",
  "bc1":     "Normal weight",
  "bc2":     "Overweight (slightly heavy)",
  "bc3":     "Obese (very heavy)",
  "btn_check":"CHECK MY RISK",
  "btn_reset":"START AGAIN",
  "waist_check":"I have measured my waist",
  "waist":   "Waist circumference (cm)",
  "waist_guide":"Measure at navel level with a tape measure, while standing.",
  "waist_cat":"Select your waist size",
  "wc0":     "Slim (men <94cm / women <80cm)",
  "wc1":     "Average (men 94-102cm / women 80-88cm)",
  "wc2":     "Large (men >102cm / women >88cm)",
  "lab_check":"I have recent lab results",
  "chol":    "Total Cholesterol (mg/dl)",
  "hdl":     "HDL Cholesterol (mg/dl)",
  "btn_update":"UPDATE MY RESULT",
  "low":     "Low Risk",
  "mod":     "Moderate Risk",
  "high":    "High Risk",
  "onein":   "1 in {x} chance of prediabetes",
  "vs_avg":  "vs Rwanda average",
  "your_risk":"Your risk",
  "x_times": "{x}x Rwanda average",
  "contrib": "What is raising your risk?",
  "rec_title":"What should you do now?",
  "modifiable":"Can be changed",
  "non_mod": "Cannot be changed",
  "layer1":  "Basic estimate",
  "layer2":  "Improved estimate",
  "layer3":  "Most accurate",
  "disc":    "Rwanda Prediabetes Risk Predictor estimates risk based on Rwanda population data (WHO STEP Survey 2021-2022). This tool does not replace clinical diagnosis. Consult a healthcare provider for confirmation.",
  "footer":  "Rwanda Biomedical Centre · Ministry of Health Rwanda · Rwanda Prediabetes Risk Predictor · Version 30.0 · 2026",
  "validated":"Validated on Rwandan adults · 9 in 10 detection rate",
  "print":   "Print Result",
  "share":   "Share",
  "retake":  "Retake this assessment in 3 months to track your risk",
},
"rw": {
  "title":   "Igikoresho cyo Gusuzuma Ibyago bya Prediabetes",
  "meta":    "Iperereza rya Rwanda WHO STEP 2021-2022",
  "subtitle":"Ministeri y'Ubuzima Rwanda · Ikoreshwa na Machine Learning",
  "tab1":    "AMAKURU YAWE",
  "tab2":    "ONGERAHO IBIPIMO",
  "l1cap":   "Subiza ibibazo byose hasi — nta bikoresho bisabwa.",
  "l2cap":   "Ibipimo bya bijyanye na ho bitera ibisubizo byiza.",
  "personal":"Amakuru Yawe",
  "medical": "Amateka y'Indwara",
  "lifestyle":"Imyitwarire",
  "bodysize":"Ingano y'Umubiri",
  "age":     "Ufite imyaka ingahe?",
  "sex":     "Igitsina",
  "male":    "Gabo","female":"Gore",
  "htn":     "Muganga yakubwiye ko ufite umuvuduko w'amaraso mwinshi?",
  "sugar_tested":"Wigeze gupima inzara (blood sugar) kwa muganga?",
  "told_high":   "Barakubwiye ko inzara yawe yari hejuru?",
  "fam_diab":    "Mu miryango yawe haba umuntu ufite Sukari (diabetes)?",
  "fam_who":     "Ni nde mu muryango?",
  "fam_close":   "Umubyeyi cyangwa umuvandimwe",
  "fam_other":   "Incuti (sekuru/nyirakuru/nyarumba/musaza/mushiki)",
  "gest":        "Wigeze kuba ufite indwara ya Sukari mu gihe cy'inda?",
  "smoking":     "Utwika?",
  "alcohol":     "Unywa inzoga kangahe?",
  "alc0":    "Ntabwo nywa inzoga",
  "alc1":    "Nayinyweye mu mwaka ushize",
  "alc2":    "Nayinyweye mu kwezi gushize",
  "activity":"Imyitozo ngorora mubiri yagufitiye",
  "active":  "Ndakora imyitozo",
  "inactive":"Sinakora imyitozo",
  "fruit":   "Iminsi ingahe mu cyumweru urya imbuto?",
  "veg":     "Iminsi ingahe mu cyumweru urya imboga?",
  "days":    ["Iminsi 0","Umunsi 1","Iminsi 2","Iminsi 3",
              "Iminsi 4","Iminsi 5","Iminsi 6","Iminsi 7"],
  "proc":    "Kangahe urya ibiryo byungutse cyangwa birimo umunyu mwinshi?",
  "proc0":   "Ntanarimwe",
  "proc1":   "Gake (iminsi 1-3 mu kwezi)",
  "proc2":   "Rimwe na rimwe (iminsi 2-3 mu cyumweru)",
  "proc3":   "Akenshi (iminsi 4-5 mu cyumweru)",
  "proc4":   "Buri munsi",
  "know_wh": "Waba uzi uburemere n'uburebure bwawe?",
  "yes":     "Yego","no":"Oya",
  "weight":  "Uburemere (kg)",
  "height":  "Uburebure (cm)",
  "bodycat": "Hitamo ingano y'umubiri wawe",
  "bc0":     "Muto cyane (underweight)",
  "bc1":     "Bisanzwe (normal)",
  "bc2":     "Uburemere buri hejuru (overweight)",
  "bc3":     "Ubushyohe (obese)",
  "btn_check":"GENZURA IBYAGO BYANGE",
  "btn_reset":"TANGIRA BUSHYA",
  "waist_check":"Nasuzumye ingano y'ikibuno cyanjye",
  "waist":   "Ingano y'ikibuno (cm)",
  "waist_guide":"Sura ku rwego rw'inda yawe ukoresheje metero, uhagaze.",
  "waist_cat":"Hitamo ingano y'ikibuno cyawe",
  "wc0":     "Gato (abagabo <94cm / abagore <80cm)",
  "wc1":     "Bisanzwe (abagabo 94-102cm / abagore 80-88cm)",
  "wc2":     "Nini (abagabo >102cm / abagore >88cm)",
  "lab_check":"Mfite ibisubizo by'isuzuma rya laboratoire",
  "chol":    "Cholesterol yose (mg/dl)",
  "hdl":     "HDL Cholesterol (mg/dl)",
  "btn_update":"HINDURA IBISUBIZO BYANGE",
  "low":     "Ibyago Bike",
  "mod":     "Ibyago Biri Hagati",
  "high":    "Ibyago Bikabije",
  "onein":   "1 mu bantu {x} ifite ibyago by'indwara ya sukari",
  "vs_avg":  "vs abaRwanda muri rusange",
  "your_risk":"Ibyago byawe",
  "x_times": "{x}x Hagati ya Rwanda",
  "contrib": "Ni iki gituma hari ibyago nk'aya?",
  "rec_title":"Ugomba gukora iki ubu?",
  "modifiable":"Bishobora guhindurwa",
  "non_mod": "Ntibishobora guhindurwa",
  "layer1":  "Igipimo cy'ibanze",
  "layer2":  "Igipimo cyagezweho",
  "layer3":  "Igipimo nyacyo",
  "disc":    "Rwanda Prediabetes Risk Predictor ipima ibyago bya prediabetes ishingiye ku makuru y'abaRwanda (WHO STEP Survey 2021-2022). Iki gikoresho ntigisimbuza isuzuma rya muganga.",
  "footer":  "Rwanda Biomedical Centre · Ministeri y'Ubuzima Rwanda · Verisiyo 27.0 · 2026",
  "validated":"Yasuzumwe ku baRwanda · Yibutsa 9 mu bantu 10",
  "print":   "Sohora",
  "share":   "Sangira",
  "retake":  "Subiramo iki gipimo nyuma y'amezi 3 kugira ngo urebe impinduka",
},
}

# =============================================================================
# SESSION STATE
# =============================================================================
DEFS = {"lang":"en","layers":0,"assessed":False,
        "prob":0.0,"pd_data":{},"risk":"low","gauge_key":0}
for k,v in DEFS.items():
    if k not in st.session_state:
        st.session_state[k] = v

params = st.query_params
if "lang" in params:
    nl = params["lang"]
    if nl in ("en","rw") and nl != st.session_state["lang"]:
        st.session_state["lang"] = nl
        st.rerun()

L = st.session_state["lang"]
T = TX[L]

# =============================================================================
# LOAD MODELS
# =============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))

RWANDA_PREV  = 13.65   # national prediabetes prevalence %
DIABETES_PREV= 2.9     # national diabetes prevalence %

# =============================================================================
# THRESHOLDS — SINGLE CONSISTENT SYSTEM
# Model classification and gauge display zones are IDENTICAL
# Low  : below 0.20     (1.5x Rwanda prevalence 13.65% — zero false negatives)
# Mod  : 0.20 to 0.5875 (between prevalence boundary and ~95% sensitivity point)
# High : at or above 0.5875 (LR ~95% sensitivity, MaxF1-aligned threshold)
# Justification: Low boundary = 1.5x national prevalence (epidemiological)
#                High boundary = threshold achieving ~95% sensitivity on LR (statistical)
#                One system = model and display consistent = fully defensible
# =============================================================================
THRESHOLD_LOW  = 0.20
THRESHOLD_HIGH = 0.5875

@st.cache_resource
def load_models():
    pkg = joblib.load(os.path.join(BASE,"deployment_package.pkl"))
    lr  = pkg["lr_model"]
    return {
        "model"  : lr,
        "imputer": pkg["mice_imputer"],
        "scaler" : pkg["scaler"],
        "vars"   : pkg["selected_vars"],
        "medians": pkg["rwanda_medians"],
    }

try:
    M   = load_models()
    ok  = True
except Exception as e:
    ok  = False
    err = str(e)

# =============================================================================
# PREDICT
# =============================================================================
def predict(overrides={}):
    inp = M["medians"].copy()
    inp.update(overrides)
    if "age" in inp and inp["age"] is not None:
        inp["age"] = max(18.0, min(float(inp["age"]), 69.0))
    if "m14" in inp and inp["m14"] is not None:
        inp["m14"] = min(float(inp["m14"]), 110.0)
    if "bmi" in inp and inp["bmi"] is not None:
        inp["bmi"] = min(float(inp["bmi"]), 45.0)
    if "b8"  in inp and inp["b8"]  is not None:
        inp["b8"]  = min(float(inp["b8"]),  350.0)
    if "b17" in inp and inp["b17"] is not None:
        inp["b17"] = max(float(inp["b17"]), 10.0)
    X   = pd.DataFrame([inp])[M["vars"]]
    Xi  = pd.DataFrame(M["imputer"].transform(X), columns=M["vars"])
    Xs  = pd.DataFrame(M["scaler"].transform(Xi), columns=M["vars"])
    return float(M["model"].predict_proba(Xs.values)[0][1])

def get_risk_zone(p):
    # SINGLE SYSTEM — model classification = display zone
    # No separate display thresholds
    if p < THRESHOLD_LOW:   return "low"
    if p < THRESHOLD_HIGH:  return "mod"
    return "high"

def get_colors(zone):
    return {
        "low" : ("#1A7A3C","#EAF7EF"),
        "mod" : ("#B7770D","#FEF9E7"),
        "high": ("#922B21","#FDEDEC"),
    }[zone]

# =============================================================================
# ANIMATED GAUGE — THREE ZONES MATCHING THRESHOLDS EXACTLY
# Green : 0 to 20%    (Low Risk)
# Yellow: 20 to 58.75% (Moderate Risk)
# Red   : 58.75 to 100% (High Risk)
# Arc coordinates calculated from threshold percentages:
#   0%     -> (30.0, 155.0)
#   20%    -> (54.8, 78.6)
#   58.75% -> (195.3, 29.9)
#   100%   -> (290.0, 155.0)
# =============================================================================
def make_gauge(prob, zone, key_suffix=""):
    import streamlit.components.v1 as components
    pct    = min(round(prob*100, 1), 90.0)
    color,_ = get_colors(zone)
    zone_colors = {
        "low" : "#1A7A3C",
        "mod" : "#B7770D",
        "high": "#922B21",
    }
    nc    = zone_colors.get(zone, color)
    angle = round(pct / 100 * 180, 1)

    html = f"""<!DOCTYPE html>
<html><head>
<style>
  body {{margin:0;padding:0;background:transparent;overflow:hidden;
        display:flex;flex-direction:column;align-items:center}}
  #needle {{
    transform-origin:160px 155px;
    transform:rotate(0deg);
    transition:transform 1.5s cubic-bezier(0.34,1.2,0.64,1);
  }}
  .pct {{
    font-family:Georgia,serif;font-size:46px;font-weight:700;
    color:{nc};text-align:center;margin-top:-8px;
    line-height:1;letter-spacing:-1px
  }}
  .risk-lbl {{
    font-size:12px;font-weight:700;letter-spacing:1.5px;
    text-transform:uppercase;color:{nc};text-align:center;
    margin-top:2px
  }}
</style>
</head><body>
<svg viewBox="0 0 320 175" width="320" height="175"
     xmlns="http://www.w3.org/2000/svg">

  <!-- Background track -->
  <path d="M 30.0 155.0 A 130 130 0 0 1 290.0 155.0"
        fill="none" stroke="#EEF1F6" stroke-width="28" stroke-linecap="round"/>

  <!-- GREEN zone: Low Risk 0-20% -->
  <path d="M 30.0 155.0 A 130 130 0 0 1 54.8 78.6"
        fill="none" stroke="#D6EFE0" stroke-width="26" stroke-linecap="butt"/>

  <!-- YELLOW zone: Moderate Risk 20-58.75% -->
  <path d="M 54.8 78.6 A 130 130 0 0 1 195.3 29.9"
        fill="none" stroke="#FDF6DC" stroke-width="26" stroke-linecap="butt"/>

  <!-- RED zone: High Risk 58.75-100% -->
  <path d="M 195.3 29.9 A 130 130 0 0 1 290.0 155.0"
        fill="none" stroke="#FADBD8" stroke-width="26" stroke-linecap="butt"/>

  <!-- Zone boundaries communicated by color transitions only -->

  <!-- Tick marks every 5% -->
  <line x1="22.0" y1="155.0" x2="38.0" y2="155.0" stroke="#8A9BAB" stroke-width="2.0"/>
  <line x1="23.7" y1="133.4" x2="39.5" y2="135.9" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="28.8" y1="112.4" x2="44.0" y2="117.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="37.0" y1="92.3" x2="51.3" y2="99.6" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="48.4" y1="73.9" x2="61.3" y2="83.3" stroke="#8A9BAB" stroke-width="2.0"/>
  <line x1="62.4" y1="57.4" x2="73.7" y2="68.7" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="78.9" y1="43.4" x2="88.3" y2="56.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="97.3" y1="32.0" x2="104.6" y2="46.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="117.4" y1="23.8" x2="122.3" y2="39.0" stroke="#8A9BAB" stroke-width="2.0"/>
  <line x1="138.4" y1="18.7" x2="140.9" y2="34.5" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="160.0" y1="17.0" x2="160.0" y2="33.0" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="181.6" y1="18.7" x2="179.1" y2="34.5" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="202.6" y1="23.8" x2="197.7" y2="39.0" stroke="#8A9BAB" stroke-width="2.0"/>
  <line x1="222.7" y1="32.0" x2="215.4" y2="46.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="241.1" y1="43.4" x2="231.7" y2="56.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="257.6" y1="57.4" x2="246.3" y2="68.7" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="271.6" y1="73.9" x2="258.7" y2="83.3" stroke="#8A9BAB" stroke-width="2.0"/>
  <line x1="283.0" y1="92.3" x2="268.7" y2="99.6" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="291.2" y1="112.4" x2="276.0" y2="117.3" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="296.3" y1="133.4" x2="280.5" y2="135.9" stroke="#8A9BAB" stroke-width="1.0"/>
  <line x1="298.0" y1="155.0" x2="282.0" y2="155.0" stroke="#8A9BAB" stroke-width="2.0"/>

  <!-- Zone labels removed — arc colors and needle communicate zones clearly -->

  <!-- Labels at clinically meaningful positions: 0, 20, 60, 100 -->
  <text x="12.0" y="162.0" font-size="10" fill="#8A9BAB" font-family="sans-serif" font-weight="600" text-anchor="middle">0</text>
  <text x="36.0" y="68.0" font-size="10" fill="#1A7A3C" font-family="sans-serif" font-weight="700" text-anchor="middle">20</text>
  <text x="207.0" y="22.0" font-size="10" fill="#922B21" font-family="sans-serif" font-weight="700" text-anchor="middle">60</text>
  <text x="308.0" y="162.0" font-size="10" fill="#8A9BAB" font-family="sans-serif" font-weight="600" text-anchor="middle">100</text>

  <!-- Needle -->
  <g id="needle">
    <line x1="160" y1="155" x2="48.0" y2="155.0"
          stroke="{nc}" stroke-width="3.5" stroke-linecap="round"/>
    <polygon points="160,150 44.0,155.0 160,160"
             fill="{nc}" opacity="0.3"/>
  </g>

  <!-- Centre hub -->
  <circle cx="160" cy="155" r="11" fill="{nc}"/>
  <circle cx="160" cy="155" r="6"  fill="white"/>

</svg>

<div class="pct">{pct}%</div>
<div class="risk-lbl">{T[zone]}</div>

<script>
window.addEventListener('load', function() {{
  setTimeout(function() {{
    document.getElementById('needle').style.transform =
      'rotate({angle}deg)';
  }}, 150);
}});
</script>
</body></html>"""
    components.html(html, height=280)

# =============================================================================
# STATIC GAUGE FOR PRINT — THREE ZONES
# =============================================================================
def make_static_gauge_svg(pct_val, zone):
    import math as _math
    _cx, _cy, _r = 160, 155, 120
    _colors = {"low":"#1A7A3C","mod":"#B7770D","high":"#922B21"}
    _nc = _colors.get(zone, "#922B21")
    _deg = 180 - (pct_val / 100 * 180)
    _rad = _math.radians(_deg)
    _ntx = round(_cx + _r * _math.cos(_rad), 1)
    _nty = round(_cy - _r * _math.sin(_rad), 1)
    return (
        f'<svg viewBox="0 0 320 175" width="260" height="145" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M 40 155 A 120 120 0 0 1 280 155" fill="none" stroke="#EEF1F6" stroke-width="26" stroke-linecap="round"/>'
        '<path d="M 40 155 A 120 120 0 0 1 62.9 84.5" fill="none" stroke="#D6EFE0" stroke-width="24" stroke-linecap="butt"/>'
        '<path d="M 62.9 84.5 A 120 120 0 0 1 192.6 39.5" fill="none" stroke="#FDF6DC" stroke-width="24" stroke-linecap="butt"/>'
        '<path d="M 192.6 39.5 A 120 120 0 0 1 280 155" fill="none" stroke="#FADBD8" stroke-width="24" stroke-linecap="butt"/>'
        f'<line x1="{_cx}" y1="{_cy}" x2="{_ntx}" y2="{_nty}" stroke="{_nc}" stroke-width="3.5" stroke-linecap="round"/>'
        f'<circle cx="{_cx}" cy="{_cy}" r="10" fill="{_nc}"/>'
        f'<circle cx="{_cx}" cy="{_cy}" r="5" fill="white"/>'
        # Zone labels removed
        '</svg>'
    )

# =============================================================================
# PRINT PAGE GENERATOR
# =============================================================================
def generate_print_page(prob, zone, pct_val, v, lang, trans, alerts_list, recs_list):
    import datetime
    today    = datetime.date.today().strftime("%d %B %Y")
    zc = {"low":"#1A7A3C","mod":"#B7770D","high":"#922B21"}
    zb = {"low":"#EAF7EF","mod":"#FEF9E7","high":"#FDEDEC"}
    nc  = zc.get(zone,"#922B21")
    bg  = zb.get(zone,"#FDEDEC")
    lbl = trans[zone]
    if prob > 0:
        oi_raw = 1/prob
        oi = round(oi_raw,1) if oi_raw<2 else round(oi_raw)
    else:
        oi = 999
    msg_map = {
        "low" :"Annual blood sugar check recommended.",
        "mod" :"Schedule blood sugar test within 6 months.",
        "high":"Seek blood sugar test urgently within 2 weeks.",
    }
    msg_rw = {
        "low" :"Gupima inzara buri mwaka bisabwa.",
        "mod" :"Gupima inzara mu mezi 6.",
        "high":"Gupima inzara mu cyumweru 2.",
    }
    urgency = msg_map.get(zone,"") if lang=="en" else msg_rw.get(zone,"")
    gauge_svg = make_static_gauge_svg(pct_val, zone)
    alert_rows = ""
    for name,val,desc,lvl,mod in alerts_list[:8]:
        col = {"alert":"#922B21","warn":"#B7770D"}.get(lvl,"#1A5276")
        alert_rows += f"<tr><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:11px;font-weight:700;color:{col};width:30%'>{name}</td><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:11px;color:#3D5166;width:20%'>{val}</td><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:10px;color:#7A8FA6'>{desc}</td></tr>"
    rec_rows = ""
    for i,(title,desc,c) in enumerate(recs_list):
        rec_rows += f"<tr><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:12px;font-weight:bold;color:{c};width:5%'>{i+1}</td><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:11px;font-weight:700;color:#0B1E30;width:35%'>{title}</td><td style='padding:5px 8px;border-bottom:1px solid #eee;font-size:11px;color:#3D5166'>{desc}</td></tr>"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prediabetes Risk Report</title>
<style>
body{{margin:0;padding:0;font-family:Arial,sans-serif;background:white;color:#0B1E30}}
.hdr{{background:#00A1DE;padding:14px 24px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.hdr-t{{font-size:16px;color:white;font-weight:bold;margin:0}}
.hdr-s{{font-size:10px;color:rgba(255,255,255,0.8);margin-top:3px}}
.bar-y{{background:#FAD201;height:5px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.bar-g{{background:#20603D;height:5px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.body{{padding:16px 24px}}
.row{{display:flex;gap:16px;align-items:flex-start;margin-bottom:14px}}
.risk-box{{background:{bg};border-left:4px solid {nc};border-radius:6px;padding:12px 16px;flex:1;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.risk-pct{{font-size:38px;font-weight:bold;color:{nc};line-height:1}}
.risk-lbl{{font-size:12px;font-weight:bold;color:{nc};text-transform:uppercase;letter-spacing:1px}}
.risk-oi{{font-size:12px;color:#3D5166;margin-top:3px}}
.risk-msg{{font-size:11px;color:#3D5166;margin-top:6px;padding-top:6px;border-top:1px solid rgba(0,0,0,0.1)}}
.sec{{font-size:9px;font-weight:bold;letter-spacing:2px;text-transform:uppercase;color:#1F618D;margin:12px 0 6px;padding-bottom:3px;border-bottom:1px solid #D5DCE8}}
table{{width:100%;border-collapse:collapse}}
.disc{{font-size:9px;color:#7A8FA6;text-align:center;padding:10px;border-top:1px solid #D5DCE8;margin-top:12px;line-height:1.5}}
@page{{margin:1cm;size:A4 portrait}}
@media print{{body{{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}}}}
</style></head><body>
<div class="hdr">
  <div class="hdr-t">Rwanda Prediabetes Risk Predictor — Report</div>
  <div class="hdr-s">Rwanda WHO STEP Survey 2021-2022 · {today}</div>
</div>
<div class="bar-y"></div>
<div class="bar-g"></div>
<div class="body">
  <div class="row">
    <div>{gauge_svg}</div>
    <div class="risk-box">
      <div class="risk-lbl">{lbl}</div>
      <div class="risk-pct">{pct_val}%</div>
      <div class="risk-oi">1 in {oi} chance of prediabetes</div>
      <div class="risk-msg">{urgency}</div>
    </div>
  </div>
  <div class="sec">What is raising your risk</div>
  <table>{alert_rows}</table>
  <div class="sec">What you should do</div>
  <table>{rec_rows}</table>
  <div class="disc">{trans["disc"]}<br><small>{trans["footer"]}</small></div>
</div>
<script>window.onload=function(){{setTimeout(function(){{window.print();}},500);}}</script>
</body></html>"""

# =============================================================================
# CLINICAL ALERTS — UNCHANGED FROM v26
# =============================================================================
def get_alerts(v, L):
    alerts = []
    age    = v.get("age",39)
    bmi    = v.get("bmi",None)
    waist  = v.get("m14",None)
    hypert = v.get("hypert",0)
    smk    = v.get("smoking",0)
    alc    = v.get("alcohol_status",0)
    pi     = v.get("pi",0)
    d1     = v.get("d1",1)
    d3     = v.get("d3",4)
    d7     = v.get("d7",5)
    b8     = v.get("b8",None)
    b17    = v.get("b17",None)
    sex    = v.get("sex_val",2)
    prov   = v.get("province",4)
    fam    = v.get("fam_diab",False)
    gest   = v.get("gest_diab",False)
    sugar_high = v.get("told_high",False)

    if L=="en":
        if sugar_high:
            alerts.append(("Previously told high blood sugar","Personal history",
                "You have previously been told your blood sugar was high. This is the strongest individual risk factor. Seek testing immediately.","alert",False))
        if fam:
            alerts.append(("Family history of diabetes","Close relative affected",
                "Having a parent or sibling with diabetes increases your risk by 2-6 times. Regular screening is essential.","alert",False))
        if gest:
            alerts.append(("History of gestational diabetes","Previous pregnancy",
                "Women who had diabetes during pregnancy have up to 10 times higher risk of developing prediabetes.","alert",False))
        if hypert==1:
            alerts.append(("Hypertension diagnosis","Confirmed",
                "Hypertension and prediabetes frequently co-occur. Managing blood pressure also protects blood sugar.","alert",False))
        if age>=55:
            alerts.append(("Age is a significant risk factor",f"{age} years old",
                "Risk of prediabetes increases substantially after age 55.","alert",False))
        elif age>=45:
            alerts.append(("Age increases your risk",f"{age} years old",
                "Prediabetes risk rises meaningfully after age 45. Annual screening is recommended.","warn",False))
        if bmi and bmi>=30:
            alerts.append(("BMI indicates obesity",f"BMI {bmi:.1f} kg/m2",
                "Obesity is the leading modifiable risk factor for prediabetes. Losing 5-7% of body weight reduces risk by up to 58%.","alert",True))
        elif bmi and bmi>=25:
            alerts.append(("BMI is above normal",f"BMI {bmi:.1f} kg/m2",
                "Overweight individuals have 2-3 times higher prediabetes risk. Weight reduction helps.","warn",True))
        if waist:
            cutoff = 94 if sex==1 else 80
            if waist>cutoff:
                alerts.append(("Waist above safe limit",f"{waist:.0f} cm (limit {cutoff} cm)",
                    "Excess abdominal fat is strongly linked to insulin resistance, a key driver of prediabetes.","alert",True))
        if b8 and b17 and b17>0:
            ratio = b8/b17
            if ratio>5:
                alerts.append(("High cholesterol ratio",f"Ratio {ratio:.1f} (safe <5)",
                    "A high total/HDL cholesterol ratio indicates metabolic risk associated with prediabetes.","warn",True))
        if b17 and b17<40:
            alerts.append(("Low HDL cholesterol",f"HDL {b17:.0f} mg/dl",
                "Low HDL is a component of metabolic syndrome and increases prediabetes risk.","warn",True))
        if smk==1:
            alerts.append(("Currently smoking","Active smoker",
                "Smoking impairs insulin sensitivity. Quitting reduces prediabetes risk significantly.","warn",True))
        if alc==2:
            alerts.append(("Recent alcohol consumption","Past 30 days",
                "Regular alcohol use can contribute to insulin resistance.","warn",True))
        if pi==1:
            alerts.append(("Low physical activity","Below WHO recommendation",
                "Physical inactivity is a major risk factor. 150 min/week of moderate activity reduces risk by 30%.","warn",True))
        if d1==0:
            alerts.append(("Very low fruit intake","Less than 1 day/week",
                "Low fruit intake is associated with higher metabolic risk. Aim for fruit daily.","warn",True))
        if d7<=2:
            alerts.append(("High processed food consumption","Often or every day",
                "Ultra-processed foods drive weight gain and insulin resistance.","warn",True))
        if prov==2:
            alerts.append(("Kigali City: highest risk province","Prevalence 26.02%",
                "Kigali has more than double the national prediabetes rate (26% vs 13.65%) due to rapid urbanisation and dietary change.","warn",False))
    else:
        if sugar_high:
            alerts.append(("Barakubwiye ko inzara yawe yari hejuru","Amateka bwite",
                "Warigeze kubwirwa ko inzara yawe yari hejuru. Ibi ni impamvu ikomeye cyane. Genda gusuzumwa vuba.","alert",False))
        if fam:
            alerts.append(("Umuryango ufite Sukari","Incuti ifatwa n'indwara",
                "Kugira umubyeyi cyangwa umuvandimwe ufite Sukari biyongera ibyago by'inshuro 2-6.","alert",False))
        if gest:
            alerts.append(("Wigeze kuba ufite Sukari mu gihe cy'inda","Inda ishize",
                "Abagore bafite Sukari mu gihe cy'inda bafite ibyago bigera inshuro 10 kurenza abandi.","alert",False))
        if hypert==1:
            alerts.append(("Umuvuduko mwinshi w'amaraso","Yemejwe",
                "Umuvuduko n'indwara ya sukari bikunze kujyana. Genzura umuvuduko birarinda inzara.","alert",False))
        if age>=55:
            alerts.append(("Imyaka yawe ni impamvu ikomeye",f"Imyaka {age}",
                "Ibyago bya prediabetes biyongera cyane nyuma y'imyaka 55.","alert",False))
        elif age>=45:
            alerts.append(("Imyaka yongera ibyago",f"Imyaka {age}",
                "Ibyago bya prediabetes biyongera nyuma y'imyaka 45.","warn",False))
        if bmi and bmi>=30:
            alerts.append(("BMI yerekana ubushyohe",f"BMI {bmi:.1f} kg/m2",
                "Ubushyohe ni impamvu nkuru yo guhindurwa. Kugabanya ibiro 5-7% bigabanya ibyago 58%.","alert",True))
        elif bmi and bmi>=25:
            alerts.append(("BMI irenze bisanzwe",f"BMI {bmi:.1f} kg/m2",
                "Uburemere burenga bwongera ibyago by'inshuro 2-3. Kugabanya ibiro birafasha.","warn",True))
        if waist:
            cutoff = 94 if sex==1 else 80
            if waist>cutoff:
                alerts.append(("Ikibuno kirenga umurego",f"{waist:.0f} cm (umurego {cutoff} cm)",
                    "Iburemere bw'inda buhuriye n'indwara ya sukari.","alert",True))
        if smk==1:
            alerts.append(("Utwika","Utwika ubu",
                "Kutwika bigabanya ubushobozi bw'insulin. Guhagarika kutwika biragabanya ibyago.","warn",True))
        if pi==1:
            alerts.append(("Imyitozo ntihagije","Munsi y'ibisabwa",
                "Kutakora imyitozo ni impamvu nkuru. Iminota 150 mu cyumweru igabanya ibyago 30%.","warn",True))
        if prov==2:
            alerts.append(("Umujyi wa Kigali: intara y'ibyago bikabije","Prevalence 26.02%",
                "Kigali ifite prediabetes inshuro 2 surpassing igihugu (26% vs 13.65%).","warn",False))
    return alerts

# =============================================================================
# RECOMMENDATIONS — THREE ZONES, vhi MERGED INTO high
# =============================================================================
def get_recs(zone, alerts, L):
    alert_tags = [a[0].lower() for a in alerts]
    has = lambda *kws: any(k in t for k in kws for t in alert_tags)

    if L=="en":
        base = {
            "low": [
                ("Annual blood pressure and blood sugar check",
                 "Visit your nearest health facility once a year.",
                 "#1A7A3C"),
                ("Maintain your healthy lifestyle",
                 "Continue regular physical activity and a balanced diet.",
                 "#1A7A3C"),
            ],
            "mod": [
                ("Blood sugar test within 6 months",
                 "Visit a health centre for fasting blood glucose or HbA1c test.",
                 "#B7770D"),
                ("Increase physical activity",
                 "Aim for 150 minutes per week of moderate exercise.",
                 "#B7770D"),
                ("Reduce processed and sugary food",
                 "Limit chips, sodas, white rice, and packaged food.",
                 "#B7770D"),
            ],
            "high": [
                ("Blood sugar test within 2 weeks",
                 "Seek fasting blood glucose test urgently at a health centre.",
                 "#922B21"),
                ("Achieve healthy weight",
                 "Losing 5-7% of body weight can reduce prediabetes risk by up to 58%.",
                 "#922B21"),
                ("150+ min/week physical activity",
                 "Brisk walking, cycling or equivalent on most days.",
                 "#922B21"),
                ("Dietary changes today",
                 "Reduce sugar, white carbs, and processed food. Increase vegetables.",
                 "#922B21"),
                ("Discuss with your doctor",
                 "At high risk your doctor may recommend structured lifestyle intervention or medication.",
                 "#922B21"),
            ],
        }
        extras = []
        if has("smoking","twika"):
            extras.append(("Quit smoking",
                "Quitting smoking improves insulin sensitivity within weeks.",
                "#B7770D" if zone=="mod" else "#922B21"))
        if has("alcohol","inzoga"):
            extras.append(("Reduce alcohol consumption",
                "Limit alcohol to reduce insulin resistance.",
                "#B7770D" if zone=="mod" else "#922B21"))
        if has("processed","ungutse"):
            extras.append(("Limit processed and salty food",
                "Replace processed food with whole foods, vegetables and fruits.",
                "#B7770D" if zone=="mod" else "#922B21"))
        if has("family","muryango"):
            extras.append(("Regular screening: family history",
                "With family history of diabetes, screen annually even at low risk.",
                "#1F618D"))
        if has("kigali","province"):
            extras.append(("Urban dietary awareness",
                "Kigali residents face higher risk due to dietary changes. Prioritise traditional foods.",
                "#1F618D"))
    else:
        base = {
            "low": [
                ("Gupima inzara n'umuvuduko buri mwaka",
                 "Genda ku ivuriro ryegereye buri mwaka.",
                 "#1A7A3C"),
                ("Komeza imyitwarire myiza",
                 "Komeza imyitozo n'indyo iringaniye.",
                 "#1A7A3C"),
            ],
            "mod": [
                ("Gupima inzara mu mezi 6",
                 "Genda ku ivuriro gupimuza inzara y'amaraso.",
                 "#B7770D"),
                ("Ongera imyitozo",
                 "Kora iminota 150 mu cyumweru.",
                 "#B7770D"),
                ("Gabanya ibiryo byungutse",
                 "Irinde chips, amafanta, n'ibiryo bifungirwa.",
                 "#B7770D"),
            ],
            "high": [
                ("Gupima inzara mu cyumweru 2",
                 "Genda ku ivuriro gupimuza inzara y'amaraso vuba.",
                 "#922B21"),
                ("Gabanya ibiro",
                 "Kugabanya ibiro 5-7% bigabanya ibyago 58%.",
                 "#922B21"),
                ("Imyitozo 150+ min/cyumweru",
                 "Kugenda vuba, gukora ku mabara ku minsi igera 5.",
                 "#922B21"),
                ("Hindura indyo uyu munsi",
                 "Gabanya sukari, ibiryo byera, n'ibiryo byungutse.",
                 "#922B21"),
                ("Bwira muganga ibyago byawe",
                 "Muganga ashobora guha imiti igabanya ibyago by'indwara ya sukari.",
                 "#922B21"),
            ],
        }
        extras = []
        if has("smoking","twika"):
            extras.append(("Hagarika kutwika",
                "Guhagarika kutwika bitezimbere ubushobozi bw'insulin.",
                "#B7770D"))
        if has("processed","ungutse"):
            extras.append(("Gabanya ibiryo byungutse",
                "Simbura ibiryo byungutse n'imboga n'imbuto.",
                "#B7770D"))
        if has("family","muryango"):
            extras.append(("Suzuma kenshi: umuryango ufite Sukari",
                "Ugomba gusuzumwa buri mwaka nubwo ibyago bike.",
                "#1F618D"))

    recs = base.get(zone, base["low"]) + extras
    return recs

# =============================================================================
# SHOW RESULTS
# =============================================================================
def show_results(prob, zone, layers, v):
    # Universal 90% display cap — no medical probability shown as 100%
    if prob > 0.90:
        prob = 0.90
        zone = get_risk_zone(prob)

    pct   = min(round(prob*100, 1), 90.0)
    color, bg = get_colors(zone)
    lbl   = T[zone]
    cls   = {"low":"risk-low","mod":"risk-mod","high":"risk-high"}[zone]

    make_gauge(prob, zone, key_suffix=str(layers))

    lnames = [T["layer1"],T["layer2"],T["layer3"]]
    dots = "".join([f'<div class="qdot {"on" if i<layers else ""}"></div>'
                    for i in range(3)])
    st.markdown(
        f'<div class="qlbl">{lnames[min(layers-1,2)]}</div>' +
        f'<div class="qdots">{dots}</div>',
        unsafe_allow_html=True)

    if prob > 0:
        onein_raw = 1 / prob
        onein = round(onein_raw,1) if onein_raw<2 else round(onein_raw)
    else:
        onein = 999
    onein = min(onein, 999)

    msg_en = {"low":"Annual blood sugar check recommended. Maintain healthy lifestyle.",
              "mod":"Schedule blood sugar test within 6 months.",
              "high":"Seek blood sugar test urgently within 2 weeks."}
    msg_rw = {"low":"Gupima inzara buri mwaka bisabwa. Komeza imyitwarire myiza.",
              "mod":"Teganya gupima inzara mu mezi 6 iri imbere.",
              "high":"Genda vuba gupimuza inzara mu cyumweru 2."}
    msg = msg_en[zone] if L=="en" else msg_rw[zone]

    st.markdown(f"""
<div class="risk-banner {cls}">
  <div class="risk-label" style="color:{color}">{lbl}</div>
  <div class="risk-pct" style="color:{color}">{pct}%</div>
  <div class="risk-onein" style="color:{color}">
    {T["onein"].replace("{x}", str(onein))}</div>
  <div class="rbar-bg">
    <div class="rbar" style="width:{pct}%;background:{color}"></div>
  </div>
  <div class="risk-msg">{msg}</div>
</div>""", unsafe_allow_html=True)

    ratio = round(pct/RWANDA_PREV,1) if RWANDA_PREV>0 else 1
    your_w  = min(pct, 100)
    avg_w   = RWANDA_PREV
    diab_w  = DIABETES_PREV
    st.markdown(f'<div class="slbl">{T["your_risk"]} vs {T["vs_avg"]}</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="card cbar-wrap">
  <div class="cbar-row">
    <div class="cbar-lbl">{T["your_risk"]}</div>
    <div class="cbar-bg">
      <div class="cbar-fill" style="width:{your_w}%;background:{color}"></div>
    </div>
    <div class="cbar-pct" style="color:{color}">{pct}%</div>
  </div>
  <div class="cbar-row">
    <div class="cbar-lbl">Rwanda prediabetes</div>
    <div class="cbar-bg">
      <div class="cbar-fill" style="width:{avg_w}%;background:#1F618D"></div>
    </div>
    <div class="cbar-pct" style="color:#1F618D">{RWANDA_PREV}%</div>
  </div>
  <div class="cbar-row">
    <div class="cbar-lbl">Rwanda diabetes</div>
    <div class="cbar-bg">
      <div class="cbar-fill" style="width:{diab_w*3}%;background:#7D3C98"></div>
    </div>
    <div class="cbar-pct" style="color:#7D3C98">{DIABETES_PREV}%</div>
  </div>
  <div style="font-size:12px;color:{color};font-weight:700;margin-top:6px">
    {T["x_times"].replace("{x}", str(ratio))}
  </div>
</div>""", unsafe_allow_html=True)

    alerts = get_alerts(v, L)
    if alerts:
        st.markdown(f'<div class="slbl">{T["contrib"]}</div>',
                    unsafe_allow_html=True)
        col_map = {"alert":"#922B21","warn":"#B7770D","ok":"#1A7A3C","info":"#1A5276"}
        badge_map = {
            "alert": ("abadge-alert","HIGH" if L=="en" else "HEJURU"),
            "warn":  ("abadge-warn","MODERATE" if L=="en" else "HAGATI"),
            "ok":    ("abadge-ok","NORMAL"),
            "info":  ("abadge-info","INFO"),
        }
        rows = ""
        for name,val,desc,lvl,modifiable in alerts:
            c    = col_map[lvl]
            bc,bt= badge_map[lvl]
            mod_txt = ('<span style="font-size:10px;color:#1A7A3C;font-weight:600;margin-left:6px">' + T["modifiable"] + '</span>'
                       if modifiable else
                       '<span style="font-size:10px;color:#7A8FA6;font-weight:600;margin-left:6px">' + T["non_mod"] + '</span>')
            rows += f"""
<div class="alert-item">
  <div class="alert-dot" style="background:{c}"></div>
  <div>
    <div class="alert-name">{name}
      <span class="abadge {bc}">{bt}</span>{mod_txt}
    </div>
    <div class="alert-val"><b>{val}</b> — {desc}</div>
  </div>
</div>"""
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    recs = get_recs(zone, alerts, L)
    st.markdown(f'<div class="slbl">{T["rec_title"]}</div>',
                unsafe_allow_html=True)
    rows = ""
    for i,(title,desc,c) in enumerate(recs):
        rows += f"""
<div class="rec-item">
  <div style="display:flex;align-items:flex-start;gap:10px">
    <div class="rec-num" style="background:{c}">{i+1}</div>
    <div>
      <div class="rec-title">{title}</div>
      <div class="rec-desc">{desc}</div>
    </div>
  </div>
</div>"""
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:#EBF5FB;border-left:3px solid #2980B9;'
        'border-radius:4px;padding:8px 14px;margin:8px 0;font-size:12px;color:#1A5276">'
        f'{T["validated"]}</div>',
        unsafe_allow_html=True)

    st.caption(T['retake'])

    c1,c2 = st.columns(2)
    with c1:
        if st.button(T["print"],
                     key=f"print_{layers}_{st.session_state['gauge_key']}",
                     use_container_width=True):
            import streamlit.components.v1 as _comp
            _alerts_p = get_alerts(v, L)
            _recs_p   = get_recs(zone, _alerts_p, L)
            _html_p   = generate_print_page(prob, zone, pct, v, L, T, _alerts_p, _recs_p)
            _comp.html(_html_p, height=800, scrolling=True)
            st.caption("PDF preview below. Click Ctrl+P to save as PDF." if L=="en"
                       else "Reba raporo hasi. Koresha Ctrl+P kubika PDF.")
    with c2:
        if st.button(T["share"],
                     key=f"share_{layers}_{st.session_state['gauge_key']}",
                     use_container_width=True):
            import datetime, urllib.parse
            risk_lbl = T[zone]
            today    = datetime.date.today().strftime("%d %b %Y")
            share_alerts = get_alerts(v, L)
            top4 = [a[0] for a in share_alerts[:4]]
            top4_str = ", ".join(top4) if top4 else ""
            action_en = {"low":"Annual blood sugar check.","mod":"Blood sugar test within 6 months.","high":"Blood sugar test within 2 weeks."}
            action_rw = {"low":"Gupima inzara buri mwaka.","mod":"Gupima inzara mu mezi 6.","high":"Gupima inzara mu cyumweru 2."}
            if L=="en":
                share_text = f"My prediabetes risk: {pct}% ({risk_lbl}). {today}.\n" + (f"Risk contributors: {top4_str}.\n" if top4_str else "") + f"{action_en.get(zone,'')}\n"
                email_sub  = "My Prediabetes Risk Result"
            else:
                share_text = f"Ibyago byanjye: {pct}% ({risk_lbl}). {today}.\n" + (f"Impamvu: {top4_str}.\n" if top4_str else "") + f"{action_rw.get(zone,'')}\n"
                email_sub  = "Ibisubizo byanjye bya Prediabetes"
            wa_text    = urllib.parse.quote(share_text)
            email_bod  = urllib.parse.quote(share_text)
            email_subj = urllib.parse.quote(email_sub)
            email_link = f"mailto:?subject={email_subj}&body={email_bod}"
            _msg_safe  = share_text.replace("'","").replace('"','').replace('\n',' ')
            st.markdown(
                '<div style="background:#F4F6FA;border-left:3px solid #1A5276;'
                'border-radius:4px;padding:10px 14px;margin:8px 0;'
                'font-size:12px;line-height:1.8;color:#0B1E30">'
                + share_text.replace("\n","<br>") + '</div>',
                unsafe_allow_html=True)
            import streamlit.components.v1 as _sc
            _lbl_phone = "Phone number e.g. +250788123456" if L=="en" else "Numero ya telefoni ex: +250788123456"
            _lbl_hint  = "Enter phone number to send via WhatsApp or SMS." if L=="en" else "Injiza numero ya telefoni kohereza."
            _sc.html(f"""
<div style="font-family:Arial,sans-serif;padding:4px 0">
  <div style="margin-bottom:10px">
    <input id="ph" type="tel" placeholder="{_lbl_phone}"
           style="width:100%;padding:8px 12px;border:1.5px solid #D5DCE8;
                  border-radius:6px;font-size:12px;color:#0B1E30;box-sizing:border-box"/>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
    <button onclick="(function(){{var p=document.getElementById('ph').value.replace(/[^0-9+]/g,'');var m=encodeURIComponent('{_msg_safe}');window.open(p?'https://wa.me/'+p+'?text='+m:'https://wa.me/?text='+m,'_blank');}})()"
       style="background:#25D366;color:white;padding:10px 20px;border-radius:6px;font-size:12px;font-weight:700;border:none;cursor:pointer">WhatsApp</button>
    <button onclick="(function(){{var p=document.getElementById('ph').value.replace(/[^0-9+]/g,'');var m=encodeURIComponent('{_msg_safe}');var a=document.createElement('a');a.href=p?'sms:'+p+'?body='+m:'sms:?body='+m;a.click();}})()"
       style="background:#5B5EA6;color:white;padding:10px 20px;border-radius:6px;font-size:12px;font-weight:700;border:none;cursor:pointer">SMS</button>
    <button onclick="(function(){{var a=document.createElement('a');a.href='{email_link}';a.click();}})()"
       style="background:#1A5276;color:white;padding:10px 20px;border-radius:6px;font-size:12px;font-weight:700;border:none;cursor:pointer">Email</button>
  </div>
  <div style="font-size:10px;color:#7A8FA6">{_lbl_hint}</div>
</div>""", height=140)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(f"""
<div class="hdr-blue">
  <div class="hdr-title">{T["title"]}</div>
</div>
<div class="hdr-yellow"></div>
<div class="hdr-green">
  <div class="lang-row">
    <a class="lbtn {"on" if L=="en" else ""}" href="?lang=en">English</a>
    <a class="lbtn {"on" if L=="rw" else ""}" href="?lang=rw">Kinyarwanda</a>
  </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"Model files not found: {BASE}")
    st.code(err); st.stop()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs([T["tab1"], T["tab2"]])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ABOUT YOU
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="slbl">{T["personal"]}</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        age = st.number_input(T["age"], 1, 120, 39, key="k_age")
    with c2:
        sex_opt = st.radio(T["sex"], [T["female"],T["male"]], horizontal=True, key="k_sex")
    sex_v = 2 if sex_opt==T["female"] else 1

    st.markdown(f'<div class="slbl">{T["medical"]}</div>', unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        htn = st.radio(T["htn"], [T["no"],T["yes"]], horizontal=True, key="k_htn")
        htn_v = 1 if htn==T["yes"] else 0
    with c4:
        tested = st.radio(T["sugar_tested"], [T["no"],T["yes"]], horizontal=True, key="k_tested")

    told_high = False
    if tested==T["yes"]:
        told = st.radio(T["told_high"], [T["no"],T["yes"]], horizontal=True, key="k_told")
        told_high = told==T["yes"]
        if told_high:
            st.markdown(
                '<div style="background:#FDEDEC;border-left:3px solid #922B21;'
                'border-radius:4px;padding:8px 12px;font-size:12px;color:#922B21;'
                'font-weight:600;margin:4px 0">'
                + ('Previous high blood sugar is the strongest risk factor. Seek testing immediately.' if L=="en" else
                   'Inzara yari hejuru ni impamvu nkuru. Genda gusuzumwa vuba.')
                + '</div>', unsafe_allow_html=True)

    c5,c6 = st.columns(2)
    with c5:
        fam = st.radio(T["fam_diab"], [T["no"],T["yes"]], horizontal=True, key="k_fam")
        fam_v = fam==T["yes"]
        if fam_v:
            fam_who = st.selectbox(T["fam_who"], [T["fam_close"],T["fam_other"]], key="k_famwho")
    with c6:
        if sex_v==2:
            gest = st.radio(T["gest"], [T["no"],T["yes"]], horizontal=True, key="k_gest")
            gest_v = gest==T["yes"]
        else:
            gest_v = False

    st.markdown(f'<div class="slbl">{T["lifestyle"]}</div>', unsafe_allow_html=True)
    c7,c8 = st.columns(2)
    with c7:
        smk = st.radio(T["smoking"], [T["no"],T["yes"]], horizontal=True, key="k_smk")
        smk_v = 1 if smk==T["yes"] else 0
    with c8:
        alc = st.selectbox(T["alcohol"], [T["alc0"],T["alc1"],T["alc2"]], key="k_alc")
        alc_v = [T["alc0"],T["alc1"],T["alc2"]].index(alc)

    c9,c10 = st.columns(2)
    with c9:
        act = st.radio(T["activity"], [T["active"],T["inactive"]], horizontal=True, key="k_act")
        pi_v = 1 if act==T["inactive"] else 0
        if act==T["active"]:
            st.markdown('<div style="background:#D6EFE0;color:#1A7A3C;border:1px solid #1A7A3C;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:-6px;margin-bottom:4px">' + ("At least 30 min on 5 days/week" if L=="en" else "Nibura iminota 30 ku minsi 5") + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#FADBD8;color:#922B21;border:1px solid #922B21;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:-6px;margin-bottom:4px">' + ("Below 30 min on 5 days/week" if L=="en" else "Munsi 30 min Iminsi 5/cyw") + '</div>', unsafe_allow_html=True)
    with c10:
        d1_opt = st.selectbox(T["fruit"], T["days"], index=1, key="k_d1")
        d1_v = T["days"].index(d1_opt)
        if d1_v < 5:
            st.markdown('<div style="background:#FADBD8;color:#922B21;border:1px solid #922B21;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:2px">' + ("Aim for at least 5 days" if L=="en" else "Gerageza iminsi 5+") + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#D6EFE0;color:#1A7A3C;border:1px solid #1A7A3C;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:2px">' + ("Good" if L=="en" else "Byiza") + '</div>', unsafe_allow_html=True)

    c11,c12 = st.columns(2)
    with c11:
        d3_opt = st.selectbox(T["veg"], T["days"], index=4, key="k_d3")
        d3_v = T["days"].index(d3_opt)
        if d3_v < 5:
            st.markdown('<div style="background:#FADBD8;color:#922B21;border:1px solid #922B21;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:2px">' + ("Aim for at least 5 days" if L=="en" else "Gerageza iminsi 5+") + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#D6EFE0;color:#1A7A3C;border:1px solid #1A7A3C;border-radius:4px;padding:3px 10px;font-size:10px;font-weight:700;display:inline-block;margin-top:2px">' + ("Good" if L=="en" else "Byiza") + '</div>', unsafe_allow_html=True)
    with c12:
        proc_opts = [T["proc0"],T["proc1"],T["proc2"],T["proc3"],T["proc4"]]
        d7_opt = st.selectbox(T["proc"], proc_opts, index=0, key="k_d7")
        d7_raw = proc_opts.index(d7_opt) + 1
        d7_v   = 6 - d7_raw

    st.markdown(f'<div class="slbl">{T["bodysize"]}</div>', unsafe_allow_html=True)
    know_wh = st.radio(T["know_wh"], [T["yes"],T["no"]], horizontal=True, key="k_knowwh")
    bmi_v = None
    if know_wh==T["yes"]:
        cw,ch = st.columns(2)
        with cw:
            wt = st.number_input(T["weight"], 30.0, 200.0, None, step=0.5, key="k_wt", placeholder="e.g. 70")
        with ch:
            ht = st.number_input(T["height"], 100.0, 220.0, None, step=0.5, key="k_ht", placeholder="e.g. 165")
        if wt and ht and ht>0:
            bmi_v = wt/((ht/100)**2)
            bc = ("#2e7d32" if bmi_v<25 else "#e65100" if bmi_v<30 else "#c62828")
            bs = ("Underweight" if bmi_v<18.5 else "Normal" if bmi_v<25 else "Overweight" if bmi_v<30 else "Obese")
            st.markdown(f'<div style="background:{bc}18;border-left:4px solid {bc};border-radius:6px;padding:8px 14px;font-size:13px;margin-top:4px"><b style="color:{bc}">BMI {bmi_v:.1f} kg/m2  {bs}</b></div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter your weight and height to calculate BMI." if L=="en" else "Injiza uburemere n'uburebure bwawe kugira ngo babare BMI.")
            bmi_v = None
    else:
        bc_opts = [T["bc0"],T["bc1"],T["bc2"],T["bc3"]]
        bc_sel  = st.selectbox(T["bodycat"], bc_opts, index=1, key="k_bc")
        bmi_map = {T["bc0"]:17.0, T["bc1"]:22.0, T["bc2"]:27.5, T["bc3"]:33.0}
        bmi_v = bmi_map[bc_sel]

    st.markdown("<br>", unsafe_allow_html=True)
    cb1,cb2 = st.columns(2)
    with cb1:
        btn1 = st.button(T["btn_check"], key="btn_l1", use_container_width=True, type="primary")
    with cb2:
        if st.button(T["btn_reset"], key="btn_reset", use_container_width=True):
            for k in list(DEFS.keys()):
                st.session_state[k] = DEFS[k]
            widget_keys = ["k_age","k_sex","k_htn","k_tested","k_told","k_fam","k_famwho","k_gest","k_smk","k_alc","k_act","k_d1","k_d3","k_d7","k_knowwh","k_wt","k_ht","k_bc","k_wc","k_waist","k_wccat","k_lab","k_chol","k_hdl"]
            for k in widget_keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    if btn1:
        if age < 18:
            st.info("Note: This tool is validated for adults 18-69. Your result is based on the closest age in our data (18)." if L=="en" else "Icyitonderwa: Iki gikoresho gikora neza ku bantu 18-69.")
        elif age > 69:
            st.info("Note: This tool is validated for adults 18-69. Your result is based on the closest age in our data (69)." if L=="en" else "Icyitonderwa: Iki gikoresho gikora neza ku bantu 18-69.")
        bmi_missing = (know_wh == T["yes"] and bmi_v is None)
        if bmi_missing:
            st.error("Please enter your weight and height before checking your risk." if L=="en" else "Injiza uburemere n'uburebure bwawe mbere yo genzura ibyago.")
        else:
            pd_data = {
                "age"           : float(age),
                "c1"            : float(sex_v),
                "hypert"        : float(htn_v),
                "smoking"       : float(smk_v),
                "alcohol_status": float(alc_v),
                "pi"            : float(pi_v),
                "d1"            : float(d1_v),
                "d3"            : float(d3_v),
                "d7"            : float(d7_v),
                "bmi"           : float(bmi_v) if bmi_v else np.nan,
                "m14"           : np.nan,
                "b8"            : np.nan,
                "b17"           : np.nan,
                "sex_val"       : float(sex_v),
                "province"      : float(4),
                "told_high"     : told_high,
                "fam_diab"      : fam_v,
                "gest_diab"     : gest_v,
            }
            prob = predict(pd_data)
            zone = get_risk_zone(prob)
            st.session_state.update({
                "prob":prob,"risk":zone,"layers":1,
                "assessed":True,"pd_data":pd_data,
                "gauge_key":st.session_state["gauge_key"]+1
            })

    if st.session_state["assessed"] and st.session_state["layers"]>=1:
        st.markdown("---")
        show_results(st.session_state["prob"], st.session_state["risk"],
                     st.session_state["layers"], st.session_state["pd_data"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMPROVE ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state["assessed"]:
        st.info("Please complete the About You tab first." if L=="en" else "Soza ubutaka bwa mbere bw'Amakuru Yawe.")
    else:
        st.caption(T["l2cap"])
        st.markdown("""
<div style="background:#EBF5FB;border-radius:6px;padding:10px 16px;margin-bottom:12px;font-size:12.5px;color:#1A5276">
<b>Each measurement improves your result:</b> Basic / Better / Most accurate
</div>""", unsafe_allow_html=True)

        pd_data = dict(st.session_state["pd_data"])
        layers  = st.session_state["layers"]

        st.markdown('<div class="slbl">Waist Measurement</div>', unsafe_allow_html=True)
        waist_measured = st.checkbox(T["waist_check"], key="k_wc")
        if waist_measured:
            waist_val = st.number_input(T["waist"], 40.0, 160.0, None, step=0.5, key="k_waist", placeholder="e.g. 82")
            st.caption(T["waist_guide"])
            if waist_val and waist_val > 0:
                cutoff = 94 if pd_data.get("c1",2)==1 else 80
                wc_col = "#2e7d32" if waist_val<=cutoff else "#c62828"
                wc_lbl = ("Within safe range" if L=="en" else "Bisanzwe") if waist_val<=cutoff else ("Above safe limit" if L=="en" else "Irenga umurego")
                st.markdown(f'<div style="background:{wc_col}18;border-left:4px solid {wc_col};border-radius:6px;padding:8px 14px;font-size:13px;margin-top:4px"><b style="color:{wc_col}">{waist_val:.0f} cm  {wc_lbl}  (limit {cutoff} cm)</b></div>', unsafe_allow_html=True)
                pd_data["m14"] = float(waist_val)
            else:
                st.warning("Please enter your waist measurement." if L=="en" else "Injiza ingano y'ikibuno cyawe.")
        else:
            wc_opts  = [T["wc0"],T["wc1"],T["wc2"]]
            wc_sel   = st.selectbox(T["waist_cat"], wc_opts, key="k_wccat")
            wc_map_m = {T["wc0"]:75.0, T["wc1"]:98.0, T["wc2"]:110.0}
            wc_map_f = {T["wc0"]:70.0, T["wc1"]:84.0, T["wc2"]:96.0}
            wc_map   = wc_map_m if pd_data.get("c1",2)==1 else wc_map_f
            pd_data["m14"] = wc_map[wc_sel]

        st.markdown('<div class="slbl">Laboratory Results</div>', unsafe_allow_html=True)
        lab_known = st.checkbox(T["lab_check"], key="k_lab")
        chol = None
        hdl  = None
        if lab_known:
            cl1,cl2 = st.columns(2)
            with cl1:
                chol = st.number_input(T["chol"], 50.0, 400.0, None, step=1.0, key="k_chol", placeholder="e.g. 180")
            with cl2:
                hdl = st.number_input(T["hdl"], 10.0, 150.0, None, step=1.0, key="k_hdl", placeholder="e.g. 50")
            if not chol and not hdl:
                st.warning("Please enter at least one lab value." if L=="en" else "Injiza nibura agaciro kamwe ka laboratoire.")
            if chol:
                pd_data["b8"]  = float(chol)
            if hdl:
                pd_data["b17"] = float(hdl)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(T["btn_update"], key="btn_l2", use_container_width=True, type="primary"):
            waist_missing = waist_measured and not (pd_data.get("m14") and pd_data.get("m14") != M["medians"].get("m14",78.5))
            lab_missing   = lab_known and not chol and not hdl
            if waist_missing:
                st.error("Please enter your waist measurement before updating." if L=="en" else "Injiza ingano y'ikibuno mbere yo hindura ibisubizo.")
            elif lab_missing:
                st.error("Please enter at least one lab value before updating." if L=="en" else "Injiza agaciro ka laboratoire mbere yo hindura ibisubizo.")
            else:
                prob  = predict(pd_data)
                zone  = get_risk_zone(prob)
                new_layers = 3 if lab_known and pd_data.get("b8") else 2
                st.session_state.update({
                    "prob":prob,"risk":zone,
                    "layers":new_layers,"pd_data":pd_data,
                    "gauge_key":st.session_state["gauge_key"]+1
                })
        if st.session_state["layers"]>=2:
            st.markdown("---")
            show_results(st.session_state["prob"], st.session_state["risk"],
                         st.session_state["layers"], st.session_state["pd_data"])

# =============================================================================
# DISCLAIMER + FOOTER
# =============================================================================
st.markdown(f'<div class="disc">{T["disc"]}</div>', unsafe_allow_html=True)

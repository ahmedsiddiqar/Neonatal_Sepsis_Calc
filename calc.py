import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math

st.set_page_config(
    page_title="Neonatal EOS Calculator",
    page_icon="👶",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}
.stButton>button {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

st.title("👶 Neonatal Early-Onset Sepsis (EOS) Calculator")
st.markdown(
    "**Exact Kaiser Permanente Model** — Coefficients replicated from Puopolo et al. 2011, "
    "Escobar et al. 2014, Kuzniewicz et al. 2017. For infants ≥34 weeks gestation."
)

# ── Incidence → intercept lookup (from R Shiny code) ─────────────────────────
INCIDENCE_MAP = {
    0.1: 38.952265,
    0.2: 39.646367,
    0.3: 40.0528,
    0.4: 40.3415,
    0.5: 40.5656,
    0.6: 40.7489,
    0.7: 40.903919,
    0.8: 41.0384,
    0.9: 41.1571,
    1.0: 41.263432,
    2.0: 41.965852,
    4.0: 42.676976,
}

# ── Core model functions (exact R translation) ────────────────────────────────

def celsius_to_fahrenheit(temp_c: float) -> float:
    """Converts °C to °F. Mirrors R: if input < 50 → convert, else assume °F."""
    if temp_c < 50:
        return temp_c * 9 / 5 + 32
    return temp_c  # already °F


def round_ga(weeks: int, days: int) -> float:
    """Replicates the imprecise rounding used by kp.org EOS calculator."""
    ga = weeks + days / 7
    return round(ga, 2)


def calculate_logit(
    intercept: float,
    ga: float,
    mat_temp_f: float,
    rom_hours: float,
    gbs_positive: int,
    gbs_unknown: int,
    abx_broad_ge4: int,
    abx_other: int,
) -> float:
    """
    Exact logistic regression formula from R Shiny code:

    logit = intercept
            + GA * -6.9325
            + GA^2 * 0.0877
            + mat_temp_F * 0.8680
            + (ROM_hours + 0.05)^0.2 * 1.2256
            + gbs_pos * 0.5771
            + gbs_unk * 0.0427
            + abx_broad_ge4 * -1.1861
            + abx_other * -1.0488
    """
    return (
        intercept
        + ga * -6.9325
        + ga**2 * 0.0877
        + mat_temp_f * 0.8680
        + ((rom_hours + 0.05) ** 0.2) * 1.2256
        + gbs_positive * 0.5771
        + gbs_unknown * 0.0427
        + abx_broad_ge4 * -1.1861
        + abx_other * -1.0488
    )


def logit_to_prob(logit: float) -> float:
    return 1 / (1 + math.exp(-logit))


def apply_clinical_lr(prior_prob: float, clinical_status: str) -> float:
    """
    Posterior probability using likelihood ratios (exact R values):
      Ill       → LR 21.2
      Equivocal → LR  5.0
      Well      → LR  0.41
    """
    lr_map = {"Well": 0.41, "Equivocal": 5.0, "Clinically Ill": 21.2}
    lr = lr_map[clinical_status]
    odds = (prior_prob / (1 - prior_prob)) * lr
    return odds / (1 + odds)


def get_recommendation(post_risk_per1000: float, pre_risk_per1000: float, ill: bool) -> str:
    """Exact decision tree from R Shiny code."""
    if post_risk_per1000 >= 3:
        return "🔴 Empiric antibiotics, vitals per NICU"
    if ill and post_risk_per1000 < 3:
        return "🟠 Strongly consider starting empiric antibiotics, vitals per NICU"
    if not ill and 1 <= post_risk_per1000 < 3:
        return "🟡 Blood culture, vitals every 4 hours for 24 hours"
    if not ill and post_risk_per1000 < 1 and pre_risk_per1000 >= 1:
        return "🟢 No culture, no antibiotics, vitals every 4 hours for 24 hours"
    if not ill and post_risk_per1000 < 1 and pre_risk_per1000 < 1:
        return "✅ No culture, no antibiotics, routine vitals"
    return "Unable to determine — check inputs"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model sources:**
    - Puopolo et al. *Pediatrics* 2011
    - Escobar et al. *Pediatrics* 2014
    - Kuzniewicz et al. *JAMA Pediatrics* 2017
    - Benitz & Achten *Lancet ID* 2020

    Coefficients are **exact** replicas of the R Shiny reference implementation.

    **Official calculator:**  
    https://neonatalsepsiscalculator.kaiserpermanente.org
    """)
    st.divider()
    st.header("🎯 Clinical Status Guide")
    st.markdown("""
    **Well:** Normal vital signs, feeding well, normal exam

    **Equivocal:** Isolated/transient tachypnea, single temp instability, feeding concerns

    **Clinically Ill:** Persistent resp. distress, hemodynamic instability, requires O₂/CPAP
    """)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Infant Calculator", "Batch Processing (XLSX)"])

with tab1:
    st.header("Calculate EOS Risk")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Incidence & Gestational Age")

        incidence_choice = st.selectbox(
            "Baseline EOS Incidence",
            options=list(INCIDENCE_MAP.keys()),
            index=4,  # 0.5 default
            format_func=lambda x: f"{x}/1000 live births",
        )
        intercept = INCIDENCE_MAP[incidence_choice]

        ga_weeks = st.number_input(
            "Gestational Age (weeks)", min_value=34, max_value=43, value=39, step=1
        )
        ga_days = st.number_input(
            "Gestational Age (days, 0–6)", min_value=0, max_value=6, value=0, step=1
        )

        st.subheader("🌡️ Maternal Factors")

        temp_input = st.number_input(
            "Highest Intrapartum Maternal Temperature (°C or °F)",
            min_value=36.0, max_value=104.0, value=37.0, step=0.1,
            help="≤40 is treated as °C; ≥96 is treated as °F"
        )
        st.caption("≤40° treated as Celsius · ≥96° treated as Fahrenheit")

        rom_hours = st.number_input(
            "Duration of Rupture of Membranes (hours)",
            min_value=0.0, max_value=240.0, value=4.0, step=0.1
        )

        gbs = st.radio(
            "Maternal GBS Status",
            ["Negative", "Positive", "Unknown"]
        )

        abx = st.radio(
            "Intrapartum Antibiotics",
            [
                "Broad spectrum ≥4 hrs prior to birth",
                "Broad spectrum 2–4 hrs prior to birth",
                "GBS-specific ≥2 hrs prior to birth",
                "No antibiotics or any antibiotics <2 hrs prior to birth",
            ]
        )

    with col2:
        st.subheader("👶 Infant Clinical Status")
        st.info("Assess at birth and serially over first 24 hours")

        clinical_status = st.radio(
            "Clinical Presentation",
            ["Well", "Equivocal", "Clinically Ill"],
            help="See sidebar for criteria"
        )

    st.divider()

    if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):

        # ── Validate inputs ──────────────────────────────────────────────────
        errors = []
        if ga_weeks not in range(34, 44):
            errors.append("Gestational age (weeks) must be 34–43.")
        if ga_days not in range(0, 7):
            errors.append("Gestational age (days) must be 0–6.")
        if not (
            (36.0 <= temp_input <= 40.0) or (96.0 <= temp_input <= 104.0)
        ):
            errors.append("Temperature must be 36–40 °C or 96–104 °F.")
        if not (0.0 <= rom_hours <= 240.0):
            errors.append("ROM duration must be 0–240 hours.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # ── Compute binary covariates (mirrors R) ────────────────────────
            gbs_pos = 1 if gbs == "Positive" else 0
            gbs_unk = 1 if gbs == "Unknown" else 0

            # intra_abxbr: broad spectrum ≥4 hrs (index 0)
            abx_broad_ge4 = 1 if abx == "Broad spectrum ≥4 hrs prior to birth" else 0
            # intra_abxsp: broad 2-4 hrs OR GBS-specific ≥2 hrs (index 1 or 2)
            abx_other = 1 if abx in [
                "Broad spectrum 2–4 hrs prior to birth",
                "GBS-specific ≥2 hrs prior to birth",
            ] else 0

            ga = round_ga(ga_weeks, ga_days)
            mat_temp_f = celsius_to_fahrenheit(temp_input)

            logit = calculate_logit(
                intercept, ga, mat_temp_f, rom_hours,
                gbs_pos, gbs_unk, abx_broad_ge4, abx_other
            )

            pre_prob = logit_to_prob(logit)
            pre_risk = round(pre_prob * 1000, 2)

            post_prob = apply_clinical_lr(pre_prob, clinical_status)
            post_risk = round(post_prob * 1000, 2)

            ill = clinical_status == "Clinically Ill"
            recommendation = get_recommendation(post_risk, pre_risk, ill)

            # ── Display results ──────────────────────────────────────────────
            st.header("📊 Risk Assessment Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pre-exam Risk (at birth)", f"{pre_risk:.2f} / 1000")
            c2.metric(
                "LR applied",
                {"Well": "0.41", "Equivocal": "5.0", "Clinically Ill": "21.2"}[clinical_status],
                f"Status: {clinical_status}"
            )
            c3.metric(
                "Post-exam Risk",
                f"{post_risk:.2f} / 1000",
                delta=f"{post_risk - pre_risk:+.2f}",
                delta_color="inverse"
            )

            st.divider()
            st.subheader("📋 Clinical Recommendation")

            color = (
                "error" if "🔴" in recommendation or "🟠" in recommendation
                else "warning" if "🟡" in recommendation
                else "success"
            )
            getattr(st, color)(recommendation)

            # Expanded guidance
            st.subheader("📋 Clinical Management")
            if post_risk >= 3 or (ill and post_risk < 3):
                st.markdown("""
- Obtain blood culture **before** starting antibiotics
- Start empiric antibiotics (ampicillin + gentamicin)
- CBC with differential
- Vitals monitoring per NICU protocol
- Continue antibiotics pending culture (minimum 48 h)
                """)
            elif not ill and 1 <= post_risk < 3:
                st.markdown("""
- Obtain blood culture
- Vital signs every **4 hours for 24 hours**
- Do **NOT** start empiric antibiotics unless clinical deterioration
- Reassess clinical status at 12 and 24 hours
- Start antibiotics if status worsens or risk rises ≥3/1000
                """)
            else:
                st.markdown("""
- Routine newborn care
- Vital signs per unit protocol (or every 4 h if pre-risk ≥1)
- No blood tests or antibiotics needed
- Parent education on warning signs
- Reassess if clinical concerns arise
                """)

# ── Batch tab ─────────────────────────────────────────────────────────────────
with tab2:
    st.header("📊 Batch Processing — Multiple Infants")
    st.markdown("""
Upload an Excel file. Required columns (case-insensitive):

| Column | Values |
|---|---|
| `Infant_ID` | any |
| `Incidence` | 0.1, 0.2 … 4.0 |
| `GA_Weeks` | 34–43 |
| `GA_Days` | 0–6 |
| `Maternal_Temp` | °C (36–40) or °F (96–104) |
| `ROM_Hours` | 0–240 |
| `GBS` | Negative / Positive / Unknown |
| `Antibiotics` | `broad_ge4` / `broad_2to4` / `gbs_ge2` / `none` |
| `Clinical_Status` | Well / Equivocal / Clinically Ill |
    """)

    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    if uploaded:
        try:
            df = pd.read_excel(uploaded)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            st.dataframe(df.head(10))

            abx_map = {
                "broad_ge4": ("broad_ge4", 0),
                "broad_2to4": ("other", 0),
                "gbs_ge2": ("other", 0),
                "none": (None, None),
            }

            if st.button("🔍 Calculate Batch Risks", type="primary"):
                results = []
                prog = st.progress(0)

                for i, row in df.iterrows():
                    prog.progress((i + 1) / len(df))

                    inc = float(row.get("incidence", 0.5))
                    intercept_b = INCIDENCE_MAP.get(inc, INCIDENCE_MAP[0.5])

                    ga_w = int(row.get("ga_weeks", 39))
                    ga_d = int(row.get("ga_days", 0))
                    ga = round_ga(ga_w, ga_d)

                    temp = float(row.get("maternal_temp", 37.0))
                    temp_f = celsius_to_fahrenheit(temp)

                    rom = float(row.get("rom_hours", 4.0))

                    gbs_val = str(row.get("gbs", "Unknown"))
                    gbs_p = 1 if gbs_val == "Positive" else 0
                    gbs_u = 1 if gbs_val == "Unknown" else 0

                    abx_raw = str(row.get("antibiotics", "none")).lower()
                    abx_br = 1 if abx_raw == "broad_ge4" else 0
                    abx_ot = 1 if abx_raw in ("broad_2to4", "gbs_ge2") else 0

                    cs = str(row.get("clinical_status", "Well"))

                    logit = calculate_logit(
                        intercept_b, ga, temp_f, rom, gbs_p, gbs_u, abx_br, abx_ot
                    )
                    pre_p = logit_to_prob(logit)
                    pre_r = round(pre_p * 1000, 2)

                    post_p = apply_clinical_lr(pre_p, cs if cs in ["Well", "Equivocal", "Clinically Ill"] else "Well")
                    post_r = round(post_p * 1000, 2)

                    ill_b = cs == "Clinically Ill"
                    rec = get_recommendation(post_r, pre_r, ill_b)

                    results.append({
                        "Infant_ID": row.get("infant_id", f"Infant_{i+1}"),
                        "GA": f"{ga_w}w{ga_d}d",
                        "Pre_Risk": pre_r,
                        "Clinical_Status": cs,
                        "Post_Risk": post_r,
                        "Recommendation": rec,
                    })

                prog.empty()
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(res_df))
                col2.metric("Antibiotics", (res_df["Recommendation"].str.contains("antibiotic")).sum())
                col3.metric("Routine", (res_df["Recommendation"].str.contains("routine")).sum())

                out = BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    res_df.to_excel(w, index=False, sheet_name="EOS_Results")
                st.download_button(
                    "📥 Download Results",
                    data=out.getvalue(),
                    file_name="eos_batch_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""


⚠️ **Disclaimer:** For clinical decision support only. Not a substitute for clinical judgment.  
Official calculator: https://neonatalsepsiscalculator.kaiserpermanente.org
""")

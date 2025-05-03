import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import numpy_financial as npf # For NPV calculation
import io
from datetime import datetime
import itertools # For sensitivity analysis combinations

# --- Configuration & Constants ---
# Define sensitivity analysis steps (percentage points)
SENSITIVITY_CAPEX_STEPS = [0.0, 12.5, 25.0]  # e.g., 0%, 12.5%, 25% increase
SENSITIVITY_OPEX_STEPS = [0.0, 12.5, 25.0]   # e.g., 0%, 12.5%, 25% increase
SENSITIVITY_REVENUE_STEPS = [0.0, -5.0, -25.0] # e.g., 0%, 5%, 10% decrease

# --- Helper Functions ---

def format_php(value):
    """Formats a number as Philippine Peso currency."""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return "PHP -"
    try:
        return f"PHP {value:,.2f}"
    except ValueError:
        return f"PHP {value:.2f}" # Fallback basic formatting

def calculate_npv(rate, initial_investment, cashflows):
    """Calculates Net Present Value using numpy_financial."""
    if rate <= -1: return np.nan # Avoid division by zero or negative denominator issues
    if initial_investment < 0: initial_investment = 0 # Ensure initial investment isn't negative
    try:
        # npf.npv requires the rate first, then the cashflows (starting from year 1)
        # We subtract the initial investment separately as it occurs at time 0.
        present_value_of_cashflows = npf.npv(rate, cashflows)
        return present_value_of_cashflows - initial_investment
    except Exception:
        return np.nan # Return NaN if calculation fails

def calculate_payback_period(initial_investment, cashflows):
    """Calculates the simple payback period."""
    if initial_investment <= 0: return 0
    cumulative_cashflow = 0
    if not isinstance(cashflows, (list, np.ndarray)):
        return float('inf')

    for i, cashflow in enumerate(cashflows):
        if cashflow <= 0 and i == len(cashflows) - 1 and cumulative_cashflow < initial_investment:
            return float('inf')

        prev_cumulative = cumulative_cashflow
        cumulative_cashflow += cashflow

        if cumulative_cashflow >= initial_investment:
            if cashflow <= 0:
                 return i if prev_cumulative >= initial_investment else float('inf')
            years = i
            needed_for_payback = initial_investment - prev_cumulative
            fractional_year = needed_for_payback / cashflow
            return years + fractional_year

    return float('inf')

# Function to load CapEx percentages from CSV
def load_capex_percentages(filepath='CapEx Percentages.csv'):
    """
    Loads CapEx breakdown percentages from a specified CSV file.
    Expected columns: 'Category', '% of Total CapEx'.
    Returns a DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath)
        # Basic validation
        if 'Category' not in df.columns or '% of Total CapEx' not in df.columns:
            st.warning(f"CSV '{filepath}' is missing required columns ('Category', '% of Total CapEx'). Falling back to default breakdown.")
            return None
        # Remove any empty rows or completely empty columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        # Clean up percentage values (assuming they might have '%')
        if df['% of Total CapEx'].dtype == 'object':
            df['% of Total CapEx'] = df['% of Total CapEx'].astype(str).str.replace('%', '', regex=False).astype(float)
        # Convert percentage to fraction
        df['Percentage'] = df['% of Total CapEx'] / 100.0
        # Ensure percentages sum reasonably close to 1 (or 100)
        if not np.isclose(df['Percentage'].sum(), 1.0, atol=0.01):
             st.warning(f"Percentages in '{filepath}' do not sum close to 100%. Sum = {df['% of Total CapEx'].sum():.2f}%. Breakdown might be inaccurate.")
        return df[['Category', 'Subcategory', 'Percentage']]
    except FileNotFoundError:
        st.info(f"Optional file '{filepath}' not found. Using CapEx breakdown based on input fields.")
        return None
    except Exception as e:
        st.error(f"Error loading CapEx percentages from '{filepath}': {e}")
        return None

# Function to calculate NPV for sensitivity analysis
def calculate_sensitivity_npv(base_capex, base_opex, base_revenue_components, discount_rate, lifespan,
                              capex_esc_pct, opex_esc_pct, rev_dec_pct):
    """
    Recalculates NPV based on sensitivity adjustments.
    base_revenue_components is the dictionary of revenue streams.
    """
    # Adjust CapEx
    sensitive_capex = base_capex * (1 + capex_esc_pct / 100.0)

    # Adjust OpEx
    sensitive_opex = base_opex * (1 + opex_esc_pct / 100.0)

    # Adjust Revenue (apply decrease to each positive component)
    sensitive_revenue_total = 0
    for component, value in base_revenue_components.items():
        if value > 0: # Only decrease positive revenue streams
             sensitive_revenue_total += value * (1 - rev_dec_pct / 100.0)
        else: # Keep negative revenue (like costs/negative tipping fees) as is
             sensitive_revenue_total += value

    # Recalculate annual gross profit
    sensitive_annual_profit = sensitive_revenue_total - sensitive_opex

    # Recalculate NPV
    sensitive_cash_flows = [sensitive_annual_profit] * int(lifespan)
    sensitive_npv = calculate_npv(discount_rate / 100.0, sensitive_capex, sensitive_cash_flows)

    return sensitive_npv


# --- Page Configuration ---
st.set_page_config(
    page_title="Hydrochar Facility Feasibility - Philippines",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main container adjustments */
    .block-container {
        padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;
        max-width: 1200px; margin: auto;
    }
    h1 { text-align: center; font-size: 2.5rem !important; font-weight: 700 !important; color: #1f2937; margin-bottom: 0.5rem !important; }
    .stApp > header { background-color: transparent; }
    p.subtitle { text-align: center; font-size: 1.1rem !important; color: #4b5563; margin-bottom: 2.5rem !important; }
    h2 { font-size: 1.75rem !important; font-weight: 600 !important; color: #111827; margin-top: 2.5rem !important; margin-bottom: 1.25rem !important; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.6rem; }
    h3 { font-size: 1.25rem !important; font-weight: 600 !important; color: #1f2937; margin-top: 2rem !important; margin-bottom: 1rem !important; }
    h4 { font-size: 1rem !important; font-weight: 600 !important; color: #374151; margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
    .metric-card { background-color: white; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); text-align: center; border: 1px solid #e5e7eb; margin-bottom: 1rem; height: 100%; display: flex; flex-direction: column; justify-content: center; }
    div[data-testid="stMetric"] { background-color: white; border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1.5rem 1.5rem 1.5rem 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); }
    div[data-testid="stMetric"] > div:nth-child(1) { color: #6b7280; font-size: 0.875rem !important; }
    div[data-testid="stMetric"] > div:nth-child(2) { font-size: 1.875rem !important; font-weight: 600 !important; color: #111827; line-height: 1.2; padding-bottom: 0.25rem; }
    div[data-testid="stMetricDelta"] { display: none; }
    div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div, div[data-testid="stTextInput"] input { border: 1px solid #d1d5db !important; border-radius: 0.375rem !important; padding: 0.5rem 0.75rem !important; transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out; box-shadow: none !important; }
    div[data-testid="stNumberInput"] input:focus, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within, div[data-testid="stTextInput"] input:focus { outline: none !important; border-color: #2563eb !important; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important; }
    div[data-testid="stSlider"] { padding-top: 0.5rem; }
    .st-emotion-cache-1xarl3l { font-size: 0.75rem !important; color: #6b7280 !important; }
    div[data-testid="stInfo"] { background-color: #eff6ff !important; border: 1px solid #bfdbfe !important; color: #1e40af !important; padding: 0.75rem 1rem !important; border-radius: 0.5rem !important; font-size: 0.875rem !important; margin-top: 0.5rem !important; }
    div[data-testid="stInfo"] strong { color: #1e40af !important; }
    .stTable, .dataframe { border: none !important; border-radius: 0.5rem !important; overflow: hidden !important; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important; margin-top: 1rem; }
    .stTable thead th, .dataframe thead th { background-color: #f9fafb !important; font-weight: 600 !important; color: #374151 !important; border-bottom: 1px solid #e5e7eb !important; padding: 0.75rem 1rem !important; text-align: left !important; }
    .stTable tbody td, .dataframe tbody td { padding: 0.75rem 1rem !important; border-bottom: 1px solid #e5e7eb !important; text-align: left !important; }
    .stTable tbody tr:last-child td, .dataframe tbody tr:last-child td { border-bottom: none !important; }
    .stTable tfoot td { font-weight: 600 !important; background-color: #f9fafb !important; color: #374151 !important; padding: 0.75rem 1rem !important; border-top: 1px solid #e5e7eb !important; }
    div[data-testid="stTabs"] button[role="tab"] { padding: 0.75rem 1rem; border-bottom-width: 2px; transition: color 0.2s ease-in-out, border-color 0.2s ease-in-out; font-weight: 500; }
    div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom-color: #2563eb; color: #2563eb; font-weight: 600; }
    div[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
    /* Sensitivity Table Specific */
    .sensitivity-table th { text-align: center !important; }
    .sensitivity-table td { text-align: right !important; padding: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Page Title and Subtitle ---
st.title("PFDA Hydrochar Plant : A Rapid Cost Analysis")
st.markdown("<p class='subtitle'>Woodfields Consultants Inc.</p>", unsafe_allow_html=True)

# --- Load External Data ---
# Attempt to load CapEx percentages from CSV
capex_percentages_df = load_capex_percentages() # Use default filename

# --- Create Main Interface Tabs ---
tab_dashboard, tab_inputs, tab_info = st.tabs(["📊 Dashboard & Results", "⚙️ Input Parameters", "ℹ️ Information"])

# --- Input Parameters Tab ---
with tab_inputs:
    st.header("⚙️ Input Parameters")

    # Create subtabs for different types of inputs
    input_tab_facility, input_tab_tech, input_tab_capex, input_tab_opex, input_tab_uasb, input_tab_financial, input_tab_sensitivity = st.tabs([
        "Facility Capacity",
        "Products and Revenue Sources",
        "CAPEx",
        "OPEx",
        "Methane Production",
        "Discount Rate",
        "Sensitivity Analysis" # Keep this tab
    ])

    # --- Sub-Tab 1: Facility & Feedstock ---
    with input_tab_facility:
        st.subheader("Facility & Feedstock Parameters")
        col1, col2 = st.columns(2)
        with col1:
            feedstock_throughput = st.number_input("Feedstock Throughput (tons/year)", min_value=100, value=22000, step=100, help="Annual amount of raw feedstock processed.")
            moisture_content = st.slider("Feedstock Moisture Content (%)", min_value=5.0, max_value=95.0, value=80.0, step=1.0, help="Percentage of water in the raw feedstock.")
            avg_feedstock_transport_dist = st.number_input("Avg. Feedstock Transport Distance (km)", min_value=0.0, value=0.0, step=5.0, help="Average distance to transport feedstock.")
        with col2:
            feedstock_options = ["Fish Waste", "Agricultural Residues", "Municipal Solid Waste", "Pulp Mill Sludge", "Other"]
            feedstock_type = st.selectbox("Type of Feedstock", options=feedstock_options, index=0, help="Select the primary feedstock type.")
            feedstock_cost = st.number_input("Tipping Fee for Feedstock (PHP/ton)", min_value=-1000.0, max_value=5000.0, value=1500.0, step=50.0, help="Fee received (+) or paid (-) per ton of feedstock.")

    # --- Sub-Tab 2: Technology & Products ---
    with input_tab_tech:
        st.subheader("HTC Technology & Product Parameters")
        st.markdown("#### HTC Process Parameters")
        col1, col2, col3 = st.columns(3)
        with col1: reaction_temp = st.number_input("Reaction Temperature (°C)", 150, 300, 250, 5)
        with col2: reaction_time = st.number_input("Reaction Time (hours)", 0.5, 12.0, 8.0, 0.5)
        with col3: water_biomass_ratio = st.number_input("Water-to-Biomass Ratio", 1.0, 20.0, 9.0, 0.5)

        st.markdown("#### Hydrochar Product Parameters")
        col1, col2 = st.columns(2)
        with col1:
            default_hydrochar_yield = 95.0 if feedstock_type == "Fish Waste" else 50.0
            hydrochar_yield_dry = st.slider("Hydrochar Yield (Dry Basis, %)", 1.0, 100.0, default_hydrochar_yield, 1.0, help="Percentage of *dry* feedstock converted to hydrochar.")
            avg_hydrochar_transport_dist = st.number_input("Avg. Hydrochar Transport Distance (km)", 0.0, 500.0, 50.0, 5.0, help="Average distance to transport hydrochar.")
        with col2:
            hydrochar_selling_price = st.number_input("Selling Price of Hydrochar (PHP/ton)", 1000.0, 50000.0, 25000.0, 500.0, help="Estimated market price.")
            carbon_credits_price = st.number_input("Carbon Credits Value (PHP/ton hydrochar)", 0.0, 50000.0, 12500.0, 500.0, help="Value of carbon credits per ton hydrochar.")

        if feedstock_type == "Fish Waste":
            st.markdown("---"); st.markdown("#### Fish Oil Product Parameters")
            st.info("Parameters for fish oil production (active for Fish Waste feedstock).")
            col1, col2, col3 = st.columns(3)
            with col1: fish_oil_yield = st.slider("Fish Oil Yield (% raw feedstock)", 1.0, 30.0, 15.0, 0.5, help="Percentage of raw fish waste converted to fish oil.")
            with col2: fish_oil_density = st.number_input("Fish Oil Density (kg/L)", 0.80, 1.00, 0.92, 0.01)
            with col3: fish_oil_price_per_liter = st.number_input("Fish Oil Price (PHP/liter)", 1.0, 10.0, 10.0, 0.5)
        else:
            fish_oil_yield = 0.0; fish_oil_density = 0.92; fish_oil_price_per_liter = 0.0

    # --- Sub-Tab 3: Capital Expenditure (CapEx) ---
    with input_tab_capex:
        st.subheader("Capital Expenditure (CapEx)")
        col1, col2 = st.columns(2)
        # These inputs are used to CALCULATE the total CapEx.
        # The breakdown visualization might use the CSV percentages if available.
        with col1:
            land_cost = st.number_input("Land Cost (PHP)", 0.0, 100000000.0, 0.0, 500000.0, format="%.0f")
            pre_processing_costs = st.number_input("Pre-processing Equipment (PHP)", 0.0, 50000000.0, 2500000.0, 100000.0, format="%.0f", help="Shredders, grinders, etc.")
            storage_costs = st.number_input("Storage Facilities (PHP)", 0.0, 20000000.0, 1000000.0, 25000.0, format="%.0f", help="Feedstock and hydrochar storage.")
        with col2:
             reactor_cost_per_ton_capacity = st.number_input("Reactor Cost per ton/yr Capacity (PHP)", 1000.0, 50000.0, 17000.0, 250.0, help="Cost of HTC reactor(s) per ton annual throughput.")
             reactor_costs_calculated = reactor_cost_per_ton_capacity * feedstock_throughput # Store calculated value
             st.info(f"Est. Reactor Cost: **{format_php(reactor_costs_calculated)}**")
             post_processing_costs = st.number_input("Post-processing Equipment (PHP)", 0.0, 500000000.0, max(3500000.0, reactor_costs_calculated * 0.05 ), 100000.0, format="%.0f", help="Dewatering, drying, etc.")
             installation_construction_costs = st.number_input("Installation & Construction (PHP)", 0.0, 500000000.0, max(3500000.0, reactor_costs_calculated * 0.05 ), 100000.0, format="%.0f")

        contingency_costs_percent = st.slider("Contingency Costs (%)", 0.0, 30.0, 25.0, 1.0, help="Percentage of total direct CapEx for unforeseen costs.")

    # --- Sub-Tab 4: Operational Expenditure (OpEx) ---
    with input_tab_opex:
        st.subheader("Operational Expenditure (OpEx - Annual)")
        st.markdown("#### Labor Costs")
        col1, col2 = st.columns(2)
        with col1: num_employees = st.number_input("Number of Employees", 1, 100, 30, 1)
        with col2:
            avg_annual_salary = st.number_input("Avg. Annual Salary per Employee (PHP)", 100000.0, 1000000.0, 450000.0, 10000.0, format="%.0f")
            st.info(f"Est. Annual Labor Cost: **{format_php(num_employees * avg_annual_salary)}**")

        st.markdown("#### Utility Costs")
        col1, col2 = st.columns(2)
        with col1:
            electricity_cost_kwh = st.number_input("Electricity Cost (PHP/kWh)", 1.0, 25.0, 12.5.0, 0.5)
            water_cost_m3 = st.number_input("Water Cost (PHP/cubic meter)", 10.0, 200.0, 50.0, 5.0)
        with col2:
            electricity_consumption_per_ton = st.number_input("Est. Electricity Use (kWh/ton feedstock)", 10.0, 2000.0, 350.0, 10.0)
            st.info(f"Est. Annual Electricity Cost (Gross): **{format_php(electricity_consumption_per_ton * feedstock_throughput * electricity_cost_kwh)}**")
            water_consumption_per_ton = st.number_input("Est. Water Use (m³/ton feedstock)", 0.1, 10.0, 0.5, 0.1)
            st.info(f"Est. Annual Water Cost: **{format_php(water_consumption_per_ton * feedstock_throughput * water_cost_m3)}**")

        st.markdown("#### Transportation & Other Costs")
        col1, col2 = st.columns(2)
        with col1:
            transport_cost_per_ton_km = st.number_input("Transportation Cost (PHP/ton-km)", 1.0, 20.0, 5.0, 0.5)
            consumables_costs = st.number_input("Annual Consumables Costs (PHP)", 0.0, 5000000.0, 1000000.0, 10000.0, format="%.0f", help="Chemicals, catalysts, etc.")
            marketing_sales_costs = st.number_input("Annual Marketing & Sales Costs (PHP)", 0.0, 5000000.0, 2000000.0, 10000.0, format="%.0f")
        with col2:
            maintenance_costs_percent = st.slider("Maintenance Costs (% of Total CapEx)", 1.0, 10.0, 2.0, 0.5, help="Annual maintenance as % of total initial CapEx.")
            waste_disposal_costs = st.number_input("Annual Waste Disposal Costs (PHP)", 0.0, 5000000.0, 100000.0, 5000.0, format="%.0f", help="Costs for disposing process residues.")

    # --- Sub-Tab 5: UASB Reactor Parameters ---
    with input_tab_uasb:
        st.subheader("Upflow Anaerobic Sludge Blanket (UASB) Reactor")
        st.info("Processes hydrochar waste water (if Fish Waste feedstock) to generate methane for energy.")
        enable_uasb = st.checkbox("Enable UASB Reactor", value=True, help="Enable/disable UASB system (only active for Fish Waste).")

        if enable_uasb:
            st.markdown("#### UASB Process Parameters")
            col1, col2 = st.columns(2)
            with col1:
                cod_concentration = st.number_input("COD Concentration (ppm)", 10000.0, 300000.0, 150000.0, 5000.0, help="Chemical Oxygen Demand in waste water.")
                cod_removal_efficiency = st.slider("COD Removal Efficiency (%)", 50.0, 99.0, 90.0, 1.0, help="% COD removed.")
            with col2:
                waste_water_percent = st.slider("Waste Water (% of feedstock)", 50.0, 95.0, 85.0, 1.0, help="% feedstock converted to waste water.")
                organic_loading_rate = st.number_input("Organic Loading Rate (kg COD/m³/day)", 5.0, 25.0, 20.0, 1.0, help="Organic matter processed per m³ reactor per day.")
                methane_per_cod = st.number_input("Methane Production (L/g COD removed)", 0.05, 0.5, 0.20, 0.01, help="Liters CH4 per kg COD removed.")

            st.markdown("#### Energy Parameters")
            col1, col2 = st.columns(2)
            with col1:
                methane_density = st.number_input("Methane Density (kg/m³)", 0.5, 0.8, 0.656, 0.001)
                methane_energy_density = st.number_input("Methane Energy Content (kWh/kg)", 10.0, 15.0, 13.9, 0.1)
            with col2:
                electricity_generation_efficiency = st.slider("Electricity Generation Efficiency (%)", 30.0, 80.0, 75.0, 1.0, help="Efficiency converting methane energy to electricity.")

            st.markdown("#### UASB Cost Parameters")
            col1, col2 = st.columns(2)
            with col1:
                uasb_reactor_cost_per_m3 = st.number_input("UASB Reactor Cost (PHP/m³)", 5000.0, 100000.0, 20000.0, 100.0, help="Cost per m³ UASB reactor volume.")
                uasb_preprocessing_cost = st.number_input("UASB Pre-processing Equipment (PHP)", 0.0, 10000000.0, 3000000.0, 100000.0, format="%.0f")
            with col2:
                uasb_postprocessing_cost = st.number_input("UASB Post-processing Equipment (PHP)", 0.0, 10000000.0, 5000000.0, 100000.0, format="%.0f")
                uasb_maintenance_percent = st.slider("Annual Maintenance Cost (% of UASB CapEx)", 1.0, 10.0, 5.0, 0.5, help="Annual maintenance as % of UASB CapEx.")
        else:
            # Assign default/zero values if UASB is disabled
            cod_concentration, cod_removal_efficiency, waste_water_percent = 0, 0, 0
            organic_loading_rate, methane_per_cod, methane_density = 0, 0, 0
            methane_energy_density, electricity_generation_efficiency = 0, 0
            uasb_reactor_cost_per_m3, uasb_preprocessing_cost = 0, 0
            uasb_postprocessing_cost, uasb_maintenance_percent = 0, 0

        # Placeholders for tables to be filled after calculations
        energy_summary_container = st.container()
        uasb_engineering_container = st.container()
        if enable_uasb:
            with energy_summary_container: st.markdown("#### Energy Production Summary"); st.info("Summary will appear after calculations.")
            with uasb_engineering_container: st.markdown("#### UASB Engineering Details"); st.info("Details will appear after calculations.")

    # --- Sub-Tab 6: Financial Parameters ---
    with input_tab_financial:
        st.subheader("Financial Parameters")
        col1, col2 = st.columns(2)
        with col1:
            discount_rate = st.slider("Discount Rate (%)", 1.0, 25.0, 4.0, 0.5, help="Rate for discounting future cash flows for NPV.")
        with col2:
            project_lifespan = st.number_input("Project Lifespan (years)", 5, 50, 25, 1, help="Expected operational duration.")

    # --- Sub-Tab 7: Sensitivity Analysis Parameters ---
    with input_tab_sensitivity:
        st.subheader("Sensitivity Analysis Parameters")
        st.info("Select the % changes to analyze. The resulting NPV will be shown on the Dashboard. A full sensitivity matrix using predefined steps will also be displayed there.")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Use unique keys for widgets if they might appear elsewhere or be recreated
            sensitivity_capex_esc = st.slider(
                "CapEx Escalation (%)", 0.0, 25.0, 20.0, 1.0,
                key="sens_capex_slider", help="Select % increase in total CapEx to analyze."
            )
        with col2:
            sensitivity_opex_esc = st.slider(
                "OpEx Escalation (%)", 0.0, 25.0, 20.0, 1.0,
                key="sens_opex_slider", help="Select % increase in annual OpEx to analyze."
            )
        with col3:
            sensitivity_rev_dec = st.slider(
                "Revenue Decrease (%)", 0.0, 25.0, 10.0, 0.5,
                key="sens_rev_slider", help="Select % decrease in annual Revenue to analyze."
            )
        st.markdown("""---""")
        st.markdown(f"""
        **Matrix Parameters:** The sensitivity matrix on the dashboard will explore these ranges:
        * CapEx Escalation: **{', '.join(map(str, SENSITIVITY_CAPEX_STEPS))}%**
        * OpEx Escalation: **{', '.join(map(str, SENSITIVITY_OPEX_STEPS))}%**
        * Revenue Decrease: **{', '.join(map(str, SENSITIVITY_REVENUE_STEPS))}%**
        """)


# --- Calculations (Performed after all inputs are defined) ---

# 1. Dry Feedstock
dry_matter_fraction = (100.0 - moisture_content) / 100.0
dry_feedstock_throughput = feedstock_throughput * dry_matter_fraction

# 2. Annual Hydrochar Production
annual_hydrochar_production = dry_feedstock_throughput * (hydrochar_yield_dry / 100.0)

# 2.0 UASB Reactor Calculations (Only if enabled AND Fish Waste)
uasb_reactor_volume = 0; annual_methane_production_liters = 0; annual_methane_production_kg = 0
annual_energy_production_kwh = 0; annual_electricity_production_kwh = 0
uasb_capex = 0; annual_uasb_opex = 0; annual_energy_savings = 0
uasb_active = enable_uasb and feedstock_type == "Fish Waste"

if uasb_active:
    try:
        annual_waste_water_tons = feedstock_throughput * (waste_water_percent / 100.0)
        daily_waste_water_tons = annual_waste_water_tons / 365
        daily_waste_water_m3 = daily_waste_water_tons # Approx. 1 ton water = 1 m³
        daily_cod_load_kg = daily_waste_water_m3 * (cod_concentration / 1000.0) # ppm = mg/L = g/m³ -> kg/m³

        uasb_reactor_volume = daily_cod_load_kg / organic_loading_rate if organic_loading_rate > 0 else 0
        annual_cod_removed_kg = daily_cod_load_kg * (cod_removal_efficiency / 100.0) * 365
        annual_methane_production_liters = annual_cod_removed_kg * methane_per_cod * 1000
        annual_methane_production_m3 = annual_methane_production_liters / 1000.0
        annual_methane_production_kg = annual_methane_production_m3 * methane_density

        annual_energy_production_kwh = annual_methane_production_kg * methane_energy_density
        annual_electricity_production_kwh = annual_energy_production_kwh * (electricity_generation_efficiency / 100.0)
        annual_energy_savings = annual_electricity_production_kwh * electricity_cost_kwh

        uasb_reactor_construction_cost = uasb_reactor_volume * uasb_reactor_cost_per_m3
        uasb_capex = uasb_preprocessing_cost + uasb_postprocessing_cost + uasb_reactor_construction_cost
        annual_uasb_opex = uasb_capex * (uasb_maintenance_percent / 100.0)

        energy_production_per_ton = annual_energy_production_kwh / feedstock_throughput if feedstock_throughput > 0 else 0
        electricity_production_per_ton = annual_electricity_production_kwh / feedstock_throughput if feedstock_throughput > 0 else 0
        energy_balance_per_ton = electricity_production_per_ton - electricity_consumption_per_ton
        hydraulic_retention_time = uasb_reactor_volume / daily_waste_water_m3 if daily_waste_water_m3 > 0 else 0
        methane_per_ton_feedstock = annual_methane_production_m3 / feedstock_throughput if feedstock_throughput > 0 else 0

        # Populate UASB Tables in Input Tab
        with energy_summary_container:
             st.markdown("#### Energy Production Summary") # Add title back
             energy_summary_data = {
                "Metric": ["Annual Methane Production", "Annual Energy Production (Methane)", "Annual Electricity Generation", "Reactor Volume", "Energy Production per Ton Feedstock", "Electricity Generation per Ton Feedstock", "Facility Electricity Consumption per Ton", "Energy Balance per Ton Feedstock"],
                "Value": [f"{annual_methane_production_m3:,.2f} m³/year", f"{annual_energy_production_kwh:,.2f} kWh/year", f"{annual_electricity_production_kwh:,.2f} kWh/year", f"{uasb_reactor_volume:,.2f} m³", f"{energy_production_per_ton:,.2f} kWh/ton", f"{electricity_production_per_ton:,.2f} kWh/ton", f"{electricity_consumption_per_ton:,.2f} kWh/ton", f"{energy_balance_per_ton:,.2f} kWh/ton"]
             }
             st.table(pd.DataFrame(energy_summary_data))
             if energy_balance_per_ton >= 0: st.success(f"**Positive Energy Balance:** UASB produces more electricity ({electricity_production_per_ton:.2f} kWh/ton) than facility consumes ({electricity_consumption_per_ton:.2f} kWh/ton).")
             else: st.warning(f"**Negative Energy Balance:** Facility consumes more electricity ({electricity_consumption_per_ton:.2f} kWh/ton) than UASB produces ({electricity_production_per_ton:.2f} kWh/ton).")

        with uasb_engineering_container:
            st.markdown("#### UASB Engineering Details") # Add title back
            engineering_details_data = {
                "Parameter": ["Waste Water Treatment Capacity (Annual)", "Waste Water Treatment Capacity (Daily)", "UASB Reactor Volume", "COD Concentration in Wastewater", "COD Loading Rate (Daily)", "Organic Loading Rate (OLR)", "COD Removal Efficiency", "Hydraulic Retention Time (HRT)", "Methane Generation Rate", "Methane Production per kg COD Removed", "Methane-to-Electricity Conversion Efficiency", "UASB System Capital Cost", "UASB Annual Maintenance Cost"],
                "Value": [f"{annual_waste_water_tons:,.2f} m³/year", f"{daily_waste_water_m3:,.2f} m³/day", f"{uasb_reactor_volume:,.2f} m³", f"{cod_concentration:,.0f} mg/L", f"{daily_cod_load_kg:,.2f} kg COD/day", f"{organic_loading_rate:,.2f} kg COD/m³/day", f"{cod_removal_efficiency:,.1f}%", f"{hydraulic_retention_time:,.2f} days", f"{methane_per_ton_feedstock:,.2f} m³ CH₄/ton feedstock", f"{methane_per_cod:,.2f} L CH₄/g COD", f"{electricity_generation_efficiency:,.1f}%", format_php(uasb_capex), format_php(annual_uasb_opex)]
            }
            st.table(pd.DataFrame(engineering_details_data))
            st.markdown("""**Notes:** Reactor volume based on OLR & COD load. HRT is avg. wastewater time in reactor. Methane yield depends on COD removal & production factor.""")

    except ZeroDivisionError:
        st.warning("Cannot calculate UASB details due to zero division (likely zero OLR or flow). Check inputs.")
        uasb_active = False # Disable further UASB impact if calculation fails
        uasb_capex = 0
        annual_uasb_opex = 0
        annual_energy_savings = 0
        annual_electricity_production_kwh = 0
    except Exception as e:
        st.error(f"Error during UASB calculation: {e}")
        uasb_active = False
        uasb_capex = 0
        annual_uasb_opex = 0
        annual_energy_savings = 0
        annual_electricity_production_kwh = 0


# 2.1 Annual Fish Oil Production (Conditional)
annual_fish_oil_production_liters = 0
annual_fish_oil_revenue = 0
is_fish_waste = (feedstock_type == "Fish Waste")
if is_fish_waste:
    annual_fish_oil_production_tons = feedstock_throughput * (fish_oil_yield / 100.0)
    if fish_oil_density > 0:
        annual_fish_oil_production_liters = (annual_fish_oil_production_tons * 1000) / fish_oil_density
    annual_fish_oil_revenue = annual_fish_oil_production_liters * fish_oil_price_per_liter

# 3. Total Capital Expenditure (CapEx)
# Define base CapEx items from inputs
capex_items_input = {
    "Land": land_cost,
    "Reactor(s)": reactor_costs_calculated, # Use the calculated value
    "Pre-processing Equipment": pre_processing_costs,
    "Post-processing Equipment": post_processing_costs,
    "Storage Facilities": storage_costs,
    "Installation & Construction": installation_construction_costs
}
# Add UASB CapEx if active
if uasb_active:
    capex_items_input["UASB System"] = uasb_capex

# Calculate total CapEx before contingency based on input items
total_capex_before_contingency = sum(v for v in capex_items_input.values() if v is not None and v > 0)
contingency_amount = total_capex_before_contingency * (contingency_costs_percent / 100.0)
total_capex = total_capex_before_contingency + contingency_amount

# Determine the final CapEx breakdown for display
# Priority: Use CSV if loaded successfully. Fallback: Use input-based items.
if capex_percentages_df is not None:
    # Calculate breakdown based on CSV percentages applied to the *final* total_capex
    capex_breakdown_list = []
    for index, row in capex_percentages_df.iterrows():
        category = row['Category']
        subcategory = row['Subcategory']
        percentage = row['Percentage']
        cost = total_capex * percentage
        capex_breakdown_list.append({'Category': category, 'Subcategory': subcategory, 'Cost (PHP)': cost, 'Percentage': percentage * 100})
    capex_breakdown_df_display = pd.DataFrame(capex_breakdown_list)
    st.sidebar.info("CapEx breakdown uses percentages from 'CapEx Percentages.csv'.")
else:
    # Fallback: Use the input-based items + calculated contingency
    capex_items_display = capex_items_input.copy()
    capex_items_display["Contingency"] = contingency_amount
    capex_breakdown_list = []
    for category, cost in capex_items_display.items():
         percentage = (cost / total_capex * 100) if total_capex > 0 and cost is not None else 0
         capex_breakdown_list.append({'Category': category, 'Cost (PHP)': cost if cost is not None else 0, 'Percentage': percentage})
    capex_breakdown_df_display = pd.DataFrame(capex_breakdown_list)
    st.sidebar.warning("CapEx breakdown uses calculated values based on input fields (CSV not found or invalid).")


# 4. Total Annual Operational Expenditure (OpEx)
total_labor_costs = num_employees * avg_annual_salary
gross_electricity_consumption_kwh = electricity_consumption_per_ton * feedstock_throughput
electricity_offset_kwh = min(annual_electricity_production_kwh, gross_electricity_consumption_kwh) if uasb_active else 0
net_electricity_consumption_kwh = gross_electricity_consumption_kwh - electricity_offset_kwh
total_electricity_cost = net_electricity_consumption_kwh * electricity_cost_kwh
total_water_cost = water_consumption_per_ton * feedstock_throughput * water_cost_m3
annual_feedstock_transport_cost = feedstock_throughput * avg_feedstock_transport_dist * transport_cost_per_ton_km
annual_hydrochar_transport_cost = annual_hydrochar_production * avg_hydrochar_transport_dist * transport_cost_per_ton_km
total_annual_transport_costs = annual_feedstock_transport_cost + annual_hydrochar_transport_cost
annual_maintenance_cost = total_capex * (maintenance_costs_percent / 100.0) # Maintenance based on TOTAL CapEx

opex_items = {
    "Labor": total_labor_costs,
    "Electricity (Net)": total_electricity_cost,
    "Water": total_water_cost,
    "Transportation": total_annual_transport_costs,
    "Maintenance": annual_maintenance_cost,
    "Consumables": consumables_costs,
    "Waste Disposal": waste_disposal_costs,
    "Marketing & Sales": marketing_sales_costs
}
# Add UASB Maintenance OpEx if active
if uasb_active:
    opex_items["UASB Maintenance"] = annual_uasb_opex

total_opex = sum(v for v in opex_items.values() if v is not None and v > 0)


# 5. Annual Revenue
annual_hydrochar_revenue = annual_hydrochar_production * hydrochar_selling_price
annual_carbon_credits_revenue = annual_hydrochar_production * carbon_credits_price
annual_tipping_fee_revenue = feedstock_throughput * feedstock_cost # Can be negative (cost)

revenue_items = {
    "Hydrochar Sales": annual_hydrochar_revenue,
    "Carbon Credits": annual_carbon_credits_revenue,
    "Tipping/Acceptance Fees": annual_tipping_fee_revenue
}
if is_fish_waste and annual_fish_oil_revenue > 0:
    revenue_items["Fish Oil Sales"] = annual_fish_oil_revenue

# Calculate total revenue (sum of positive components for display)
total_revenue_positive_components = sum(v for v in revenue_items.values() if v is not None and v > 0)

# Calculate UASB savings percentage relative to OpEx *without* savings
uasb_savings_percentage = 0.0
if uasb_active and annual_energy_savings > 0:
    total_opex_without_savings = total_opex + annual_energy_savings # OpEx if no savings occurred
    uasb_savings_percentage = (annual_energy_savings / total_opex_without_savings) * 100 if total_opex_without_savings > 0 else 0.0

# 6. Annual Gross Profit (All revenue components - Total OpEx)
# Use the full sum of revenue_items, including potentially negative tipping fees
annual_gross_profit = sum(v for v in revenue_items.values() if v is not None) - total_opex

# 7. Profitability Metrics
hydrochar_production_cost_per_ton = total_opex / annual_hydrochar_production if annual_hydrochar_production > 0 else 0
profit_margin = (annual_gross_profit / total_revenue_positive_components) * 100 if total_revenue_positive_components > 0 else 0

# Base NPV Calculation
cash_flows = [annual_gross_profit] * int(project_lifespan)
base_npv = calculate_npv(discount_rate / 100.0, total_capex, cash_flows)

# Base Payback Period Calculation
base_payback_period = calculate_payback_period(total_capex, cash_flows)


# --- Dashboard & Results Tab ---
with tab_dashboard:
    st.header("📊 Dashboard Summary")
    st.markdown("Economic viability assessment of the Hydrothermal Carbonization (HTC) facility. Adjust parameters in 'Input Parameters' tab.")

    # --- Excel Download Section ---
    st.sidebar.header("📥 Export Results")
    excel_buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Sheet 1: Engineering Features
            engineering_data = {
                "Parameter": ["Feedstock Type", "Annual Feedstock Throughput (tons)", "Feedstock Moisture Content (%)", "Dry Matter Content (tons/year)", "Hydrochar Yield (Dry Basis, %)", "Annual Hydrochar Production (tons)"],
                "Value": [feedstock_type, f"{feedstock_throughput:,.0f}", f"{moisture_content:.1f}", f"{dry_feedstock_throughput:,.2f}", f"{hydrochar_yield_dry:.1f}", f"{annual_hydrochar_production:,.2f}"]
            }
            if uasb_active:
                engineering_data["Parameter"].extend(["--- UASB System ---", "UASB Reactor Volume (m³)", "Annual Methane Production (m³)", "Annual Electricity Generation (kWh)", "Energy Balance per Ton Feedstock (kWh/ton)"])
                engineering_data["Value"].extend(["", f"{uasb_reactor_volume:,.2f}", f"{annual_methane_production_m3:,.2f}", f"{annual_electricity_production_kwh:,.2f}", f"{energy_balance_per_ton:,.2f}"])
            if is_fish_waste:
                engineering_data["Parameter"].extend(["--- Fish Oil Production ---", "Fish Oil Yield (% raw feedstock)", "Annual Fish Oil Production (liters)"])
                engineering_data["Value"].extend(["", f"{fish_oil_yield:.1f}", f"{annual_fish_oil_production_liters:,.2f}"])
            engineering_data["Parameter"].extend(["--- HTC Process ---", "Reaction Temp (°C)", "Reaction Time (hr)", "Water:Biomass Ratio"])
            engineering_data["Value"].extend(["", f"{reaction_temp}", f"{reaction_time}", f"{water_biomass_ratio}"])
            pd.DataFrame(engineering_data).to_excel(writer, sheet_name='Engineering Features', index=False)

            # Sheet 2: Cost Breakdown
            capex_df_excel = capex_breakdown_df_display.copy()
            capex_df_excel['Cost (PHP)'] = capex_df_excel['Cost (PHP)'].apply(format_php)
            capex_df_excel['Percentage'] = capex_df_excel['Percentage'].apply(lambda x: f"{x:.2f}%")
            capex_df_excel.rename(columns={'Category': 'Capital Expenditure Item', 'Percentage': '% of Total'}, inplace=True)

            opex_df_excel = pd.DataFrame(opex_items.items(), columns=['Operational Expenditure Item', 'Annual Cost (PHP)'])
            opex_df_excel['% of Total'] = [(v/total_opex*100) if total_opex > 0 else 0 for v in opex_df_excel['Annual Cost (PHP)']]
            opex_df_excel['Annual Cost (PHP)'] = opex_df_excel['Annual Cost (PHP)'].apply(format_php)
            opex_df_excel['% of Total'] = opex_df_excel['% of Total'].apply(lambda x: f"{x:.2f}%")

            pd.DataFrame([{"Cost Category": "CAPITAL EXPENDITURE (CAPEX)", "Amount (PHP)": format_php(total_capex), "Details": "One-time investment"}]).to_excel(writer, sheet_name='Cost Breakdown', index=False, startrow=0)
            capex_df_excel.to_excel(writer, sheet_name='Cost Breakdown', index=False, startrow=2)
            pd.DataFrame([{"Cost Category": "OPERATIONAL EXPENDITURE (OPEX)", "Amount (PHP)": format_php(total_opex), "Details": "Annual recurring costs"}]).to_excel(writer, sheet_name='Cost Breakdown', index=False, startrow=len(capex_df_excel)+4)
            opex_df_excel.to_excel(writer, sheet_name='Cost Breakdown', index=False, startrow=len(capex_df_excel)+6)

            # Sheet 3: Financial KPIs
            payback_display_excel = f"{base_payback_period:.2f} years" if base_payback_period != float('inf') else f"> {int(project_lifespan)} years"
            kpi_data = {
                "Key Performance Indicator": ["Net Present Value (NPV)", "Discount Rate Used", "Simple Payback Period", "Annual Revenue (Positive Streams)", "Annual Gross Profit", "Gross Profit Margin", "Hydrochar Production Cost", "Project Lifespan"],
                "Value": [format_php(base_npv), f"{discount_rate:.1f}%", payback_display_excel, format_php(total_revenue_positive_components), format_php(annual_gross_profit), f"{profit_margin:.2f}%", f"{format_php(hydrochar_production_cost_per_ton)} / ton", f"{int(project_lifespan)} years"]
            }
            pd.DataFrame(kpi_data).to_excel(writer, sheet_name='Financial KPIs', index=False)

            # Sheet 4: Revenue Analysis
            revenue_df_excel = pd.DataFrame(revenue_items.items(), columns=['Revenue Stream', 'Annual Amount (PHP)'])
            revenue_df_excel['% of Total Positive'] = [(v/total_revenue_positive_components*100) if total_revenue_positive_components > 0 and v > 0 else '-' for v in revenue_df_excel['Annual Amount (PHP)']]
            revenue_df_excel['Annual Amount (PHP)'] = revenue_df_excel['Annual Amount (PHP)'].apply(format_php)
            revenue_df_excel['% of Total Positive'] = revenue_df_excel['% of Total Positive'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (float, int)) else x)
            revenue_df_excel.to_excel(writer, sheet_name='Revenue Analysis', index=False)

            # Sheet 5: Cash Flow Projection
            years = list(range(int(project_lifespan) + 1))
            cashflow_proj = [-total_capex] + cash_flows
            cumulative_cashflow = np.cumsum(cashflow_proj).tolist()
            cashflow_df_excel = pd.DataFrame({'Year': years, 'Annual Cash Flow (PHP)': map(format_php, cashflow_proj), 'Cumulative Cash Flow (PHP)': map(format_php, cumulative_cashflow)})
            cashflow_df_excel.to_excel(writer, sheet_name='Cash Flow Projection', index=False)

            # Sheet 6: Sensitivity Analysis (if parameters exist)
            if 'sensitivity_capex_esc' in locals(): # Check if sensitivity inputs were rendered
                sensitivity_results = []
                param_combinations = list(itertools.product(SENSITIVITY_CAPEX_STEPS, SENSITIVITY_OPEX_STEPS, SENSITIVITY_REVENUE_STEPS))
                for capex_p, opex_p, rev_p in param_combinations:
                    npv_sens = calculate_sensitivity_npv(total_capex, total_opex, revenue_items, discount_rate, project_lifespan, capex_p, opex_p, rev_p)
                    sensitivity_results.append({
                        'CapEx Escalation (%)': capex_p,
                        'OpEx Escalation (%)': opex_p,
                        'Revenue Decrease (%)': rev_p,
                        'Resulting NPV (PHP)': format_php(npv_sens) if not pd.isna(npv_sens) else 'N/A'
                    })
                sens_df_excel = pd.DataFrame(sensitivity_results)
                sens_df_excel.to_excel(writer, sheet_name='Sensitivity Analysis', index=False)

            # Format Excel columns
            workbook = writer.book
            money_format = workbook.add_format({'num_format': '"PHP "#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.0%'})
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.autofit() # Adjust column widths
                # Apply formatting (example for specific sheets/columns)
                if sheet_name == 'Cost Breakdown':
                    worksheet.set_column('B:B', 18, money_format) # Column B width 18, money format
                if sheet_name == 'Financial KPIs':
                     worksheet.set_column('B:B', 25) # Wider column for values
                if sheet_name == 'Sensitivity Analysis':
                    worksheet.set_column('D:D', 18) # Wider column for NPV

        excel_download_ready = True
    except Exception as e:
        st.sidebar.error(f"Failed to generate Excel report: {e}")
        excel_download_ready = False

    if excel_download_ready:
        st.sidebar.download_button(
            label="📥 Download Full Report (Excel)",
            data=excel_buffer.getvalue(),
            file_name=f"Hydrochar_UASB_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # Correct MIME for .xlsx
            help="Download comprehensive results including sensitivity analysis."
        )
    else:
        st.sidebar.warning("Excel report generation failed. Check inputs or logs.")


    # --- Summary Section ---
    st.subheader("Key Project Overview")
    col1, col2 = st.columns(2)
    payback_display = f"{base_payback_period:.2f} years" if base_payback_period != float('inf') else f"> {int(project_lifespan)} years"
    with col1:
        st.markdown(f"""
        * Feedstock: **{feedstock_type}** ({feedstock_throughput:,.0f} tons/yr)
        * Hydrochar Output: **{annual_hydrochar_production:,.1f} tons/yr** ({hydrochar_yield_dry:.1f}% yield)
        * Lifespan: **{int(project_lifespan)} years**
        * UASB Active: **{'Yes' if uasb_active else 'No'}**
        """)
    with col2:
        st.markdown(f"""
        * Total CapEx: **{format_php(total_capex)}**
        * Annual OpEx: **{format_php(total_opex)}**
        * Annual Revenue: **{format_php(total_revenue_positive_components)}** (Positive Streams)
        * Annual Profit: **{format_php(annual_gross_profit)}**
        """)

    # --- Key Metric: NPV ---
    st.metric(label=f"Base Net Present Value (NPV @ {discount_rate:.1f}%)", value=format_php(base_npv))
    st.caption(f"Present value of future profits minus initial investment, discounted at {discount_rate:.1f}%.")

    # --- Financial KPIs in Metrics ---
    st.subheader("Financial KPIs")
    cols = st.columns(4)
    with cols[0]: st.metric("Base NPV", format_php(base_npv))
    with cols[1]: st.metric("Payback Period", payback_display)
    with cols[2]: st.metric("Hydrochar Prod. Cost", f"{format_php(hydrochar_production_cost_per_ton)} / ton")
    with cols[3]: st.metric("Gross Profit Margin", f"{profit_margin:.2f}%" if total_revenue_positive_components > 0 else "-")


    # --- Visualizations ---
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)

    with col1: # Revenue Sources Pie Chart
        st.markdown("<h4 style='text-align: center;'>Revenue Sources</h4>", unsafe_allow_html=True)
        display_revenue_items = {k: v for k, v in revenue_items.items() if v is not None and v > 0} # Only positive
        if display_revenue_items and total_revenue_positive_components > 0:
            revenue_df = pd.DataFrame(display_revenue_items.items(), columns=['Source', 'Revenue (PHP)'])
            fig_revenue = px.pie(revenue_df, values='Revenue (PHP)', names='Source', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel1, height=350)
            fig_revenue.update_traces(textposition='inside', textinfo='percent+label',
                                      hovertemplate='<b>%{label}</b><br>Revenue: %{customdata[0]}<br>%{percent}<extra></extra>',
                                      customdata=[[format_php(v)] for v in revenue_df['Revenue (PHP)']])
            fig_revenue.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_revenue, use_container_width=True)
        else:
            st.warning("No positive revenue sources to display.")

    with col2: # Cash Position Line Chart
        st.markdown("<h4 style='text-align: center;'>Projected Cash Position (Cumulative)</h4>", unsafe_allow_html=True)
        years_simple = list(range(int(project_lifespan) + 1))
        cashflow_proj_simple = [-total_capex] + cash_flows
        cumulative_cashflow_simple = np.cumsum(cashflow_proj_simple).tolist()
        cashflow_df_simple = pd.DataFrame({'Year': years_simple, 'Net Cash Position (PHP)': cumulative_cashflow_simple})
        fig_simple_cashflow = px.line(cashflow_df_simple, x='Year', y='Net Cash Position (PHP)', markers=True, height=350)
        fig_simple_cashflow.add_hline(y=0, line_dash="dash", line_color="red")
        fig_simple_cashflow.update_layout(margin=dict(t=20, b=0, l=0, r=0))
        fig_simple_cashflow.update_traces(hovertemplate='End of Year %{x}: %{y:,.0f} PHP<extra></extra>')
        st.plotly_chart(fig_simple_cashflow, use_container_width=True)


    # --- Detailed Results Tables ---
    st.header("💰 Detailed Financials")
    col1, col2 = st.columns(2)

    with col1: # Revenue Breakdown Table
        st.subheader("Revenue Breakdown")
        revenue_df_table = pd.DataFrame(revenue_items.items(), columns=['Source', 'Amount (PHP)'])
        revenue_df_table['Percentage'] = revenue_df_table['Amount (PHP)'].apply(
            lambda x: f"{(x/total_revenue_positive_components*100):.1f}%" if total_revenue_positive_components > 0 and x > 0 else ("-" if x <= 0 else "0.0%")
        )
        revenue_df_table['Amount (PHP)'] = revenue_df_table['Amount (PHP)'].apply(format_php)
        # Add footer for total positive revenue
        footer = pd.DataFrame([{"Source": "Total Annual Revenue (Positive)", "Amount (PHP)": format_php(total_revenue_positive_components), "Percentage": "100%"}])
        revenue_table_display = pd.concat([revenue_df_table, footer], ignore_index=True)
        st.table(revenue_table_display.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))

        # UASB Savings Table (if applicable)
        if uasb_active and annual_energy_savings > 0:
            st.markdown("#### UASB Operational Savings")
            uasb_savings_data = {
                "Savings Category": ["Electricity Cost Reduction"],
                "Amount (PHP)": [format_php(annual_energy_savings)],
                "Impact (% of OpEx w/o Savings)": [f"{uasb_savings_percentage:.2f}%"]
            }
            st.table(pd.DataFrame(uasb_savings_data))
            st.info(f"UASB electricity generation reduces estimated annual OpEx by {uasb_savings_percentage:.2f}%.")

    with col2: # Overall Summary Table
        st.subheader("Overall Summary")
        summary_metrics = ["Annual Feedstock Input", "Annual Hydrochar Output", "Total CapEx", "Annual OpEx", "Annual Revenue (Positive)", "Annual Gross Profit"]
        summary_values = [f"{feedstock_throughput:,.0f} tons", f"{annual_hydrochar_production:,.2f} tons", format_php(total_capex), format_php(total_opex), format_php(total_revenue_positive_components), format_php(annual_gross_profit)]
        if is_fish_waste and annual_fish_oil_production_liters > 0:
            summary_metrics.insert(2, "Annual Fish Oil Output")
            summary_values.insert(2, f"{annual_fish_oil_production_liters:,.2f} liters")
        if uasb_active and uasb_savings_percentage > 0:
            summary_metrics.append("OpEx Savings from UASB")
            summary_values.append(f"{uasb_savings_percentage:.2f}%")
        summary_df = pd.DataFrame({"Metric": summary_metrics, "Value": summary_values})
        st.table(summary_df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))


    # --- Cost Breakdown Charts & Tables ---
    st.header("📉 Cost Breakdowns")
    
    # CapEx Breakdown
    st.markdown("<h4 style='text-align: center;'>Capital Expenditure (CapEx)</h4>", unsafe_allow_html=True)
    # Use the capex_breakdown_df_display determined earlier
    capex_df_chart = capex_breakdown_df_display[capex_breakdown_df_display['Cost (PHP)'] > 0].copy()
    if not capex_df_chart.empty:
        # If we have subcategories, we need to aggregate by category for the pie chart
        if 'Subcategory' in capex_df_chart.columns:
            # Group by Category and sum the costs
            capex_df_agg = capex_df_chart.groupby('Category')['Cost (PHP)'].sum().reset_index()
            fig_capex = px.pie(capex_df_agg, values='Cost (PHP)', names='Category', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set3, height=350)
            fig_capex.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate='<b>%{label}</b><br>Cost: %{customdata[0]}<br>%{percent}<extra></extra>',
                                  customdata=[[format_php(v)] for v in capex_df_agg['Cost (PHP)']])
        else:
            # Simple case - just use the data as is
            fig_capex = px.pie(capex_df_chart, values='Cost (PHP)', names='Category', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set3, height=350)
            fig_capex.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate='<b>%{label}</b><br>Cost: %{customdata[0]}<br>%{percent}<extra></extra>',
                                  customdata=[[format_php(v)] for v in capex_df_chart['Cost (PHP)']])
        fig_capex.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_capex, use_container_width=True)
        
        # Display Table below chart including Subcategory if available
        capex_df_table = capex_df_chart.copy()
        capex_df_table['Cost (PHP)'] = capex_df_table['Cost (PHP)'].apply(format_php)
        capex_df_table['Percentage'] = capex_df_table['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        # Check if Subcategory exists in the dataframe (from CSV)
        if 'Subcategory' in capex_df_table.columns:
            # First display the detailed table with subcategories
            st.dataframe(capex_df_table[['Category', 'Subcategory', 'Cost (PHP)', 'Percentage']], 
                         use_container_width=True, hide_index=True)
            
            # Then create and display the aggregated table by Category
            capex_agg_table = capex_df_chart.groupby('Category').agg({
                'Cost (PHP)': 'sum',
                'Percentage': 'sum'
            }).reset_index()
            # Sort by Percentage in descending order
            capex_agg_table = capex_agg_table.sort_values(by='Percentage', ascending=False)
            capex_agg_table['Cost (PHP)'] = capex_agg_table['Cost (PHP)'].apply(format_php)
            capex_agg_table['Percentage'] = capex_agg_table['Percentage'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(capex_agg_table[['Category', 'Cost (PHP)', 'Percentage']], 
                        use_container_width=True, hide_index=True)
        else:
            st.dataframe(capex_df_table[['Category', 'Cost (PHP)', 'Percentage']], 
                        use_container_width=True, hide_index=True)
    else:
        st.warning("No positive CapEx items to display.")

    # OpEx Breakdown - keeping under the same column layout
    st.markdown("<h4 style='text-align: center;'>Annual Operational Expenditure (OpEx)</h4>", unsafe_allow_html=True)
    opex_df_chart = pd.DataFrame(opex_items.items(), columns=['Category', 'Cost (PHP)'])
    opex_df_chart = opex_df_chart[opex_df_chart['Cost (PHP)'] > 0].copy()
    if not opex_df_chart.empty:
        fig_opex = px.pie(opex_df_chart, values='Cost (PHP)', names='Category', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Pastel, height=350)
        fig_opex.update_traces(textposition='inside', textinfo='percent+label',
                               hovertemplate='<b>%{label}</b><br>Cost: %{customdata[0]}<br>%{percent}<extra></extra>',
                               customdata=[[format_php(v)] for v in opex_df_chart['Cost (PHP)']])
        fig_opex.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_opex, use_container_width=True)
        
        # Display Table below chart
        opex_df_table = opex_df_chart.copy()
        opex_df_table['Percentage'] = opex_df_table['Cost (PHP)'].apply(lambda x: (x/total_opex*100) if total_opex > 0 else 0)
        opex_df_table['Cost (PHP)'] = opex_df_table['Cost (PHP)'].apply(format_php)
        opex_df_table['Percentage'] = opex_df_table['Percentage'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(opex_df_table[['Category', 'Cost (PHP)', 'Percentage']], use_container_width=True, hide_index=True)
    else:
        st.warning("No positive OpEx items to display.")

    # --- Sensitivity Analysis Section ---
    st.header("🔬 Sensitivity Analysis")
    st.markdown("""
    This section explores how the Net Present Value (NPV) changes based on potential variations
    in Capital Expenditure (CapEx), Operational Expenditure (OpEx), and Revenue.
    """)

    # 1. NPV based on user-selected sensitivity parameters from Input tab
    st.subheader("NPV at Selected Sensitivity Levels")
    # Recalculate NPV using the slider values
    selected_sensitivity_npv = calculate_sensitivity_npv(
        base_capex=total_capex,
        base_opex=total_opex,
        base_revenue_components=revenue_items, # Pass the dict
        discount_rate=discount_rate,
        lifespan=project_lifespan,
        capex_esc_pct=sensitivity_capex_esc, # Value from slider
        opex_esc_pct=sensitivity_opex_esc,   # Value from slider
        rev_dec_pct=sensitivity_rev_dec      # Value from slider
    )

    st.metric(
        label=f"Calculated NPV (CapEx +{sensitivity_capex_esc:.1f}%, OpEx +{sensitivity_opex_esc:.1f}%, Revenue -{sensitivity_rev_dec:.1f}%)",
        value=format_php(selected_sensitivity_npv) if not pd.isna(selected_sensitivity_npv) else "N/A",
        delta=format_php(selected_sensitivity_npv - base_npv) if not pd.isna(selected_sensitivity_npv) and not pd.isna(base_npv) else None,
        delta_color="inverse" # Red for decrease from base NPV
    )
    st.caption("Compares the NPV under selected sensitivity parameters against the base case NPV.")


    # 2. Sensitivity Matrix Table
    st.subheader("NPV Sensitivity Matrix")
    st.markdown(f"""
    The table below shows calculated NPVs for different combinations of CapEx and OpEx escalations,
    assuming a specific Revenue decrease of **{SENSITIVITY_REVENUE_STEPS[2]}%** (maximum value from range).
    *Base Case NPV (0% changes): {format_php(base_npv)}*
    """) # Using the maximum revenue decrease for the main table display

    # Generate data for the matrix (using the maximum revenue decrease step)
    matrix_data = []
    fixed_revenue_decrease = SENSITIVITY_REVENUE_STEPS[2] # e.g., -5.0%

    for opex_p in SENSITIVITY_OPEX_STEPS:
        row_data = {'OpEx Escalation (%)': f"{opex_p}%"}
        for capex_p in SENSITIVITY_CAPEX_STEPS:
            npv_val = calculate_sensitivity_npv(total_capex, total_opex, revenue_items, discount_rate, project_lifespan,
                                                capex_p, opex_p, abs(fixed_revenue_decrease)) # Use absolute for decrease %
            row_data[f"CapEx +{capex_p}%"] = format_php(npv_val) if not pd.isna(npv_val) else "N/A"
        matrix_data.append(row_data)

    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.set_index('OpEx Escalation (%)', inplace=True)

    # Display the matrix table using st.dataframe for better formatting control
    st.dataframe(matrix_df, use_container_width=True)
    st.caption(f"Table shows NPVs with Revenue Decrease fixed at {abs(fixed_revenue_decrease)}%. See downloadable Excel report for full 3-way sensitivity.")

    # Optional: Add a tornado plot or other sensitivity visualization if needed (more complex)


# --- Information Tab ---
with tab_info:
    st.header("ℹ️ Information on Integrated Hydrochar & UASB Systems")
    tab_htc, tab_uasb, tab_integration, tab_revenue = st.tabs(["Hydrochar Process", "UASB Technology", "Integrated System", "Revenue Streams"])
    # (Content for these tabs remains the same as provided previously)
    with tab_htc:
        st.markdown("""
        ### Hydrothermal Carbonization (HTC)
        HTC converts wet biomass into hydrochar (coal-like solid) using hot, pressurized water (180-250°C).
        #### Advantages:
        * **Handles Wet Feedstock:** No pre-drying needed (good for PH waste like rice/coconut husk, fish waste).
        * **Versatile:** Processes various organic materials.
        * **Valuable Products:** Hydrochar (biofuel, soil amendment, adsorbent), process liquids.
        * **Waste Valorization:** Reduces landfill burden.
        * **Carbon Sequestration:** Stabilizes biomass carbon.
        #### Philippine Context:
        * Abundant feedstock available.
        * Logistics (collection/transport) are key.
        * Market development for hydrochar needed.
        * Technology cost/scalability considerations.
        * Policy/incentives influence feasibility.
        """)
    with tab_uasb:
        st.markdown("""
        ### Upflow Anaerobic Sludge Blanket (UASB) Reactor
        Anaerobic wastewater treatment generating biogas (methane). Treats high-strength organic wastewater, like HTC process liquid.
        #### Operation:
        Wastewater flows up through microbial sludge blanket -> Organics degraded -> Biogas (CH4, CO2) produced -> Separator collects gas, treated water, and retains sludge.
        #### Key Parameters:
        * **Organic Loading Rate (OLR):** kg COD/m³/day (determines size).
        * **Chemical Oxygen Demand (COD):** Organic content (higher = more CH4).
        * **Hydraulic Retention Time (HRT):** Time wastewater stays in reactor.
        * **Temperature & pH:** Need optimal ranges (Mesophilic/Thermophilic, pH 6.8-7.2).
        #### Advantages:
        * **Energy Generation:** Methane for heat/electricity.
        * **Low OpEx:** Minimal energy input, low sludge production.
        * **Compact:** Small footprint.
        * **Efficient:** High COD removal (80-95%).
        """)
    with tab_integration:
        st.markdown("""
        ### Integrated HTC-UASB System
        Combines HTC and UASB for maximum energy/resource recovery from wet organic waste.
        #### Process Flow:
        Waste -> HTC -> Hydrochar (Solid) + Process Liquid -> UASB -> Biogas + Treated Water
        Biogas -> Energy Generation (Electricity/Heat) -> Offsets facility needs.
        #### Synergies:
        * **Thermal Integration:** HTC heat can warm UASB.
        * **Water Recycling:** Treated UASB water potentially reused in HTC.
        * **Energy Self-Sufficiency:** UASB energy significantly reduces HTC energy demand.
        #### Considerations:
        * Matching scales of HTC and UASB.
        * UASB start-up time (sludge granulation).
        * Process control and monitoring.
        """)
    with tab_revenue:
        st.markdown("""
        ### Revenue Streams & Economic Benefits
        #### Primary Sources:
        1.  **Hydrochar Sales:** As fuel, soil amendment, adsorbent precursor.
        2.  **Fish Oil:** (If Fish Waste) Industrial uses, animal feed, biodiesel potential.
        3.  **Carbon Credits:** Soil sequestration, fossil fuel displacement.
        4.  **Tipping Fees:** Payment for accepting waste.
        #### UASB Benefits:
        1.  **Energy Savings:** Offsets electricity/heat costs.
        2.  **Reduced Treatment Costs:** Lower external wastewater disposal fees.
        3.  **Environmental Compliance:** Meets discharge standards.
        #### Economic Synergies:
        * Extracts value from solid *and* liquid waste fractions.
        * Improves overall energy ROI.
        * Diversifies revenue, reducing risk.
        """)
    st.info("This tool provides a preliminary economic assessment. Detailed feasibility requires site-specific data, market analysis, and engineering design.")

# --- Footer ---
st.markdown("---")
st.caption(f"""

*Webapp developed by:*

**Engr. John Paul Renzo Jucar, MSc.**  
*Principal Engineer*  
Environment Division  
Woodfields Consultants Inc.
  
**Engr. Marisheen Macarasig, CE, SE**  
*Vice President*  
Environment Division  
Woodfields Consultants Inc.

**Reynaldo Medina, PhD**  
*Chairman Emeritus*  
Woodfields Consultants Inc.
            
**Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**
""")

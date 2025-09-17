#Import all required libraries for the dashboard
import streamlit as st  # Web dashboard framework
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
import plotly.express as px  # Interactive visualizations
import plotly.graph_objects as go  # Advanced plotting
from datetime import datetime, timedelta  # Date/time handling
from collections import defaultdict  # Dictionary with default values
from sklearn.linear_model import LinearRegression  # Machine learning model
import warnings
warnings.filterwarnings('ignore')  # Hide warnings for cleaner output

# --- Page config ---
st.set_page_config(
    page_title="🍕 Pizza Dashboard",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Simple CSS ---
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #FF6B6B; text-align: center; margin-bottom: 2rem;}
.metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍕 Smart Byte - Pizza Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### TechLabs Data Science Project - Winter 2025")

# Add project introduction
with st.expander("📖 About This Project", expanded=False):
    st.markdown("""
    **Welcome to the Pizza Inventory Optimization Dashboard!**

    This project demonstrates key data science concepts applied to a real-world business problem:

    🎯 **Problem**: How can a pizza restaurant optimize inventory to minimize waste while ensuring customer satisfaction?

    📊 **Data**: 48,620 pizza orders from 2015 with 32 pizza types and 64 ingredients

    🧠 **Methods Used**:
    - **Machine Learning**: Linear regression for sales forecasting
    - **Inventory Simulation**: FEFO (First Expired, First Out) logic
    - **Optimization**: Bayesian optimization for parameter tuning
    - **Data Visualization**: Interactive charts and dashboards

    🔍 **Key Learning Areas**:
    - Data preprocessing and feature engineering
    - Time series forecasting
    - Simulation modeling
    - Business KPI calculation
    - Dashboard development with Streamlit

    **Navigation Guide**:
    - 📊 **KPI Overview**: Explore business metrics and trends
    - 📦 **Inventory Tracking**: Run simulations and optimize parameters
    - 📈 **Sales Forecast**: Test ML predictions and plan future demand
    """)

# --- Load data ---
@st.cache_data  #This decorator caches the function result to avoid reloading data on every interaction
def load_data():
    try:
        #pd.read_excel() reads Excel files into a pandas DataFrame (like a table)
        df = pd.read_excel('Data Model - Pizza Sales.xlsx')
        #Convert text dates to datetime objects for time-based analysis
        df["order_date"] = pd.to_datetime(df["order_date"])
        #Data cleaning - replace corrupted characters and standardize ingredient names
        df['pizza_ingredients'] = df['pizza_ingredients'].str.replace('慛', 'N')
        df['pizza_ingredients'] = df['pizza_ingredients'].str.replace('Artichokes', 'Artichoke')
        df['pizza_ingredients'] = df['pizza_ingredients'].str.replace('garlic', 'Garlic', case=False)
        return df
    except FileNotFoundError:
        st.error("❌ 'Data Model - Pizza Sales.xlsx' nicht gefunden.")
        return None

df = load_data()
if df is None:
    st.stop()

# Sidebar/menu
page = st.selectbox(
    "Navigate to:",
    ["📊 KPI Overview", "📦 Inventory Tracking", "📈 Sales Forecast"],
    label_visibility="collapsed"
)

# --- Sidebar: Zeitraum ---
st.sidebar.header("🎛️ Control Panel")
st.sidebar.subheader("📅 Analysis Period")
analysis_period = st.sidebar.selectbox(
    "Select analysis period:",
    ["First 3 Months (Jan-Mar)", "Full Year", "Custom Range"]
)

if analysis_period == "Custom Range":
    start_date = st.sidebar.date_input("Start Date", df['order_date'].min())
    end_date = st.sidebar.date_input("End Date", df['order_date'].max())
elif analysis_period == "First 3 Months (Jan-Mar)":
    start_date = pd.to_datetime('2015-01-01').date()
    end_date = pd.to_datetime('2015-03-31').date()
else:
    start_date = df['order_date'].min().date()
    end_date = df['order_date'].max().date()

# --- Helper: Recipes + ingredient data ---
@st.cache_data
def get_ingredient_data():
    ingredient_masses_grams = {
        "Sliced Ham": 90, "Pepperoni": 80, "Bacon": 70, "Calabrese Salami": 80,
        "Capocollo": 80, "Chicken": 100, "Barbecued Chicken": 100,
        "Prosciutto di San Daniele": 60, "Prosciutto": 60,
        "Beef Chuck Roast": 110, "Italian Sausage": 100, "Chorizo Sausage": 100,
        "Soppressata Salami": 80, "Anchovies": 35, "Genoa Salami": 80,
        "Coarse Sicilian Salami": 80, "Luganega Sausage": 100, "Pancetta": 70,
        "Nduja Salami": 70, "Mozzarella Cheese": 130, "Provolone Cheese": 90,
        "Smoked Gouda Cheese": 90, "Romano Cheese": 25, "Blue Cheese": 60,
        "Feta Cheese": 70, "Asiago Cheese": 70, "Goat Cheese": 60,
        "Ricotta Cheese": 80, "Gorgonzola Piccante Cheese": 60,
        "Parmigiano Reggiano Cheese": 25, "Fontina Cheese": 90,
        "Gouda Cheese": 90, "Brie Carre Cheese": 60, "Mushrooms": 70,
        "Red Onions": 50, "Onions": 50, "Caramelized Onions": 60,
        "Red Peppers": 70, "Green Peppers": 70, "Friggitello Peppers": 40,
        "Jalapeno Peppers": 25, "Peperoncini verdi": 25, "Tomatoes": 80,
        "Plum Tomatoes": 80, "Sun-dried Tomatoes": 35, "Spinach": 40,
        "Arugula": 20, "Artichoke": 70, "Zucchini": 70, "Eggplant": 70,
        "Garlic": 6, "Oregano": 2, "Kalamata Olives": 45, "Green Olives": 45,
        "Pineapple": 80, "Corn": 60, "Cilantro": 3, "Pears": 60, "Thyme": 1,
        "Chipotle Sauce": 50, "Barbecue Sauce": 70, "Alfredo Sauce": 80,
        "Pesto Sauce": 50, "Thai Sweet Chilli Sauce": 45
    }
    shelf_life_days = {
        "Sliced Ham": 5, "Pepperoni": 14, "Bacon": 7, "Calabrese Salami": 14,
        "Capocollo": 10, "Chicken": 5, "Barbecued Chicken": 5,
        "Prosciutto di San Daniele": 10, "Prosciutto": 10,
        "Beef Chuck Roast": 5, "Italian Sausage": 7, "Chorizo Sausage": 10,
        "Soppressata Salami": 14, "Anchovies": 30, "Genoa Salami": 14,
        "Coarse Sicilian Salami": 14, "Luganega Sausage": 7, "Pancetta": 10,
        "Nduja Salami": 10, "Mozzarella Cheese": 7, "Provolone Cheese": 14,
        "Smoked Gouda Cheese": 14, "Romano Cheese": 30, "Blue Cheese": 10,
        "Feta Cheese": 7, "Asiago Cheese": 14, "Goat Cheese": 7,
        "Ricotta Cheese": 7, "Gorgonzola Piccante Cheese": 10,
        "Parmigiano Reggiano Cheese": 30, "Fontina Cheese": 14,
        "Gouda Cheese": 14, "Brie Carre Cheese": 7, "Mushrooms": 3,
        "Red Onions": 14, "Onions": 14, "Caramelized Onions": 5,
        "Red Peppers": 5, "Green Peppers": 5, "Friggitello Peppers": 5,
        "Jalapeno Peppers": 7, "Peperoncini verdi": 7, "Tomatoes": 5,
        "Plum Tomatoes": 5, "Sun-dried Tomatoes": 30, "Spinach": 3,
        "Arugula": 3, "Artichoke": 5, "Zucchini": 5, "Eggplant": 5,
        "Garlic": 30, "Oregano": 30, "Kalamata Olives": 30,
        "Green Olives": 30, "Pineapple": 5, "Corn": 7, "Cilantro": 3,
        "Pears": 7, "Thyme": 30, "Chipotle Sauce": 30, "Barbecue Sauce": 30,
        "Alfredo Sauce": 7, "Pesto Sauce": 7, "Thai Sweet Chilli Sauce": 30
    }
    ingredient_masses_kg = {k: v / 1000 for k, v in ingredient_masses_grams.items()}
    return ingredient_masses_kg, shelf_life_days

ingredient_masses_kg, shelf_life_days = get_ingredient_data()

@st.cache_data
def get_recipes(df):
    df_copy = df.copy()
    #Split comma-separated ingredients into lists, removing extra spaces
    df_copy['ingredient_list'] = df_copy['pizza_ingredients'].str.split(',').apply(
        lambda lst: [ing.strip() for ing in lst]  # lambda = anonymous function to clean each ingredient
    )
    #Create dictionary mapping pizza names to their ingredient lists
    # drop_duplicates removes repeated pizza recipes, set_index makes pizza_name the key
    recipes_dict = df_copy[['pizza_name','ingredient_list']].drop_duplicates('pizza_name') \
        .set_index('pizza_name')['ingredient_list'].to_dict()
    return recipes_dict

recipes_dict = get_recipes(df)

# --- Daily ingredient usage (for Ingredient page) ---
@st.cache_data
def calculate_daily_usage(df, recipes_dict, ingredient_masses_kg):
    #Complex data transformation to calculate daily pizza demand
    # set_index makes order_date the index, groupby groups by pizza type
    # resample('D') groups by day, unstack pivots pizza names to columns
    daily_demand_pizzas = df.set_index('order_date').groupby('pizza_name')['quantity'] \
        .resample('D').sum().unstack(level=0).fillna(0)

    #Set comprehension to get all unique ingredients from all recipes
    all_ingredients = sorted({ing for ings in recipes_dict.values() for ing in ings})
    #Create empty DataFrame with dates as rows and ingredients as columns
    ingredient_usage = pd.DataFrame(index=daily_demand_pizzas.index,
                                    columns=all_ingredients, dtype=float).fillna(0.0)

    #Double loop to calculate ingredient usage for each day
    for date in daily_demand_pizzas.index:  # Loop through each day
        for pizza_name in daily_demand_pizzas.columns:  # Loop through each pizza type
            qty = daily_demand_pizzas.loc[date, pizza_name]  # How many of this pizza on this day
            if qty > 0 and pizza_name in recipes_dict:
                #For each ingredient in this pizza's recipe
                for ing in recipes_dict[pizza_name]:
                    if ing in ingredient_masses_kg:
                        #Add ingredient weight (quantity × weight per pizza)
                        ingredient_usage.loc[date, ing] += qty * ingredient_masses_kg[ing]
    return ingredient_usage

daily_ingredient_usage = calculate_daily_usage(df, recipes_dict, ingredient_masses_kg)

# --- Forecasting Functions ---
@st.cache_data
def prepare_forecasting_data(df):
    """Prepare daily aggregated data for forecasting"""
    #Group all orders by date and calculate daily totals
    daily_sales = df.groupby('order_date').agg({
        'quantity': 'sum',        # Total pizzas sold per day
        'total_price': 'sum',     # Total revenue per day
        'order_id': 'nunique'     # Number of unique orders per day
    }).reset_index()

    #Rename columns to be more descriptive
    daily_sales.columns = ['date', 'pizzas_sold', 'revenue', 'orders_count']
    daily_sales['avg_order_value'] = daily_sales['revenue'] / daily_sales['orders_count']

    #Feature engineering - create time-based features for ML model
    daily_sales['weekday'] = daily_sales['date'].dt.day_name()  # Monday, Tuesday, etc.
    daily_sales['day_of_year'] = daily_sales['date'].dt.dayofyear  # 1-365
    daily_sales['week_of_year'] = daily_sales['date'].dt.isocalendar().week  # 1-52
    daily_sales['month'] = daily_sales['date'].dt.month  # 1-12
    daily_sales['is_weekend'] = daily_sales['date'].dt.weekday >= 5  # True/False

    return daily_sales

@st.cache_data
def create_features_for_forecasting(data):
    """Create numerical features for forecasting models"""
    features_df = data.copy()

    #Convert text weekdays to numbers for machine learning
    weekday_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                      'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    features_df['weekday_num'] = features_df['weekday'].map(weekday_mapping)

    #Cyclical encoding - converts circular patterns (days, weeks) to sin/cos
    # This helps ML models understand that Dec 31 is close to Jan 1
    features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['week_sin'] = np.sin(2 * np.pi * features_df['weekday_num'] / 7)
    features_df['week_cos'] = np.cos(2 * np.pi * features_df['weekday_num'] / 7)

    return features_df

@st.cache_data
def train_forecasting_models(df):
    """Train the forecasting models"""
    daily_sales = prepare_forecasting_data(df)
    forecast_data = create_features_for_forecasting(daily_sales)

    #Define which columns to use as input features for the ML model
    feature_cols = ['day_of_year', 'weekday_num', 'month', 'is_weekend',
                    'day_sin', 'day_cos', 'week_sin', 'week_cos']

    #Split data into training and testing sets (80%/20% split)
    split_idx = int(len(forecast_data) * 0.8)  # 80% for training
    train_data = forecast_data.iloc[:split_idx]  # First 80% of data

    X_train = train_data[feature_cols]
    y_train_pizzas = train_data['pizzas_sold']
    y_train_revenue = train_data['revenue']

    # Train models
    pizza_model = LinearRegression()
    pizza_model.fit(X_train, y_train_pizzas)

    revenue_model = LinearRegression()
    revenue_model.fit(X_train, y_train_revenue)

    # Get pizza mix for ingredient predictions
    pizza_mix = df.groupby('pizza_name')['quantity'].sum() / df['quantity'].sum()

    return pizza_model, revenue_model, pizza_mix, feature_cols

def predict_future_demand(start_date, end_date, pizza_model, revenue_model, feature_cols):
    """Predict pizza demand and revenue for future dates"""
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    future_df = pd.DataFrame({
        'date': future_dates,
        'day_of_year': future_dates.dayofyear,
        'weekday': future_dates.day_name(),
        'month': future_dates.month,
        'is_weekend': future_dates.weekday >= 5
    })

    future_df = create_features_for_forecasting(future_df)
    X_future = future_df[feature_cols]

    future_df['predicted_pizzas'] = pizza_model.predict(X_future)
    future_df['predicted_revenue'] = revenue_model.predict(X_future)

    return future_df[['date', 'weekday', 'predicted_pizzas', 'predicted_revenue']]

def predict_ingredient_demand(predicted_pizzas, pizza_mix_dict, recipes_dict):
    """Convert predicted pizza demand to ingredient demand"""
    ingredient_demand = defaultdict(int)

    for pizza_name, proportion in pizza_mix_dict.items():
        if pizza_name in recipes_dict:
            expected_qty = predicted_pizzas * proportion
            for ingredient in recipes_dict[pizza_name]:
                ingredient_demand[ingredient] += expected_qty

    return dict(ingredient_demand)

# Initialize forecasting models
pizza_model, revenue_model, pizza_mix, feature_cols = train_forecasting_models(df)

# --- Inventory Simulation Functions ---
@st.cache_data
def calculate_inventory_parameters(daily_ingredient_usage, ingredient_masses_kg, shelf_life_days):
    """Calculate inventory parameters for simulation"""
    first_two_weeks = daily_ingredient_usage[
        (daily_ingredient_usage.index >= '2015-01-01') &
        (daily_ingredient_usage.index < '2015-01-15')
    ]

    first_three_months = daily_ingredient_usage[
        (daily_ingredient_usage.index >= '2015-01-01') &
        (daily_ingredient_usage.index < '2015-04-01')
    ]

    inventory_parameters = []

    for ingredient in daily_ingredient_usage.columns:
        # Starting inventory: total demand in first 2 weeks * 2.2
        starting_inventory = first_two_weeks[ingredient].sum() * 2.2

        # Restock target: max daily demand in first 3 months * 0.95
        max_daily_demand = first_three_months[ingredient].max()
        restock_target = max_daily_demand * 0.95

        # Alarm threshold: max daily demand * 0.5
        alarm_threshold = max_daily_demand * 0.5

        # Expiry time
        expiry_time = shelf_life_days.get(ingredient, 7)

        inventory_parameters.append({
            'ingredient': ingredient,
            'starting_inventory_kg': starting_inventory,
            'restock_target_kg': restock_target,
            'alarm_threshold_kg': alarm_threshold,
            'expiry_days': expiry_time,
            'ingredient_per_pizza_kg': ingredient_masses_kg.get(ingredient, 0)
        })

    return pd.DataFrame(inventory_parameters).set_index('ingredient')

def simulate_inventory_fixed(df, inventory_params, recipes_dict, ingredient_masses_kg,
                      start_date='2015-01-01', end_date='2015-04-01',
                      restock_interval_days=7, alarm_threshold=3):
    """
    Enhanced inventory simulation with FIXED quantity-based demand loss calculation
    """
    from collections import defaultdict

    #Extract parameters from DataFrame and convert to dictionaries for easy lookup
    alarm_threshold_kg = {k: float(v) for k, v in inventory_params['alarm_threshold_kg'].items()}
    restock_target_kg = {k: float(v) for k, v in inventory_params['restock_target_kg'].items()}
    expiry_days = {k: int(v) for k, v in inventory_params['expiry_days'].items()}

    # Setup
    df_sim = df.copy()
    df_sim["order_date"] = pd.to_datetime(df_sim["order_date"])
    df_sim["date"] = df_sim["order_date"].dt.date
    days = pd.date_range(start_date, end_date, freq="D")
    start_day = pd.to_datetime(start_date).date()

    # Initialize inventory
    inventory = {}
    for ingredient in inventory_params.index:
        starting_amount = float(inventory_params.loc[ingredient, 'starting_inventory_kg'])
        expiry_date = start_day + timedelta(days=int(inventory_params.loc[ingredient, 'expiry_days']))
        inventory[ingredient] = [{'quantity': starting_amount, 'expiry_date': expiry_date}]

    # Tracking variables
    lost_demand = 0
    waste_kg = defaultdict(float)
    manual_alarms = 0
    restock_events = 0
    weekly_alarms = 0
    current_restock_interval = restock_interval_days
    last_restock_date = start_day
    daily_stats = []

    # Helper functions
    def get_current_stock(ingredient):
        return sum(batch['quantity'] for batch in inventory.get(ingredient, []))

    def remove_expired(today):
        nonlocal waste_kg
        for ingredient in inventory:
            new_batches = []
            for batch in inventory[ingredient]:
                if batch['expiry_date'] > today:
                    new_batches.append(batch)
                else:
                    waste_kg[ingredient] += batch['quantity']
            inventory[ingredient] = new_batches

    def consume_fefo(ingredient, needed_amount):
        remaining_needed = needed_amount
        inventory[ingredient].sort(key=lambda x: x['expiry_date'])

        for batch in inventory[ingredient][:]:
            if remaining_needed <= 0:
                break
            taken = min(batch['quantity'], remaining_needed)
            batch['quantity'] -= taken
            remaining_needed -= taken
            if batch['quantity'] == 0:
                inventory[ingredient].remove(batch)

        return needed_amount - remaining_needed

    # Main simulation loop
    for day_dt in days:
        day = day_dt.date()
        day_lost_demand = 0
        day_waste = 0

        remove_expired(day)
        day_waste = sum(waste_kg.values()) - sum([stat.get('cumulative_waste', 0) for stat in daily_stats[-1:]])

        # Process orders with FIXED quantity handling
        day_orders = df_sim[df_sim["date"] == day]
        for _, order in day_orders.iterrows():
            pizza = order['pizza_name']
            qty = int(order['quantity'])

            if pizza in recipes_dict:
                needed = {}
                can_fulfill = True

                for ingredient in recipes_dict[pizza]:
                    if ingredient in ingredient_masses_kg:
                        needed[ingredient] = qty * ingredient_masses_kg[ingredient]
                        if get_current_stock(ingredient) < needed[ingredient]:
                            can_fulfill = False
                            break

                if can_fulfill:
                    for ingredient, amt_needed in needed.items():
                        consumed = consume_fefo(ingredient, amt_needed)
                        if consumed < amt_needed:
                            can_fulfill = False
                            break

                # FIXED: Count actual pizza quantities lost
                if not can_fulfill:
                    lost_demand += qty
                    day_lost_demand += qty

        # Check alarms
        daily_alarms = 0
        for ingredient in inventory:
            current_stock = get_current_stock(ingredient)
            threshold = alarm_threshold_kg[ingredient]
            if current_stock < threshold:
                daily_alarms += 1

        manual_alarms += daily_alarms
        weekly_alarms += daily_alarms

        # Restock logic
        should_restock = False
        if (day - start_day).days % 7 == 0 and day != start_day:
            should_restock = True
        if (day - last_restock_date).days >= current_restock_interval:
            should_restock = True

        if should_restock:
            for ingredient in inventory:
                current_stock = get_current_stock(ingredient)
                target = restock_target_kg[ingredient]
                order_quantity = target - current_stock
                if order_quantity > 0:
                    inventory[ingredient].append({
                        'quantity': order_quantity,
                        'expiry_date': day + timedelta(days=expiry_days[ingredient])
                    })
            restock_events += 1
            last_restock_date = day

        # Weekly interval adjustment
        if day.weekday() == 6:  # Sunday
            if weekly_alarms >= alarm_threshold:
                current_restock_interval = max(3, current_restock_interval - 1)
            weekly_alarms = 0

        # Store daily stats
        daily_stats.append({
            'date': day,
            'lost_demand': day_lost_demand,
            'daily_waste': day_waste,
            'cumulative_waste': sum(waste_kg.values()),
            'total_alarms': daily_alarms,
            'restock_interval': current_restock_interval
        })

    return {
        'lost_demand': lost_demand,
        'waste_kg': dict(waste_kg),
        'waste_total_kg': sum(waste_kg.values()),
        'manual_alarms': manual_alarms,
        'restock_events': restock_events,
        'final_restock_interval': current_restock_interval,
        'daily_stats': daily_stats
    }

# Initialize inventory parameters
inventory_params = calculate_inventory_parameters(daily_ingredient_usage, ingredient_masses_kg, shelf_life_days)

# --- Filter for selected period ---
filtered_df = df[(df['order_date'].dt.date >= start_date) &
                 (df['order_date'].dt.date <= end_date)]

# --- KPI Overview ---
if page == "📊 KPI Overview":
    st.header("📊 Business Overview")


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = pd.to_numeric(filtered_df['total_price'], errors='coerce').sum()
        orders = len(filtered_df)
        st.metric("💰 Total Revenue", f"€{total_revenue:,.2f}")
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:4px 12px;
                border-radius:16px;
                background-color:#06402B;
                color:#50C878;
                font-weight:bold;
                font-size:0.9rem;">
                {orders:,} orders
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        total_pizzas = filtered_df['quantity'].sum()
        varieties = filtered_df['pizza_name'].nunique()
        st.metric("🍕 Total Pizzas Sold", f"{total_pizzas:,}")
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:4px 12px;
                border-radius:16px;
                background-color:#06402B;
                color:#50C878;
                font-weight:bold;
                font-size:0.9rem;">
                {varieties:,} varieties
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        avg_order_value = pd.to_numeric(filtered_df['total_price'], errors='coerce').mean()
        st.metric("📋 Average Order Value", f"€{avg_order_value:.2f}")

    with col4:
        #Set comprehension to count unique ingredients across all recipes
        # The nested for loops flatten all ingredient lists into one set (removes duplicates)
        unique_ingredients = len({ing.strip() for recipe in recipes_dict.values() for ing in recipe})
        st.metric("🥕 Ingredients Managed", f"{unique_ingredients}")
        


    #Create two-column layout for side-by-side charts
    col1, col2 = st.columns(2)
    with col1:
        #Group by date and sum revenue for time series chart
        daily_revenue = filtered_df.groupby(filtered_df['order_date'].dt.date)['total_price'].sum()
        #Create interactive line chart with Plotly
        fig_revenue = px.line(x=daily_revenue.index, y=daily_revenue.values,
                              title="📈 Daily Revenue Trend",
                              labels={'x': 'Date', 'y': 'Revenue (€)'})
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        #Group by pizza name, sum quantities, get top 10
        top_pizzas = filtered_df.groupby('pizza_name')['quantity'].sum().nlargest(10)
        #Create horizontal bar chart (orientation='h' makes it horizontal)
        fig_pizzas = px.bar(x=top_pizzas.values, y=top_pizzas.index, orientation='h',
                            title="🏆 Top 10 Most Popular Pizzas",
                            labels={'x': 'Quantity Sold', 'y': 'Pizza Name'})
        fig_pizzas.update_layout(height=400)
        st.plotly_chart(fig_pizzas, use_container_width=True)

# --- Inventory Tracking page ---
elif page == "📦 Inventory Tracking":
    st.header("📦 Advanced Inventory Management Simulation")

    # Inventory tracking type selection
    tracking_type = st.selectbox(
        "Select tracking analysis:",
        ["📊 Ingredient Usage Overview", "🎮 Inventory Simulation"]
    )

    if tracking_type == "📊 Ingredient Usage Overview":
        st.subheader("📊 Ingredient Usage & Risk Analysis")

        #Debug section to check for missing ingredients
        with st.expander("🔍 Debug: Ingredient Matching", expanded=False):
            # Get all ingredients from recipes
            all_recipe_ingredients = sorted({ing for ings in recipes_dict.values() for ing in ings})

            # Get all ingredients from our predefined list
            predefined_ingredients = sorted(ingredient_masses_kg.keys())

            # Find ingredients in recipes but not in our predefined list
            missing_from_predefined = set(all_recipe_ingredients) - set(predefined_ingredients)

            # Find ingredients in predefined list but not used in any recipe
            unused_predefined = set(predefined_ingredients) - set(all_recipe_ingredients)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Ingredients in recipes but missing from predefined list:**")
                if missing_from_predefined:
                    for ing in sorted(missing_from_predefined):
                        st.write(f"• {ing}")
                else:
                    st.write("✅ All recipe ingredients are predefined")

            with col2:
                st.write("**Predefined ingredients not used in any recipe:**")
                if unused_predefined:
                    for ing in sorted(unused_predefined):
                        st.write(f"• {ing}")
                else:
                    st.write("✅ All predefined ingredients are used")
        
        period_usage = daily_ingredient_usage[
            (daily_ingredient_usage.index.date >= start_date) &
            (daily_ingredient_usage.index.date <= end_date)
        ]
        total_usage = period_usage.sum().sort_values(ascending=False).head(15)

        col1, col2 = st.columns(2)
        with col1:
            fig_ingredients = px.bar(x=total_usage.index, y=total_usage.values,
                                     title="🥕 Top 15 Ingredients by Usage",
                                     labels={'x': 'Ingredient', 'y': 'Total Usage (kg)'})
            fig_ingredients.update_xaxes(tickangle=45)
            st.plotly_chart(fig_ingredients, use_container_width=True)

        with col2:
            shelf_life_df = pd.DataFrame([
                {'Ingredient': ing,
                 'Shelf_Life_Days': shelf_life_days.get(ing, 7),
                 'Usage_kg': total_usage.get(ing, 0)}
                for ing in total_usage.index
            ])
            shelf_life_df['Risk_Level'] = pd.cut(
                shelf_life_df['Shelf_Life_Days'],
                bins=[0, 3, 7, 14, float('inf')],
                labels=['Very High', 'High', 'Medium', 'Low']
            )
            fig_shelf = px.scatter(
                shelf_life_df, x='Shelf_Life_Days', y='Usage_kg',
                color='Risk_Level', size='Usage_kg', hover_data=['Ingredient'],
                title="📊 Ingredient Risk (Usage vs Shelf Life)",
                color_discrete_map={'Very High': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
            )
            st.plotly_chart(fig_shelf, use_container_width=True)

        # Show inventory parameters table
        st.subheader("📋 Calculated Inventory Parameters")

        display_params = inventory_params.copy()
        display_params['starting_inventory_kg'] = display_params['starting_inventory_kg'].round(2)
        display_params['restock_target_kg'] = display_params['restock_target_kg'].round(2)
        display_params['alarm_threshold_kg'] = display_params['alarm_threshold_kg'].round(2)

        st.dataframe(
            display_params.head(10),
            use_container_width=True,
            column_config={
                "starting_inventory_kg": st.column_config.NumberColumn("Starting Inventory (kg)", format="%.2f"),
                "restock_target_kg": st.column_config.NumberColumn("Restock Target (kg)", format="%.2f"),
                "alarm_threshold_kg": st.column_config.NumberColumn("Alarm Threshold (kg)", format="%.2f"),
                "expiry_days": st.column_config.NumberColumn("Expiry Days", format="%d"),
                "ingredient_per_pizza_kg": st.column_config.NumberColumn("Per Pizza (kg)", format="%.3f")
            }
        )

    elif tracking_type == "🎮 Inventory Simulation":
        st.subheader("🎮 Interactive Inventory Simulation")

        st.write("Run inventory simulations with different parameters to see their impact on waste and lost demand.")

        # Simulation parameters
        col1, col2, col3 = st.columns(3)

        with col1:
            sim_start_date = st.date_input(
                "Simulation Start Date:",
                value=datetime(2015, 1, 1).date(),
                min_value=datetime(2015, 1, 1).date(),
                max_value=datetime(2015, 12, 31).date()
            )

        with col2:
            sim_end_date = st.date_input(
                "Simulation End Date:",
                value=datetime(2015, 3, 31).date(),
                min_value=sim_start_date,
                max_value=datetime(2015, 12, 31).date()
            )

        with col3:
            sim_duration = (sim_end_date - sim_start_date).days
            st.metric("Simulation Days", sim_duration)

        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            # Display optimal scenarios first
            st.subheader("🎯 Optimized Parameter Scenarios")
            st.markdown("""
            **📉 SCENARIO 1 - MINIMIZE LOST DEMAND:**
            - Parameters: init=1.01, min=0.28, max=2.99

            **♻️ SCENARIO 2 - MINIMIZE WASTE:**
            - Parameters: init=1.01, min=0.15, max=1.75
            """)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📉 Use Lost Demand Optimal", key="use_lost_demand_optimal"):
                    st.session_state.init_factor = 1.01
                    st.session_state.min_factor = 0.28
                    st.session_state.max_factor = 2.99

            with col2:
                if st.button("♻️ Use Waste Optimal", key="use_waste_optimal"):
                    st.session_state.init_factor = 1.01
                    st.session_state.min_factor = 0.15
                    st.session_state.max_factor = 1.75

            with col3:
                if st.button("🔄 Reset to Default", key="reset_to_default"):
                    st.session_state.init_factor = 2.2
                    st.session_state.min_factor = 0.5
                    st.session_state.max_factor = 0.95

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                restock_interval = st.slider(
                    "Restock Interval (days):",
                    min_value=3, max_value=14, value=7,
                    help="How often to restock ingredients"
                )

                alarm_threshold = st.slider(
                    "Weekly Alarm Threshold:",
                    min_value=1, max_value=10, value=3,
                    help="Number of weekly alarms needed to reduce restock interval"
                )

            with col2:
                parameter_multipliers = st.checkbox("Adjust Parameter Multipliers", value=False)

                if parameter_multipliers:
                    # Use session state values if they exist, otherwise use defaults
                    init_default = st.session_state.get('init_factor', 2.2)
                    min_default = st.session_state.get('min_factor', 0.5)
                    max_default = st.session_state.get('max_factor', 0.95)

                    init_factor = st.slider("Starting Inventory Factor:", 1.0, 3.0, init_default, 0.01)
                    min_factor = st.slider("Alarm Threshold Factor:", 0.1, 1.0, min_default, 0.01)
                    max_factor = st.slider("Restock Target Factor:", 0.5, 3.0, max_default, 0.01)
                else:
                    init_factor, min_factor, max_factor = 2.2, 0.5, 0.95

        if st.button("🚀 Run Inventory Simulation", key="run_simulation"):
            with st.spinner("Running inventory simulation..."):

                # Adjust parameters if needed
                if parameter_multipliers:
                    adjusted_params = inventory_params.copy()

                    # Recalculate with new factors
                    first_two_weeks = daily_ingredient_usage[
                        (daily_ingredient_usage.index >= '2015-01-01') &
                        (daily_ingredient_usage.index < '2015-01-15')
                    ]
                    first_three_months = daily_ingredient_usage[
                        (daily_ingredient_usage.index >= '2015-01-01') &
                        (daily_ingredient_usage.index < '2015-04-01')
                    ]

                    for ingredient in adjusted_params.index:
                        starting_inv = first_two_weeks[ingredient].sum() * init_factor
                        max_daily = first_three_months[ingredient].max()

                        adjusted_params.loc[ingredient, 'starting_inventory_kg'] = starting_inv
                        adjusted_params.loc[ingredient, 'restock_target_kg'] = max_daily * max_factor
                        adjusted_params.loc[ingredient, 'alarm_threshold_kg'] = max_daily * min_factor

                    simulation_params = adjusted_params
                else:
                    simulation_params = inventory_params

                # Run simulation
                result = simulate_inventory_fixed(
                    df=df,
                    inventory_params=simulation_params,
                    recipes_dict=recipes_dict,
                    ingredient_masses_kg=ingredient_masses_kg,
                    start_date=sim_start_date.strftime('%Y-%m-%d'),
                    end_date=sim_end_date.strftime('%Y-%m-%d'),
                    restock_interval_days=restock_interval,
                    alarm_threshold=alarm_threshold
                )

                # Display results
                st.success("✅ Simulation completed!")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🍕 Lost Demand", f"{result['lost_demand']:,} pizzas")

                with col2:
                    st.metric("♻️ Total Waste", f"{result['waste_total_kg']:.1f} kg")

                with col3:
                    st.metric("🚨 Total Alarms", f"{result['manual_alarms']:,}")

                with col4:
                    st.metric("📦 Restock Events", f"{result['restock_events']:,}")

                # Calculate performance percentages
                total_orders = len(df[(df['order_date'] >= pd.to_datetime(sim_start_date)) &
                                     (df['order_date'] <= pd.to_datetime(sim_end_date))])
                period_usage = daily_ingredient_usage[
                    (daily_ingredient_usage.index.date >= sim_start_date) &
                    (daily_ingredient_usage.index.date <= sim_end_date)
                ].sum().sum()

                if total_orders > 0 and period_usage > 0:
                    lost_demand_pct = (result['lost_demand'] / total_orders) * 100
                    waste_pct = (result['waste_total_kg'] / period_usage) * 100

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Lost Demand Rate", f"{lost_demand_pct:.1f}%")
                    with col2:
                        st.metric("📊 Waste Rate", f"{waste_pct:.1f}%")

                # Daily performance charts
                if result['daily_stats']:
                    daily_df = pd.DataFrame(result['daily_stats'])
                    daily_df['date'] = pd.to_datetime(daily_df['date'])

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_lost = px.line(
                            daily_df, x='date', y='lost_demand',
                            title="📉 Daily Lost Demand",
                            labels={'lost_demand': 'Lost Pizzas', 'date': 'Date'}
                        )
                        st.plotly_chart(fig_lost, use_container_width=True)

                    with col2:
                        fig_waste = px.line(
                            daily_df, x='date', y='cumulative_waste',
                            title="♻️ Cumulative Waste",
                            labels={'cumulative_waste': 'Waste (kg)', 'date': 'Date'}
                        )
                        st.plotly_chart(fig_waste, use_container_width=True)

                # Waste breakdown by ingredient
                if result['waste_kg']:
                    st.subheader("🗂️ Waste Breakdown by Ingredient")

                    waste_df = pd.DataFrame([
                        {'Ingredient': ing, 'Waste_kg': waste}
                        for ing, waste in result['waste_kg'].items()
                        if waste > 0
                    ]).sort_values('Waste_kg', ascending=False)

                    if len(waste_df) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.dataframe(waste_df.head(10), use_container_width=True, hide_index=True)

                        with col2:
                            fig_waste_breakdown = px.bar(
                                waste_df.head(10), x='Waste_kg', y='Ingredient',
                                orientation='h',
                                title="Top 10 Ingredients by Waste"
                            )
                            st.plotly_chart(fig_waste_breakdown, use_container_width=True)


    elif tracking_type == "🎮 Inventory Simulation":
        st.subheader("🎮 Interactive Inventory Simulation")

        st.write("Run inventory simulations with different parameters to see their impact on waste and lost demand.")

        # Simulation parameters
        col1, col2, col3 = st.columns(3)

        with col1:
            sim_start_date = st.date_input(
                "Simulation Start Date:",
                value=datetime(2015, 1, 1).date(),
                min_value=datetime(2015, 1, 1).date(),
                max_value=datetime(2015, 12, 31).date()
            )

        with col2:
            sim_end_date = st.date_input(
                "Simulation End Date:",
                value=datetime(2015, 3, 31).date(),
                min_value=sim_start_date,
                max_value=datetime(2015, 12, 31).date()
            )

        with col3:
            sim_duration = (sim_end_date - sim_start_date).days
            st.metric("Simulation Days", sim_duration)

        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            # Display optimal scenarios first
            st.subheader("🎯 Optimized Parameter Scenarios")
            st.markdown("""
            **📉 SCENARIO 1 - MINIMIZE LOST DEMAND:**
            - Parameters: init=1.01, min=0.28, max=2.99

            **♻️ SCENARIO 2 - MINIMIZE WASTE:**
            - Parameters: init=1.01, min=0.15, max=1.75
            """)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📉 Use Lost Demand Optimal", key="use_lost_demand_optimal"):
                    st.session_state.init_factor = 1.01
                    st.session_state.min_factor = 0.28
                    st.session_state.max_factor = 2.99

            with col2:
                if st.button("♻️ Use Waste Optimal", key="use_waste_optimal"):
                    st.session_state.init_factor = 1.01
                    st.session_state.min_factor = 0.15
                    st.session_state.max_factor = 1.75

            with col3:
                if st.button("🔄 Reset to Default", key="reset_to_default"):
                    st.session_state.init_factor = 2.2
                    st.session_state.min_factor = 0.5
                    st.session_state.max_factor = 0.95

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                restock_interval = st.slider(
                    "Restock Interval (days):",
                    min_value=3, max_value=14, value=7,
                    help="How often to restock ingredients"
                )

                alarm_threshold = st.slider(
                    "Weekly Alarm Threshold:",
                    min_value=1, max_value=10, value=3,
                    help="Number of weekly alarms needed to reduce restock interval"
                )

            with col2:
                parameter_multipliers = st.checkbox("Adjust Parameter Multipliers", value=False)

                if parameter_multipliers:
                    # Use session state values if they exist, otherwise use defaults
                    init_default = st.session_state.get('init_factor', 2.2)
                    min_default = st.session_state.get('min_factor', 0.5)
                    max_default = st.session_state.get('max_factor', 0.95)

                    init_factor = st.slider("Starting Inventory Factor:", 1.0, 3.0, init_default, 0.01)
                    min_factor = st.slider("Alarm Threshold Factor:", 0.1, 1.0, min_default, 0.01)
                    max_factor = st.slider("Restock Target Factor:", 0.5, 3.0, max_default, 0.01)
                else:
                    init_factor, min_factor, max_factor = 2.2, 0.5, 0.95

        if st.button("🚀 Run Inventory Simulation", key="run_simulation"):
            with st.spinner("Running inventory simulation..."):

                # Adjust parameters if needed
                if parameter_multipliers:
                    adjusted_params = inventory_params.copy()

                    # Recalculate with new factors
                    first_two_weeks = daily_ingredient_usage[
                        (daily_ingredient_usage.index >= '2015-01-01') &
                        (daily_ingredient_usage.index < '2015-01-15')
                    ]
                    first_three_months = daily_ingredient_usage[
                        (daily_ingredient_usage.index >= '2015-01-01') &
                        (daily_ingredient_usage.index < '2015-04-01')
                    ]

                    for ingredient in adjusted_params.index:
                        starting_inv = first_two_weeks[ingredient].sum() * init_factor
                        max_daily = first_three_months[ingredient].max()

                        adjusted_params.loc[ingredient, 'starting_inventory_kg'] = starting_inv
                        adjusted_params.loc[ingredient, 'restock_target_kg'] = max_daily * max_factor
                        adjusted_params.loc[ingredient, 'alarm_threshold_kg'] = max_daily * min_factor

                    simulation_params = adjusted_params
                else:
                    simulation_params = inventory_params

                # Run simulation
                result = simulate_inventory_fixed(
                    df=df,
                    inventory_params=simulation_params,
                    recipes_dict=recipes_dict,
                    ingredient_masses_kg=ingredient_masses_kg,
                    start_date=sim_start_date.strftime('%Y-%m-%d'),
                    end_date=sim_end_date.strftime('%Y-%m-%d'),
                    restock_interval_days=restock_interval,
                    alarm_threshold=alarm_threshold
                )

                # Display results
                st.success("✅ Simulation completed!")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🍕 Lost Demand", f"{result['lost_demand']:,} pizzas")

                with col2:
                    st.metric("♻️ Total Waste", f"{result['waste_total_kg']:.1f} kg")

                with col3:
                    st.metric("🚨 Total Alarms", f"{result['manual_alarms']:,}")

                with col4:
                    st.metric("📦 Restock Events", f"{result['restock_events']:,}")

                # Calculate performance percentages
                total_orders = len(df[(df['order_date'] >= pd.to_datetime(sim_start_date)) &
                                     (df['order_date'] <= pd.to_datetime(sim_end_date))])
                period_usage = daily_ingredient_usage[
                    (daily_ingredient_usage.index.date >= sim_start_date) &
                    (daily_ingredient_usage.index.date <= sim_end_date)
                ].sum().sum()

                if total_orders > 0 and period_usage > 0:
                    lost_demand_pct = (result['lost_demand'] / total_orders) * 100
                    waste_pct = (result['waste_total_kg'] / period_usage) * 100

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Lost Demand Rate", f"{lost_demand_pct:.1f}%")
                    with col2:
                        st.metric("📊 Waste Rate", f"{waste_pct:.1f}%")

                # Daily performance charts
                if result['daily_stats']:
                    daily_df = pd.DataFrame(result['daily_stats'])
                    daily_df['date'] = pd.to_datetime(daily_df['date'])

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_lost = px.line(
                            daily_df, x='date', y='lost_demand',
                            title="📉 Daily Lost Demand",
                            labels={'lost_demand': 'Lost Pizzas', 'date': 'Date'}
                        )
                        st.plotly_chart(fig_lost, use_container_width=True)

                    with col2:
                        fig_waste = px.line(
                            daily_df, x='date', y='cumulative_waste',
                            title="♻️ Cumulative Waste",
                            labels={'cumulative_waste': 'Waste (kg)', 'date': 'Date'}
                        )
                        st.plotly_chart(fig_waste, use_container_width=True)

                # Waste breakdown by ingredient
                if result['waste_kg']:
                    st.subheader("🗂️ Waste Breakdown by Ingredient")

                    waste_df = pd.DataFrame([
                        {'Ingredient': ing, 'Waste_kg': waste}
                        for ing, waste in result['waste_kg'].items()
                        if waste > 0
                    ]).sort_values('Waste_kg', ascending=False)

                    if len(waste_df) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.dataframe(waste_df.head(10), use_container_width=True, hide_index=True)

                        with col2:
                            fig_waste_breakdown = px.bar(
                                waste_df.head(10), x='Waste_kg', y='Ingredient',
                                orientation='h',
                                title="Top 10 Ingredients by Waste"
                            )
                            st.plotly_chart(fig_waste_breakdown, use_container_width=True)


# --- Sales Forecast page ---
elif page == "📈 Sales Forecast":
    st.header("📈 Sales Forecasting & Demand Prediction")


    # Forecast type selection
    forecast_type = st.selectbox(
        "Select forecast type:",
        ["📅 Single Day Prediction", "📊 Multi-Day Forecast", "🥕 Ingredient Planning"]
    )

    if forecast_type == "📅 Single Day Prediction":
        st.subheader("🎯 Predict Sales for a Specific Date")

        # Date input
        st.info("💡 You can test predictions on historical dates (2015) or forecast future dates")
        target_date = st.date_input(
            "Select date for prediction:",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime(2015, 1, 1).date(),
            max_value=datetime.now().date() + timedelta(days=365)
        )

        if st.button("🔮 Generate Prediction", key="single_day"):
            # Get prediction
            prediction_df = predict_future_demand(
                target_date, target_date, pizza_model, revenue_model, feature_cols
            )

            if len(prediction_df) > 0:
                day_data = prediction_df.iloc[0]
                predicted_pizzas = day_data['predicted_pizzas']
                predicted_revenue = day_data['predicted_revenue']
                weekday = day_data['weekday']

                # Display main metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🍕 Predicted Pizzas", f"{predicted_pizzas:.0f}")

                with col2:
                    st.metric("💰 Expected Revenue", f"€{predicted_revenue:.2f}")

                with col3:
                    st.metric("📅 Day of Week", weekday)

                with col4:
                    # Calculate performance vs average
                    daily_sales = prepare_forecasting_data(df)
                    avg_daily = daily_sales['pizzas_sold'].mean()
                    performance = ((predicted_pizzas / avg_daily - 1) * 100)
                    st.metric("📈 vs Average", f"{performance:+.1f}%")

                # Pizza type predictions
                st.subheader("🍕 Pizza Types to Prepare")

                pizza_types_prediction = {}
                for pizza_name, proportion in pizza_mix.items():
                    expected_qty = predicted_pizzas * proportion
                    if expected_qty >= 0.5:
                        pizza_types_prediction[pizza_name] = expected_qty

                # Sort and display top pizzas
                sorted_pizzas = sorted(pizza_types_prediction.items(), key=lambda x: x[1], reverse=True)

                pizza_df = pd.DataFrame(sorted_pizzas[:10], columns=['Pizza Type', 'Expected Quantity'])
                pizza_df['Expected Quantity'] = pizza_df['Expected Quantity'].round(0).astype(int)

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(pizza_df, use_container_width=True, hide_index=True)

                with col2:
                    fig_pizza_pred = px.bar(
                        pizza_df, x='Expected Quantity', y='Pizza Type',
                        orientation='h',
                        title="Top 10 Pizza Predictions"
                    )
                    fig_pizza_pred.update_layout(height=400)
                    st.plotly_chart(fig_pizza_pred, use_container_width=True)

                # Ingredient requirements
                st.subheader("🥕 Ingredient Requirements")

                ingredient_requirements = predict_ingredient_demand(
                    predicted_pizzas, pizza_mix.to_dict(), recipes_dict
                )

                # Sort ingredients by quantity
                sorted_ingredients = sorted(ingredient_requirements.items(), key=lambda x: x[1], reverse=True)

                # Display top 20 ingredients
                ingredient_df = pd.DataFrame(sorted_ingredients[:20], columns=['Ingredient', 'Required Units'])
                ingredient_df['Required Units'] = ingredient_df['Required Units'].round(1)

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(ingredient_df, use_container_width=True, hide_index=True)

                with col2:
                    fig_ingredients = px.bar(
                        ingredient_df.head(15), x='Required Units', y='Ingredient',
                        orientation='h',
                        title="Top 15 Ingredient Requirements"
                    )
                    fig_ingredients.update_layout(height=500)
                    st.plotly_chart(fig_ingredients, use_container_width=True)

                # Recommendations
                st.subheader("🎯 Preparation Recommendations")

                recommendations = []
                if weekday in ['Saturday', 'Sunday']:
                    recommendations.append("📅 Weekend day - expect higher demand, consider extra staffing")
                else:
                    recommendations.append("📅 Weekday - standard staffing levels should suffice")

                # Revenue-based recommendations
                daily_sales = prepare_forecasting_data(df)
                avg_revenue = daily_sales['revenue'].mean()
                if predicted_revenue > avg_revenue * 1.1:
                    recommendations.append("💰 High revenue day expected - prepare for busy service")
                elif predicted_revenue < avg_revenue * 0.9:
                    recommendations.append("💰 Lower revenue day - good time for prep work or promotions")

                # Critical ingredients
                critical_ingredients = sorted_ingredients[:3]
                recommendations.append(f"🚨 CRITICAL: Ensure sufficient stock of top ingredients:")
                for ingredient, qty in critical_ingredients:
                    recommendations.append(f"- {ingredient}: {qty:.1f} units")

                for rec in recommendations:
                    st.write(f"- {rec}")

    elif forecast_type == "📊 Multi-Day Forecast":
        st.subheader("📊 Multi-Day Sales Forecast")

        st.info("💡 You can test predictions on historical dates (2015) or forecast future dates")

        col1, col2 = st.columns(2)
        with col1:
            start_forecast = st.date_input(
                "Start date:",
                value=datetime.now().date() + timedelta(days=1),
                min_value=datetime(2015, 1, 1).date(),
                max_value=datetime.now().date() + timedelta(days=365)
            )

        with col2:
            end_forecast = st.date_input(
                "End date:",
                value=datetime.now().date() + timedelta(days=7),
                min_value=start_forecast,
                max_value=datetime.now().date() + timedelta(days=365)
            )

        if st.button("📊 Generate Forecast", key="multi_day"):
            prediction_df = predict_future_demand(
                start_forecast, end_forecast, pizza_model, revenue_model, feature_cols
            )

            if len(prediction_df) > 0:
                # Summary metrics
                total_pizzas = prediction_df['predicted_pizzas'].sum()
                total_revenue = prediction_df['predicted_revenue'].sum()
                num_days = len(prediction_df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🍕 Total Pizzas", f"{total_pizzas:.0f}")

                with col2:
                    st.metric("💰 Total Revenue", f"€{total_revenue:.2f}")

                with col3:
                    st.metric("📅 Days Forecasted", num_days)

                # Charts
                col1, col2 = st.columns(2)

                with col1:
                    fig_pizzas = px.line(
                        prediction_df, x='date', y='predicted_pizzas',
                        title="📈 Daily Pizza Demand Forecast",
                        labels={'predicted_pizzas': 'Predicted Pizzas', 'date': 'Date'}
                    )
                    fig_pizzas.update_traces(mode='lines+markers')
                    st.plotly_chart(fig_pizzas, use_container_width=True)

                with col2:
                    fig_revenue = px.line(
                        prediction_df, x='date', y='predicted_revenue',
                        title="💰 Daily Revenue Forecast",
                        labels={'predicted_revenue': 'Predicted Revenue (€)', 'date': 'Date'}
                    )
                    fig_revenue.update_traces(mode='lines+markers')
                    st.plotly_chart(fig_revenue, use_container_width=True)

                # Weekday analysis
                st.subheader("📊 Forecast by Weekday")

                weekday_summary = prediction_df.groupby('weekday').agg({
                    'predicted_pizzas': ['sum', 'mean'],
                    'predicted_revenue': ['sum', 'mean']
                }).round(2)

                weekday_summary.columns = ['Total Pizzas', 'Avg Pizzas', 'Total Revenue', 'Avg Revenue']

                st.dataframe(weekday_summary, use_container_width=True)

                # Detailed forecast table
                st.subheader("📋 Detailed Daily Forecast")

                display_df = prediction_df.copy()
                display_df['predicted_pizzas'] = display_df['predicted_pizzas'].round(0).astype(int)
                display_df['predicted_revenue'] = display_df['predicted_revenue'].round(2)
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

                st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif forecast_type == "🥕 Ingredient Planning":
        st.subheader("🥕 Ingredient Planning Assistant")

        st.write("Plan ingredient purchases based on sales forecasts")

        planning_days = st.slider("Planning horizon (days):", 1, 30, 7)

        start_planning = datetime.now().date() + timedelta(days=1)
        end_planning = start_planning + timedelta(days=planning_days-1)

        if st.button("🥕 Generate Ingredient Plan", key="ingredient_plan"):
            prediction_df = predict_future_demand(
                start_planning, end_planning, pizza_model, revenue_model, feature_cols
            )

            if len(prediction_df) > 0:
                total_pizzas = prediction_df['predicted_pizzas'].sum()

                # Calculate total ingredient requirements
                total_ingredient_requirements = predict_ingredient_demand(
                    total_pizzas, pizza_mix.to_dict(), recipes_dict
                )

                # Convert to dataframe and add weights
                ingredient_plan_df = pd.DataFrame([
                    {
                        'Ingredient': ingredient,
                        'Total Units': qty,
                        'Weight (kg)': qty * ingredient_masses_kg.get(ingredient, 0),
                        'Shelf Life (days)': shelf_life_days.get(ingredient, 7)
                    }
                    for ingredient, qty in total_ingredient_requirements.items()
                    if qty >= 1
                ]).sort_values('Total Units', ascending=False)

                # Add risk categories
                ingredient_plan_df['Risk Level'] = pd.cut(
                    ingredient_plan_df['Shelf Life (days)'],
                    bins=[0, 3, 7, 14, float('inf')],
                    labels=['Very High', 'High', 'Medium', 'Low']
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("📅 Planning Period", f"{planning_days} days")
                    st.metric("🍕 Total Pizzas", f"{total_pizzas:.0f}")
                    st.metric("🥕 Ingredients", len(ingredient_plan_df))

                with col2:
                    total_weight = ingredient_plan_df['Weight (kg)'].sum()
                    high_risk = len(ingredient_plan_df[ingredient_plan_df['Risk Level'].isin(['Very High', 'High'])])
                    st.metric("⚖️ Total Weight", f"{total_weight:.1f} kg")
                    st.metric("🚨 High Risk Items", high_risk)

                # Ingredient planning table with risk colors
                st.subheader("📋 Complete Ingredient Shopping List")

                # Round numerical columns
                ingredient_plan_df['Total Units'] = ingredient_plan_df['Total Units'].round(1)
                ingredient_plan_df['Weight (kg)'] = ingredient_plan_df['Weight (kg)'].round(2)

                st.dataframe(
                    ingredient_plan_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Risk Level": st.column_config.SelectboxColumn(
                            "Risk Level",
                            help="Spoilage risk based on shelf life",
                            options=["Low", "Medium", "High", "Very High"],
                        )
                    }
                )

                # Visual analysis
                col1, col2 = st.columns(2)

                with col1:
                    fig_risk = px.scatter(
                        ingredient_plan_df,
                        x='Shelf Life (days)',
                        y='Total Units',
                        color='Risk Level',
                        size='Weight (kg)',
                        hover_data=['Ingredient'],
                        title="🚨 Ingredient Risk Analysis",
                        color_discrete_map={
                            'Very High': 'red',
                            'High': 'orange',
                            'Medium': 'yellow',
                            'Low': 'green'
                        }
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                with col2:
                    # Top ingredients by weight
                    top_weight = ingredient_plan_df.nlargest(10, 'Weight (kg)')
                    fig_weight = px.bar(
                        top_weight,
                        x='Weight (kg)',
                        y='Ingredient',
                        orientation='h',
                        title="📦 Top 10 Ingredients by Weight"
                    )
                    st.plotly_chart(fig_weight, use_container_width=True)

# --- Footer ---
st.markdown("---")

with st.expander("🚀 Technical Implementation & Learning Resources", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🛠️ Technologies Used**:
        - **Python**: Main programming language
        - **Streamlit**: Web dashboard framework
        - **Pandas**: Data manipulation and analysis
        - **Scikit-learn**: Machine learning models
        - **Plotly**: Interactive visualizations
        - **NumPy**: Numerical computations

        **📊 Data Science Concepts**:
        - **Exploratory Data Analysis (EDA)**
        - **Feature Engineering**: Creating time-based features
        - **Model Training & Validation**: 80/20 split
        - **Business Metrics**: KPI calculation
        """)

    with col2:
        st.markdown("""
        **📚 Key Learning Outcomes**:
        - Convert business problems to data problems
        - Build end-to-end data science projects
        - Create interactive dashboards
        - Apply ML to real-world scenarios
        - Understand inventory optimization
        - Simulate business scenarios

        **🔗 Next Steps for Learning**:
        - Add more sophisticated optimization algorithms
        - Integrate real-time data sources
        """)

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
  <p>🍕 Pizza Inventory Management Dashboard | TechLabs Data Science Project Winter 2025</p>
  <p><em>Demonstrating data science concepts through real-world business applications</em></p>
</div>
""", unsafe_allow_html=True)

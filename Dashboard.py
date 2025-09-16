import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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

st.markdown('<h1 class="main-header">🍕 Pizza Inventory Management Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Fokus: KPIs und Zutaten-Nutzung")

# --- Load data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Data Model - Pizza Sales.xlsx')
        df["order_date"] = pd.to_datetime(df["order_date"])
        df['pizza_ingredients'] = df['pizza_ingredients'].str.replace('慛', 'N')
        df['pizza_ingredients'] = df['pizza_ingredients'].str.replace('Artichokes', 'Artichoke')
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
    df_copy['ingredient_list'] = df_copy['pizza_ingredients'].str.split(',').apply(
        lambda lst: [ing.strip() for ing in lst]
    )
    recipes_dict = df_copy[['pizza_name','ingredient_list']].drop_duplicates('pizza_name') \
        .set_index('pizza_name')['ingredient_list'].to_dict()
    return recipes_dict

recipes_dict = get_recipes(df)

# --- Daily ingredient usage (for Ingredient page) ---
@st.cache_data
def calculate_daily_usage(df, recipes_dict, ingredient_masses_kg):
    daily_demand_pizzas = df.set_index('order_date').groupby('pizza_name')['quantity'] \
        .resample('D').sum().unstack(level=0).fillna(0)

    all_ingredients = sorted({ing for ings in recipes_dict.values() for ing in ings})
    ingredient_usage = pd.DataFrame(index=daily_demand_pizzas.index,
                                    columns=all_ingredients, dtype=float).fillna(0.0)

    for date in daily_demand_pizzas.index:
        for pizza_name in daily_demand_pizzas.columns:
            qty = daily_demand_pizzas.loc[date, pizza_name]
            if qty > 0 and pizza_name in recipes_dict:
                for ing in recipes_dict[pizza_name]:
                    if ing in ingredient_masses_kg:
                        ingredient_usage.loc[date, ing] += qty * ingredient_masses_kg[ing]
    return ingredient_usage

daily_ingredient_usage = calculate_daily_usage(df, recipes_dict, ingredient_masses_kg)

# --- Forecasting Functions ---
@st.cache_data
def prepare_forecasting_data(df):
    """Prepare daily aggregated data for forecasting"""
    daily_sales = df.groupby('order_date').agg({
        'quantity': 'sum',
        'total_price': 'sum',
        'order_id': 'nunique'
    }).reset_index()

    daily_sales.columns = ['date', 'pizzas_sold', 'revenue', 'orders_count']
    daily_sales['avg_order_value'] = daily_sales['revenue'] / daily_sales['orders_count']
    daily_sales['weekday'] = daily_sales['date'].dt.day_name()
    daily_sales['day_of_year'] = daily_sales['date'].dt.dayofyear
    daily_sales['week_of_year'] = daily_sales['date'].dt.isocalendar().week
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['is_weekend'] = daily_sales['date'].dt.weekday >= 5

    return daily_sales

@st.cache_data
def create_features_for_forecasting(data):
    """Create numerical features for forecasting models"""
    features_df = data.copy()

    weekday_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                      'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    features_df['weekday_num'] = features_df['weekday'].map(weekday_mapping)

    # Cyclical encoding for seasonal patterns
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

    feature_cols = ['day_of_year', 'weekday_num', 'month', 'is_weekend',
                    'day_sin', 'day_cos', 'week_sin', 'week_cos']

    # Split data (80% train, 20% test)
    split_idx = int(len(forecast_data) * 0.8)
    train_data = forecast_data.iloc[:split_idx]

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

    # Extract parameters with clear names
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
        unique_ingredients = len({ing.strip() for recipe in recipes_dict.values() for ing in recipe})
        st.metric("🥕 Ingredients Managed", f"{unique_ingredients}")
        


    col1, col2 = st.columns(2)
    with col1:
        daily_revenue = filtered_df.groupby(filtered_df['order_date'].dt.date)['total_price'].sum()
        fig_revenue = px.line(x=daily_revenue.index, y=daily_revenue.values,
                              title="📈 Daily Revenue Trend",
                              labels={'x': 'Date', 'y': 'Revenue (€)'})
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        top_pizzas = filtered_df.groupby('pizza_name')['quantity'].sum().nlargest(10)
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
        ["📊 Ingredient Usage Overview", "🎮 Inventory Simulation", "⚖️ Parameter Optimization", "📈 Performance Analysis"]
    )

    if tracking_type == "📊 Ingredient Usage Overview":
        st.subheader("📊 Ingredient Usage & Risk Analysis")

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
                    init_factor = st.slider("Starting Inventory Factor:", 1.0, 3.0, 2.2, 0.1)
                    min_factor = st.slider("Alarm Threshold Factor:", 0.1, 1.0, 0.5, 0.1)
                    max_factor = st.slider("Restock Target Factor:", 0.5, 2.0, 0.95, 0.05)
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

    elif tracking_type == "⚖️ Parameter Optimization":
        st.subheader("⚖️ Inventory Parameter Optimization")

        st.write("Find optimal inventory parameters to minimize waste and lost demand.")

        col1, col2 = st.columns(2)

        with col1:
            optimization_objective = st.selectbox(
                "Optimization Objective:",
                ["Minimize Total Cost", "Minimize Lost Demand", "Minimize Waste", "Balance Both"]
            )

            optimization_trials = st.slider(
                "Number of Optimization Trials:",
                min_value=10, max_value=100, value=50,
                help="More trials = better results but longer computation"
            )

        with col2:
            alpha_weight = st.slider(
                "Waste vs Lost Demand Weight:",
                min_value=0.01, max_value=10.0, value=1.0, step=0.1,
                help="Higher = prioritize waste reduction, Lower = prioritize demand fulfillment"
            )

        if st.button("🔍 Run Optimization", key="run_optimization"):
            st.info("🚧 Optimization feature coming soon! This would run Bayesian optimization to find optimal parameters.")

            # Placeholder for optimization results
            st.subheader("📊 Optimization Results (Example)")

            example_results = pd.DataFrame({
                'Parameter': ['Starting Inventory Factor', 'Alarm Threshold Factor', 'Restock Target Factor'],
                'Current Value': [2.2, 0.5, 0.95],
                'Optimized Value': [1.8, 0.3, 1.2],
                'Improvement': ['-18%', '-40%', '+26%']
            })

            st.dataframe(example_results, use_container_width=True, hide_index=True)

    elif tracking_type == "📈 Performance Analysis":
        st.subheader("📈 Historical Performance Analysis")

        st.write("Analyze inventory performance across different time periods.")

        # Period comparison
        periods = {
            "Q1 2015": ("2015-01-01", "2015-03-31"),
            "Q2 2015": ("2015-04-01", "2015-06-30"),
            "Q3 2015": ("2015-07-01", "2015-09-30"),
            "Q4 2015": ("2015-10-01", "2015-12-31")
        }

        if st.button("📊 Run Performance Analysis", key="performance_analysis"):
            with st.spinner("Analyzing performance across quarters..."):

                quarterly_results = []

                for quarter, (start, end) in periods.items():
                    result = simulate_inventory_fixed(
                        df=df,
                        inventory_params=inventory_params,
                        recipes_dict=recipes_dict,
                        ingredient_masses_kg=ingredient_masses_kg,
                        start_date=start,
                        end_date=end,
                        restock_interval_days=7,
                        alarm_threshold=3
                    )

                    # Calculate period metrics
                    period_orders = len(df[(df['order_date'] >= start) & (df['order_date'] <= end)])
                    period_usage = daily_ingredient_usage[
                        (daily_ingredient_usage.index >= start) &
                        (daily_ingredient_usage.index <= end)
                    ].sum().sum()

                    quarterly_results.append({
                        'Quarter': quarter,
                        'Lost Demand': result['lost_demand'],
                        'Waste (kg)': result['waste_total_kg'],
                        'Alarms': result['manual_alarms'],
                        'Restock Events': result['restock_events'],
                        'Lost Demand %': (result['lost_demand'] / period_orders * 100) if period_orders > 0 else 0,
                        'Waste %': (result['waste_total_kg'] / period_usage * 100) if period_usage > 0 else 0
                    })

                quarterly_df = pd.DataFrame(quarterly_results)

                # Display quarterly comparison
                st.subheader("📅 Quarterly Performance Comparison")
                st.dataframe(quarterly_df, use_container_width=True, hide_index=True)

                # Performance charts
                col1, col2 = st.columns(2)

                with col1:
                    fig_quarterly_lost = px.bar(
                        quarterly_df, x='Quarter', y='Lost Demand %',
                        title="📉 Lost Demand Rate by Quarter"
                    )
                    st.plotly_chart(fig_quarterly_lost, use_container_width=True)

                with col2:
                    fig_quarterly_waste = px.bar(
                        quarterly_df, x='Quarter', y='Waste %',
                        title="♻️ Waste Rate by Quarter"
                    )
                    st.plotly_chart(fig_quarterly_waste, use_container_width=True)

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
        target_date = st.date_input(
            "Select date for prediction:",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date(),
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
                    recommendations.append(f"  • {ingredient}: {qty:.1f} units")

                for rec in recommendations:
                    st.write(f"• {rec}")

    elif forecast_type == "📊 Multi-Day Forecast":
        st.subheader("📊 Multi-Day Sales Forecast")

        col1, col2 = st.columns(2)
        with col1:
            start_forecast = st.date_input(
                "Start date:",
                value=datetime.now().date() + timedelta(days=1)
            )

        with col2:
            end_forecast = st.date_input(
                "End date:",
                value=datetime.now().date() + timedelta(days=7),
                min_value=start_forecast
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
st.markdown("""
<div style='text-align: center; color: #666;'>
  <p>🍕 Smart Byte Dashboard | Better overview over your data :)</p>
</div>
""", unsafe_allow_html=True)

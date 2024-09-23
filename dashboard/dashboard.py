import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
base_path = os.getcwd()  # Dapatkan direktori kerja saat ini

order_items_dataset = os.path.join(base_path, 'order_items_dataset.csv')
orders_dataset = os.path.join(base_path, 'orders_dataset.csv')
product_category_name_translation = os.path.join(base_path, 'product_category_name_translation.csv')
products_dataset = os.path.join(base_path, 'products_dataset.csv')
sellers_dataset = os.path.join(base_path, 'sellers_dataset.csv')

try:
    order_item = pd.read_csv(order_items_dataset)
    orders = pd.read_csv(orders_dataset)
    product_category = pd.read_csv(product_category_name_translation)
    product = pd.read_csv(products_dataset)
    sellers = pd.read_csv(sellers_dataset)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")

# Handle missing values
orders = orders.fillna({
    'order_approved_at': 'Unknown', 
    'order_delivered_carrier_date': 'Unknown', 
    'order_delivered_customer_date': 'Unknown'
})
order_item = order_item.fillna({
    'price': order_item['price'].mean(), 
    'freight_value': order_item['freight_value'].mean()
})
product = product.fillna({
    'product_weight_g': product['product_weight_g'].mean(),
    'product_length_cm': product['product_length_cm'].mean(),
    'product_height_cm': product['product_height_cm'].mean(),
    'product_width_cm': product['product_width_cm'].mean()
})

# Convert data types
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'], errors='coerce')
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'], errors='coerce')
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# Sidebar filters
st.sidebar.header('Filters')

# Year selection with an "All Years" option
years = orders['order_purchase_timestamp'].dt.year.unique().tolist()
all_years = st.sidebar.checkbox('Select All Years')

if all_years:
    selected_years = years
else:
    selected_years = st.sidebar.multiselect('Select Year(s)', years, default=years[0])

# Month selection with an "All Months" option
months = orders['order_purchase_timestamp'].dt.month_name().unique().tolist()
all_months = st.sidebar.checkbox('Select All Months')

if all_months:
    selected_months = months 
else:
    selected_months = st.sidebar.multiselect('Select Month(s)', months, default=months[0])

# Filter dataset based on year and month selections
filtered_orders = orders[orders['order_purchase_timestamp'].dt.year.isin(selected_years)]

# If "All Months" is not checked, filter by the selected months
if not all_months:
    filtered_orders = filtered_orders[filtered_orders['order_purchase_timestamp'].dt.month_name().isin(selected_months)]

# Start building the Streamlit app
st.title("Dashboard: E-Commerce Public Dataset")

# 1. Total Orders per Year (2016-2018)
st.subheader(f"Total Orders per Year ({', '.join(map(str, selected_years))})")

# Extract year and count orders per year
orders['year'] = orders['order_purchase_timestamp'].dt.year
total_orders_yearly = orders.groupby('year').size().reset_index(name='total_orders')

# Filter for selected years
filtered_orders_yearly = total_orders_yearly[total_orders_yearly['year'].isin(selected_years)]

# Plot lineplot for total orders per year
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=filtered_orders_yearly, x='year', y='total_orders', marker='o', color='blue', linewidth=2, ax=ax)

# Set only whole years as ticks on the x-axis
ax.set_xticks(filtered_orders_yearly['year'])

for x, y in zip(filtered_orders_yearly['year'], filtered_orders_yearly['total_orders']):
    plt.text(x, y, f'{y}', ha='center', va='bottom', fontsize=10)

plt.title(f'Total Orders Per Year ({", ".join(map(str, selected_years))})', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Orders', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# 2. Average Orders per Month (2016-2018)
st.subheader(f"Average Orders per Month ({', '.join(map(str, selected_years))})")

# Filter orders data by selected year and months
filtered_orders_by_year = filtered_orders.copy()

# Extract month and calculate average orders per month
filtered_orders_by_year['month'] = filtered_orders_by_year['order_purchase_timestamp'].dt.month_name()
avg_orders_monthly = filtered_orders_by_year.groupby('month')['order_id'].count().reset_index(name='average_orders')

# Ensure months are ordered correctly
month_order = pd.Series(pd.date_range('2024-01', '2024-12', freq='M').strftime('%B')).tolist()
avg_orders_monthly = avg_orders_monthly.set_index('month').reindex(month_order, fill_value=0).reset_index()

# Plot lineplot for average orders per month
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=avg_orders_monthly, x='month', y='average_orders', marker='o', color='blue', linewidth=2, ax=ax)

for x, y in zip(avg_orders_monthly['month'], avg_orders_monthly['average_orders']):
    plt.text(x, y, f'{y}', ha='center', va='bottom', fontsize=10)

plt.title(f'Average Orders Per Month ({", ".join(map(str, selected_years))})', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Orders', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# 3. Top 10 Product Categories by Total Orders
st.subheader(f"Top 10 Product Categories ({'All Months' if all_months else ', '.join(selected_months)}) {', '.join(map(str, selected_years))}")

# Merge data to get product categories
merged_data = order_item.merge(product[['product_id', 'product_category_name']], on='product_id')
merged_data_filtered = merged_data[merged_data['order_id'].isin(filtered_orders['order_id'])]

top_categories = merged_data_filtered['product_category_name'].value_counts().head(10).reset_index()
top_categories.columns = ['product_category_name', 'total_orders']

# Plot top product categories
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=top_categories, x='product_category_name', y='total_orders', color='blue', ax=ax) 

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12)

plt.title(f'Top 10 Product Categories ({", ".join(map(str, selected_years))}, {"All Months" if all_months else ", ".join(selected_months)})', fontsize=16, fontweight='bold')
plt.xlabel('Product Category', fontsize=14)
plt.ylabel('Total Orders', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# 4. Top 5 Cities with the Most Sellers
st.subheader(f"Top 5 Cities with the Most Sellers ({', '.join(map(str, selected_years))}, {'All Months' if all_months else ', '.join(selected_months)})")

# Count sellers per city
top_seller_cities = sellers['seller_city'].value_counts().head(5).reset_index()
top_seller_cities.columns = ['seller_city', 'total_sellers']

# Plot top seller cities
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_seller_cities, x='seller_city', y='total_sellers', color='blue', ax=ax) 

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12)

plt.title(f'Top 5 Cities with the Most Sellers ({", ".join(map(str, selected_years))}, {"All Months" if all_months else ", ".join(selected_months)})', fontsize=16, fontweight='bold')
plt.xlabel('City', fontsize=14)
plt.ylabel('Total Sellers', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
st.pyplot(fig)
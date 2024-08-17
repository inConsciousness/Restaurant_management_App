import pandas as pd
import numpy as np

def generate_orders():
    num_orders = 500
    num_customers = 100
    np.random.seed(42)
    
    orders = pd.DataFrame({
        'order_id': range(1, num_orders + 1),
        'customer_id': np.random.randint(1, num_customers + 1, size=num_orders),
        'item_id': np.random.randint(1, 50, size=num_orders),  # Assuming 50 menu items
        'rating': np.random.randint(1, 6, size=num_orders)  # Ratings between 1 and 5
    })
    
    orders.to_csv('orders.csv',index=False)
    print("orders.csv generated")

if __name__ == "__main__":
    generate_orders()

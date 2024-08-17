import pandas as pd
import numpy as np

def generate_customers():
    num_customers = 100
    np.random.seed(42)
    
    customers = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'age': np.random.randint(18, 70, size=num_customers),
        'gender': np.random.choice(['male', 'female'], size=num_customers),
        'preference': np.random.choice(['vegan', 'vegetarian', 'non-vegetarian'], size=num_customers),
        'likes_spicy': np.random.choice([True, False], size=num_customers)
    })
    customers.to_csv('customers.csv',index=False)
    print("customers.csv generated")

if __name__ == "__main__":
    generate_customers()

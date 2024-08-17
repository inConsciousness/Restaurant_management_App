import pandas as pd
import numpy as np

def generate_menu():
    num_menu_items = 50
    np.random.seed(42)
    
    menu = pd.DataFrame({
        'item_id': range(1, num_menu_items + 1),
        'description': [f"Delicious dish {i}" for i in range(1, num_menu_items + 1)],
        'ingredients': [f"Ingredient {i}" for i in range(1, num_menu_items + 1)],
        'category': np.random.choice(['appetizer', 'main course', 'dessert'], size=num_menu_items)
    })
    menu.to_csv('menu.csv',index=False)
    print("menu.csv generated")

if __name__ == "__main__":
    generate_menu()

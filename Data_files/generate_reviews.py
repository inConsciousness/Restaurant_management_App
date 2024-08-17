import pandas as pd
import numpy as np

def generate_reviews():
    num_reviews = 300
    num_customers = 100
    num_menu_items = 50
    np.random.seed(42)
    
    reviews = pd.DataFrame({
        'customer_id': np.random.randint(1, num_customers + 1, size=num_reviews),
        'item_id': np.random.randint(1, num_menu_items + 1, size=num_reviews),
        'review_text': [f"This is a review for dish {np.random.randint(1, num_menu_items + 1)}" for _ in range(num_reviews)],
        'rating': np.random.randint(1, 6, size=num_reviews)  # Ratings between 1 and 5
    })
    
    reviews.to_csv('reviews.csv',index=False)
    print("reviews.csv generated")

if __name__ == "__main__":
    generate_reviews()

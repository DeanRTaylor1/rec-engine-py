import pandas as pd
import random

class DataSimulator:
    def __init__(self, num_users=1000, num_items=100, max_interactions_per_user=20, max_purchase_conut=10):
        self.num_users = num_users
        self.num_items = num_items
        self.max_interactions_per_user = max_interactions_per_user
        self.max_purchase_count = max_purchase_conut
    
    def generate_data(self):
        data = []

        for user_id in range(1, self.num_users + 1):
            num_interactions = random.randint(1, self.max_interactions_per_user)

            # Randomly sample items for the user to interact with
            items_interacted_with = random.sample(range(1, self.num_items + 1), num_interactions)

            for item_id in items_interacted_with:
                purchase_count = random.randint(1, self.max_purchase_count)
                data.append([user_id, item_id, purchase_count])

        df = pd.DataFrame(data, columns=['userID', 'itemID', 'purchaseCount'])
        return df

    def save_to_csv(self, filename="simulated_data.csv"):
        df = self.generate_data()
        df.to_csv(filename, index=False)


simulator = DataSimulator(num_users=5000, num_items=200)
simulator.save_to_csv("test_data.csv")

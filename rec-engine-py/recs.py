import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

class Recommender:
    def __init__(self, data_dict):
        # Convert the input dictionary to a pandas DataFrame
        self.df = pd.DataFrame(data_dict)
        
        # Create a reader with the appropriate rating scale, derived from the purchase counts
        reader = Reader(rating_scale=(self.df['purchaseCount'].min(), self.df['purchaseCount'].max()))
        
        # Convert the DataFrame to a Surprise dataset
        self.data = Dataset.load_from_df(self.df, reader)
        
        # Split the data into a training and a test set for later evaluation
        self.trainset, self.testset = train_test_split(self.data, test_size=.25)
        
        # Initialize the Singular Value Decomposition (SVD) model
        self.model = SVD()
        
        # Train the SVD model using the training data
        self.model.fit(self.trainset)
        
    def _get_top_n_recommendations(self, predictions, n=10):
        # Create a dictionary to hold the top n item recommendations for each user
        top_n = {}
        
        # Process each prediction
        for uid, iid, true_r, est, _ in predictions:
            # Append item predictions for each user
            top_n.setdefault(uid, []).append((iid, est))
            
        # Sort and retrieve the top n items for each user
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        
        return top_n
    
    def recommend_for_user(self, user_id, n=10):
        # Build a training set from the entire dataset
        trainset = self.data.build_full_trainset()
        
        # Create an anti-testset: all the items that the user has not interacted with
        anti_testset = trainset.build_anti_testset()
        
        # Filter out interactions for all users except the target user
        anti_testset = filter(lambda x: x[0] == user_id, anti_testset)
        
        # Use the trained model to predict ratings for the user-item pairs in the anti-testset
        predictions = self.model.test(anti_testset)
        
        # Retrieve the top n items for the user based on the predictions
        top_n = self._get_top_n_recommendations(predictions, n=n)
        return [item[0] for item in top_n[user_id]]

    def evaluate(self):
        # Use the trained model to predict ratings for the test set
        predictions = self.model.test(self.testset)
        
        # Calculate and return the Root Mean Squared Error (RMSE) for the predictions
        return accuracy.rmse(predictions)


# # Sample usage:
# data_dict = {
#     'userID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
#     'itemID': ['A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'A'],
#     'purchaseCount': [5, 3, 3, 1, 1, 5, 5, 4, 4, 2]
# }

# Load data from CSV
df = pd.read_csv('test_data.csv')

# Convert DataFrame to dictionary format
data_dict = {
    'userID': df['userID'].tolist(),
    'itemID': df['itemID'].tolist(),
    'purchaseCount': df['purchaseCount'].tolist()
}

recommender = Recommender(data_dict)
print(f"RMSE: {recommender.evaluate()}")
user_id = 20
recommended_items = recommender.recommend_for_user(user_id)
print(f"Recommended items for user {user_id}: {recommended_items}")

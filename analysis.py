# Complete Python script including the addition of similarity scores for identifying similar users

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
file_path = 'clean_data.csv'
user_visits_data = pd.read_csv(file_path)

# Preprocessing: Creating a user-venue matrix
user_venue_matrix = pd.pivot_table(user_visits_data, values='visitCount', index='userId',
                                   columns='venueCategory', aggfunc=np.sum, fill_value=0)

# NMF Model
nmf = NMF(n_components=40, init='random', random_state=0)
user_nmf_features = nmf.fit_transform(user_venue_matrix)
venue_nmf_features = nmf.components_

# Preparing the user-venue matrix for SVD
user_venue_matrix_filled = user_venue_matrix.astype('float32')
user_venue_matrix_sparse = csr_matrix(user_venue_matrix_filled.values)

# SVD Model
U, sigma, Vt = svds(user_venue_matrix_sparse, k=40)
sigma = np.diag(sigma)

# Function to generate recommendations
def generate_recommendations(user_id, user_features, venue_features, user_venue_matrix, top_n=10):
    user_idx = user_venue_matrix.index.get_loc(user_id)
    predicted_scores = np.dot(user_features[user_idx, :], venue_features)
    sorted_venue_indices = predicted_scores.argsort()[::-1]
    top_venue_indices = sorted_venue_indices[:top_n]
    recommended_venues = user_venue_matrix.columns[top_venue_indices]
    return recommended_venues

# Generating recommendations for user 6 using NMF and SVD
nmf_recommendations = generate_recommendations(42, user_nmf_features, venue_nmf_features, user_venue_matrix)
svd_recommendations = generate_recommendations(42, U.dot(sigma), Vt, user_venue_matrix)

# Function to enrich recommendations with venue details
def enrich_recommendations(recommendations, user_visits_data):
    enriched_recommendations = []
    for venue in recommendations:
        venue_details = user_visits_data[user_visits_data['venueCategory'] == venue]
        most_visited = venue_details.groupby(['venueIdEncoded', 'latitude', 'longitude']).agg({'visitCount': 'sum'}).reset_index().sort_values('visitCount', ascending=False).iloc[0]
        enriched_recommendations.append({
            'Venue Category': venue,
            'Latitude': most_visited['latitude'],
            'Longitude': most_visited['longitude'],
            'Total Visits': most_visited['visitCount']
        })
    return pd.DataFrame(enriched_recommendations)

# Enriching NMF and SVD recommendations
enriched_nmf_recommendations = enrich_recommendations(nmf_recommendations, user_visits_data)
enriched_svd_recommendations = enrich_recommendations(svd_recommendations, user_visits_data)

# Computing relevant items for user i
user_i_data = user_visits_data[user_visits_data['userId'] == 42]
relevant_items_set = set(user_i_data['venueCategory'])
total_unique_venues = set(user_visits_data['venueCategory'])

# Function to compute metrics
def compute_metrics(recommended_items, relevant_items, total_items):
    recommended_set = set(recommended_items)
    relevant_set = set(relevant_items)
    true_positives = len(recommended_set & relevant_set)
    precision = true_positives / len(recommended_set) if len(recommended_set) > 0 else 0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0
    coverage = len(recommended_set) / len(total_items) if len(total_items) > 0 else 0
    return {"Precision": precision, "Recall": recall, "Coverage": coverage}

# Computing metrics for NMF and SVD recommendations
nmf_metrics = compute_metrics(nmf_recommendations, relevant_items_set, total_unique_venues)
svd_metrics = compute_metrics(svd_recommendations, relevant_items_set, total_unique_venues)

# Compute Cosine Similarity Matrix for Similar User Identification
cosine_sim_matrix = cosine_similarity(user_venue_matrix)

# Convert Cosine Similarity Matrix to DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_venue_matrix.index, columns=user_venue_matrix.index)

# Identifying Similar Users to User i, including similarity scores
similar_users_scores = cosine_sim_df[42].sort_values(ascending=False)[1:11]  # Top 10 similar users to User i with scores

# Convert the similar users and their scores to a DataFrame
similar_users_df = similar_users_scores.reset_index()
similar_users_df.columns = ['UserId', 'SimilarityScore']

# Finding Common Venues Visited by Similar Users, limited to top 10 and including GPS coordinates
common_venues_df = user_visits_data[user_visits_data['userId'].isin(similar_users_df['UserId'])]
common_venues = common_venues_df.groupby(['venueCategory', 'latitude', 'longitude']).agg({'visitCount': 'sum'}).reset_index()
common_venues_sorted = common_venues.sort_values(by='visitCount', ascending=False).head(10)

# Saving Results to CSV Files
enriched_nmf_recommendations.to_csv('enriched_nmf_recommendations.csv', index=False)
enriched_svd_recommendations.to_csv('enriched_svd_recommendations.csv', index=False)
pd.DataFrame([nmf_metrics]).to_csv('nmf_metrics.csv', index=False)
pd.DataFrame([svd_metrics]).to_csv('svd_metrics.csv', index=False)
similar_users_df.to_csv('similar_users_with_scores.csv', index=False)
common_venues_sorted.to_csv('top_common_venues.csv', index=False)

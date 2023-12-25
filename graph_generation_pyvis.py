import pandas as pd
import networkx as nx
import re
from datetime import datetime
from pyvis.network import Network
from tqdm import tqdm

# Load your data into a DataFrame
data = pd.read_csv('nyc_user_visits_with_timestamps.csv')

# Function to convert timestamp strings to datetime objects
def parse_timestamps(ts_string):
    datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    timestamps = re.findall(datetime_pattern, ts_string)
    return [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]

# Apply the parsing function to the 'visitTimestamps' column
data['visitTimestamps'] = data['visitTimestamps'].apply(parse_timestamps)

# Filter data to include only the first 4 users
first_four_users = data['userId'].unique()[:10]
filtered_data = data[data['userId'].isin(first_four_users)]

# Function to convert datetime objects to strings
def datetime_to_string(dt_list):
    return [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in dt_list]

# Initialize the graph
G = nx.Graph()

# Add nodes with labels
for user in tqdm(first_four_users, desc="Adding User Nodes"):
    G.add_node(f"U{user}", label=f"User {user}", type='user', user_id=int(user))

for _, row in tqdm(filtered_data.drop_duplicates('venueIdEncoded').iterrows(), total=filtered_data['venueIdEncoded'].nunique(), desc="Adding Venue Nodes"):
    venue_node = f"V{row['venueIdEncoded']}"
    label = f"{row['venueCategory']} (ID: {row['venueIdEncoded']})\nLat: {row['latitude']}, Long: {row['longitude']}"
    G.add_node(venue_node, label=label, color='orange', type='venue', venue_id=int(row['venueIdEncoded']),
               venue_category=row['venueCategory'], venue_category_encoded=int(row['venueCategoryEncoded']),
               latitude=row['latitude'], longitude=row['longitude'])

# Add edges with titles (hover-text)
for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Adding Edges"):
    user_node = f"U{row['userId']}"
    venue_node = f"V{row['venueIdEncoded']}"
    visit_count = int(row['visitCount'])
    visit_timestamps = datetime_to_string(row['visitTimestamps'])
    edge_title = f"Visits: {visit_count}, Dates: {', '.join(visit_timestamps)}"

    if G.has_edge(user_node, venue_node):
        G[user_node][venue_node]['visit_count'] += visit_count
        G[user_node][venue_node]['visit_timestamps'].extend(visit_timestamps)
    else:
        G.add_edge(user_node, venue_node, title=edge_title, visit_count=visit_count, visit_timestamps=visit_timestamps)

# Create a Pyvis network from the NetworkX graph
net = Network(notebook=False)
net.from_nx(G)

# Customize the visualization
net.show_buttons(filter_=['physics'])
net.toggle_physics(True)

# Save and show the graph
net.save_graph('graph.html')
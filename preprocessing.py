import pandas as pd
import os
import glob
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("foursquare_data/dataset_TSMC2014_NYC.csv")  # Replace with your actual dataset file path
# Convert utcTimestamp to the correct format
df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S +0000 %Y')
# Format the timestamp column as desired
df['utcTimestamp'] = df['utcTimestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
# Drop the 'timezoneOffset' column
df = df.drop(columns=['timezoneOffset'])

# Save the updated dataset to a new CSV file
df.to_csv("nyc_dataset.csv", index=False)  # Replace with your desired output file path

# Creating distinct dataframes for each unique user ID
distinct_user_dfs = {uid: df[df['userId'] == uid] for uid in df['userId'].unique()}

# Creating a directory to store the user dataframes
output_dir = 'nyc_users'  # Replace with your desired output path
os.makedirs(output_dir, exist_ok=True)

# Saving each dataframe as a separate CSV file in the specified directory
for user_id, user_df in distinct_user_dfs.items():
    user_df.to_csv(os.path.join(output_dir, f'user_{user_id}.csv'), index=False)

# Directory containing the user data files
data_directory = 'nyc_users/'

# Getting all CSV file paths in the directory
csv_file_paths = glob.glob(os.path.join(data_directory, '*.csv'))

# Loading and concatenating all user data into a single DataFrame
combined_data = pd.concat([pd.read_csv(file) for file in csv_file_paths])

# Data Cleaning and Preprocessing
# Converting 'utcTimestamp' to datetime
combined_data['utcTimestamp'] = pd.to_datetime(combined_data['utcTimestamp'])

# Extracting time-based features: hour of day and day of week
combined_data['hourOfDay'] = combined_data['utcTimestamp'].dt.hour
combined_data['dayOfWeek'] = combined_data['utcTimestamp'].dt.dayofweek

# Encoding categorical variables
label_encoder = LabelEncoder()
combined_data['venueIdEncoded'] = label_encoder.fit_transform(combined_data['venueId'])
combined_data['venueCategoryEncoded'] = label_encoder.fit_transform(combined_data['venueCategory'])

# Displaying the normalized user preferences (for example)
print(combined_data.head())

# Saving the combined data to a CSV file
combined_data.to_csv('nyc_combined_users_all.csv', index=False)

# Analyzing user visit patterns
user_visit_counts = combined_data.groupby(['userId', 'venueIdEncoded', 'venueCategory', 'venueCategoryEncoded', 'latitude', 'longitude']).size().reset_index(name='visitCount')

# Group by 'userId' and 'venueIdEncoded' and aggregate 'utcTimestamp' into a list
visit_timestamps = combined_data.groupby(['userId', 'venueIdEncoded'])['utcTimestamp'].apply(list).reset_index(name='visitTimestamps')

# Merging the visit timestamps with the user_visit_counts DataFrame
user_visit_counts = user_visit_counts.merge(visit_timestamps, on=['userId', 'venueIdEncoded'])

# Displaying the updated DataFrame
print(user_visit_counts)

# Saving the updated DataFrame to CSV
user_visit_counts.to_csv('clean_data.csv', index=False)

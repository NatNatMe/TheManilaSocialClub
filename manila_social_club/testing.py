import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('skintone.csv')

# Function to recommend a color based on skin tone
def recommend_color_for_skin_tone(new_skin_tone):
    # Calculate the Euclidean distance between the new skin tone and the existing skin tones
    df['distance'] = np.sqrt((df['r'] - new_skin_tone[0])**2 + (df['g'] - new_skin_tone[1])**2 + (df['b'] - new_skin_tone[2])**2)
    
    # Find the skin tone category that is closest to the new skin tone
    closest_skin_tone_category = df.loc[df['distance'].idxmin(), 'skin_tone']
    
    # Filter the dataframe to get the colors for the closest skin tone category
    recommended_colors = df[df['skin_tone'] == closest_skin_tone_category]['color'].unique()
    
    # Randomly select up to 3 colors
    recommended_colors = np.random.choice(recommended_colors, size=min(3, len(recommended_colors)), replace=False)
    
    return recommended_colors

# New skin tone to recommend a color for (e.g., [255, 224, 189])
new_skin_tone = np.array([255, 224, 189])

# Get the recommended colors
recommended_colors = recommend_color_for_skin_tone(new_skin_tone)
print(f"This is the detected skin tone: {new_skin_tone}")
print(f"Recommended colors for the given skin tone: {recommended_colors}")

import pandas as pd
import random
import re

def create_messy_dataset(clean_file='research_papers_dataset.csv', 
                        output_file='messy_research_papers.csv'):
    """
    Create a messy version of the clean dataset to simulate real-world data problems
    """
    # Load the clean dataset
    df = pd.read_csv(clean_file)
    print(f"Loaded clean dataset with {len(df)} rows")
    
    # Create a copy for messing up
    messy_df = df.copy()
    
    # 1. Introduce missing abstracts (10% of data)
    missing_abstract_idx = random.sample(range(len(messy_df)), k=int(0.1 * len(messy_df)))
    messy_df.loc[missing_abstract_idx, 'abstract'] = ''
    
    # 2. Create inconsistent categories
    category_variations = {
        'Technology': ['Tech', 'technology', 'TECHNOLOGY', 'Computer Science'],
        'Healthcare': ['Health', 'healthcare', 'Medical', 'Medicine'],
        'Finance': ['Financial', 'finance', 'Economics', 'Economic'],
        'Education': ['Educational', 'education', 'Learning', 'Teaching'],
        'Environment': ['Environmental', 'environment', 'Climate', 'Green']
    }
    
    for i, row in messy_df.iterrows():
        if random.random() < 0.15:  # 15% chance of category variation
            original_cat = row['category']
            if original_cat in category_variations:
                new_cat = random.choice(category_variations[original_cat])
                messy_df.loc[i, 'category'] = new_cat
    
    # 3. Add HTML tags and special characters to some abstracts
    html_noise = ['<p>', '</p>', '<br>', '&amp;', '&lt;', '&gt;', '\n', '\t']
    for i in random.sample(range(len(messy_df)), k=int(0.08 * len(messy_df))):
        if messy_df.loc[i, 'abstract']:  # Only if abstract exists
            noise = random.choice(html_noise)
            messy_df.loc[i, 'abstract'] = noise + messy_df.loc[i, 'abstract'] + noise
    
    # 4. Create duplicate entries with slight variations
    duplicate_indices = random.sample(range(len(messy_df)), k=int(0.05 * len(messy_df)))
    duplicates = []
    
    for idx in duplicate_indices:
        dup_row = messy_df.loc[idx].copy()
        # Slightly modify the duplicate
        dup_row['title'] = dup_row['title'] + ' (Duplicate)'
        duplicates.append(dup_row)
    
    # Add duplicates to dataframe
    duplicate_df = pd.DataFrame(duplicates)
    messy_df = pd.concat([messy_df, duplicate_df], ignore_index=True)
    
    # 5. Add some completely empty rows
    empty_rows = pd.DataFrame({
        'title': [''] * 5,
        'abstract': [''] * 5,
        'category': [''] * 5,
        'text': [''] * 5
    })
    messy_df = pd.concat([messy_df, empty_rows], ignore_index=True)
    
    # 6. Shuffle the dataset
    messy_df = messy_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 7. Update the 'text' column to reflect changes
    messy_df['text'] = messy_df['title'].astype(str) + ' ' + messy_df['abstract'].astype(str)
    
    # Save the messy dataset
    messy_df.to_csv(output_file, index=False)
    
    print(f"\nMessy dataset created: {output_file}")
    print(f"Total rows: {len(messy_df)} (added {len(messy_df) - len(df)} problematic rows)")
    print(f"Missing abstracts: {messy_df['abstract'].eq('').sum()}")
    print(f"Empty rows: {messy_df['title'].eq('').sum()}")
    print(f"Category variations: {messy_df['category'].nunique()} unique categories")
    
    return messy_df

# Create the messy dataset
if __name__ == "__main__":
    messy_data = create_messy_dataset()
    print("Messy dataset created successfully!")

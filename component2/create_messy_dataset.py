"""
Create Messy Dataset for Testing Advanced Data Processing Pipeline
Simulates real-world data quality issues found in production systems
"""

import pandas as pd
import random
import re
import numpy as np

def create_messy_dataset(clean_file='research_papers_dataset.csv', 
                        output_file='messy_research_papers.csv'):
    """
    Create a messy version of the clean dataset to simulate real-world data problems
    
    Args:
        clean_file (str): Path to clean dataset
        output_file (str): Path to save messy dataset
        
    Returns:
        pandas.DataFrame: Messy dataset with various data quality issues
    """
    # TODO: Load the clean dataset and create messy version
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Real-world data is never clean. This function simulates the types of problems
    # you'll encounter when working with data from websites, databases, and user uploads.
    # Think of this as taking a perfect dataset and making it realistically messy.
    # 
    # DATA QUALITY ISSUES TO SIMULATE:
    # 1. Missing abstracts (10% of papers will have empty abstracts)
    # 2. Inconsistent category labels ("Tech" vs "Technology", "Health" vs "Healthcare")
    # 3. HTML tags and special characters in text
    # 4. Duplicate entries with slight variations
    # 5. Completely empty/invalid rows
    # 6. Varying text quality and lengths
    # 
    # SUB-STEPS TO IMPLEMENT:
    # 1a. Load the clean dataset using pd.read_csv()
    # 1b. Create a copy to modify (messy_df = df.copy())
    # 1c. Introduce missing abstracts (randomly set 10% of abstracts to empty string)
    # 1d. Create category inconsistencies (map some categories to variations)
    # 1e. Add HTML tags to some abstracts (inject <p>, </p>, &amp; etc.)
    # 1f. Create duplicate entries with slight modifications
    # 1g. Add some completely empty rows
    # 1h. Update the 'text' column to reflect all changes
    # 1i. Shuffle the dataset and save to CSV
    # 1j. Print summary statistics of problems introduced
    # 
    # YOUR CODE HERE:
    
    
    print("Messy dataset creation completed!")

def add_category_inconsistencies(df):
    """
    Add inconsistent category labels to simulate real-world data entry problems
    
    Args:
        df (pandas.DataFrame): Dataset to modify
        
    Returns:
        pandas.DataFrame: Dataset with inconsistent categories
    """
    # TODO: Create category mapping variations
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # In real datasets, the same category often appears with different spellings,
    # capitalizations, or abbreviations. This simulates that problem.
    # 
    # CATEGORY VARIATIONS TO CREATE:
    # Technology -> Tech, technology, TECHNOLOGY, Computer Science
    # Healthcare -> Health, healthcare, Medical, Medicine
    # Finance -> Financial, finance, Economics, Economic
    # Education -> Educational, education, Learning, Teaching
    # Environment -> Environmental, environment, Climate, Green
    # 
    # PROCESS:
    # 1. Create dictionary mapping standard categories to their variations
    # 2. For 15% of rows, randomly replace category with a variation
    # 3. Return modified dataset
    # 
    # YOUR CODE HERE:
    
    return df

def add_html_noise(df, noise_percentage=0.08):
    """
    Add HTML tags and special characters to simulate web-scraped data
    
    Args:
        df (pandas.DataFrame): Dataset to modify
        noise_percentage (float): Percentage of rows to add noise to
        
    Returns:
        pandas.DataFrame: Dataset with HTML noise
    """
    # TODO: Add realistic HTML and encoding issues
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # When data comes from websites, it often contains HTML tags and encoded characters.
    # This function simulates that realistic scenario.
    # 
    # HTML ISSUES TO SIMULATE:
    # - HTML tags: <p>, </p>, <br>, <div>, </div>
    # - Encoded characters: &amp;, &lt;, &gt;, &nbsp;
    # - Line breaks and tab characters: \n, \t
    # 
    # PROCESS:
    # 1. Create list of common HTML noise elements
    # 2. Randomly select percentage of rows to modify
    # 3. For selected rows, randomly inject HTML noise into abstracts
    # 4. Return modified dataset
    # 
    # YOUR CODE HERE:
    
    return df

def create_duplicates(df, duplicate_percentage=0.05):
    """
    Create duplicate entries with slight variations
    
    Args:
        df (pandas.DataFrame): Dataset to modify  
        duplicate_percentage (float): Percentage of duplicates to create
        
    Returns:
        pandas.DataFrame: Dataset with duplicates added
    """
    # TODO: Create realistic duplicate entries
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # Real datasets often have duplicate entries that aren't exactly identical.
    # They might have slight differences in spacing, punctuation, or wording.
    # 
    # DUPLICATE VARIATIONS TO CREATE:
    # - Add " (Duplicate)" to titles
    # - Change spacing or punctuation slightly
    # - Modify capitalization
    # - Add extra whitespace
    # 
    # PROCESS:
    # 1. Randomly select rows to duplicate based on percentage
    # 2. For each selected row, create modified copy
    # 3. Add variations to make duplicates realistic but not identical
    # 4. Concatenate duplicates with original dataset
    # 5. Return combined dataset
    # 
    # YOUR CODE HERE:
    
    return df

# Example usage and testing
if __name__ == "__main__":
    print("Creating messy dataset for testing...")
    
    # TODO: Run the messy dataset creation process
    # STEP-BY-STEP INSTRUCTIONS FOR STUDENTS:
    # This section ties everything together to create your test dataset
    # 
    # EXECUTION STEPS:
    # 1. Call create_messy_dataset() function
    # 2. Print confirmation message
    # 3. Load and display basic statistics about the messy dataset
    # 4. Show examples of the types of problems created
    # 
    # YOUR CODE HERE:
    
    
    print("Messy dataset ready for pipeline testing!")

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges messages data file and categories data file into a pandas dataframe.
    
    Parameters
    ----------
    messages_filepath : csv file location
    
    categories_filepath : csv file location.
    
    Returns
    -------
    pandas dataFrame
    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    ##### Merge datasets #####
    df = pd.concat([messages, categories], axis=1) # This effective merges on the index, avoiding problems with duplicate id values.
    return df


def clean_data(df):
    """
    Cleans pandas dataframe `df` as follows.
    
    1. The last column (i.e., the category column) is split up into multiple columns for each possible category.
    2. Each column contains either a zero or a one.
    3. Duplicate rows are removed.
    
    
    Parameters
    ----------
    df : pandas DataFrame
    
    Returns
    -------
    pandas dataFrame
    
    """
    
    ##### Create a new `categories` dataframe containing individual category columns. #####
    categories_from_df = df.iloc[:,-1]
    categories = categories_from_df.str.split(";", expand = True)
    
    # Use the first row to extract a list of new column names for categories.
    row = categories.iloc[0,:]
    category_colnames = [row[:-2] for row in row]
    categories.columns = category_colnames
    
    # Convert category values to just 0 or 1.
    for column in categories:
        # Set each value to be the last character of the string.
        categories[column] = categories[column].astype(str).str.slice(-1,-2,-1)
    
        # Convert column from string to numeric.
        categories[column] = categories[column].astype(int)
        
    ##### Replace categories column in df with new category columns #####
    # Drop the original categories column from `df`.
    df.drop(columns='categories', inplace = True)
    # Concatenate the original dataframe with the new `categories` dataframe.
    df = pd.concat([df, categories], axis=1)  
    df = df.drop_duplicates().reset_index() # Drop duplicates
    return  df
    

def save_data(df, database_filename):
    """
    Saves pandas dataframe to SQLite database.
    
    The name of the table in the database is `DisasterResponse'.
    
    Parameters
    ----------
    df : pandas DataFrame
    database_filename : Name of the sql database file
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    """ Loads, cleans and saves data to a database. """
    if len(sys.argv) == 4:

        # Collects arguments into their appropriate variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
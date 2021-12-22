import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages csv (at messages_filepath) and categories csv (at categories_filepath) and merges them into a pandas dataframe.
    Returns merged dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    ##### Merge datasets #####
    df = pd.concat([messages, categories], axis=1) # This effective merges on the index, avoiding problems with duplicate id values.
    return df


def clean_data(df):
    '''
    Cleans pandas dataframe df as follows:
    - The last column (i.e. the category column) is split up into multiple columns for each possible category
    - Each column contains either a zero or one
    - Duplicate rows are removed
    - Returns cleaned df 
    '''
    ##### create a separate dataframe with individual category columns #####
    categories_from_df = df.iloc[:,-1]
    categories = categories_from_df.str.split(";", expand = True)
    
    # use the first row to extract a list of new column names for categories.
    row = categories.iloc[0,:]
    category_colnames = [row[:-2] for row in row]
    categories.columns = category_colnames
    
    # Convert category values to just 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1,-2,-1)
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    ##### Replace categories column in df with new category columns #####
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)  
    df = df.drop_duplicates().reset_index() # Drop duplicates
    return  df
    

def save_data(df, database_filename):
    ''' 
    Saves df to sql database 
    The name of the database file is database_filename
    The name of the table in the database is also database_filename (without .db)
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename[:-3], engine, index=False)
    
    pass  


def main():
    ''' Loads, cleans and saves data to a database'''
    if len(sys.argv) == 4:

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
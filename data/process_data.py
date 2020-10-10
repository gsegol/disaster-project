import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.head())
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.head())  
    #
    # merge datasets using the common id
    df = messages.merge(categories, how='outer', on=['id'])
    df.head()
    
    return df
    
def clean_data(df):
    #Split `categories` into separate category columns.  Use ";" as the separator.
    # print the first row for reference
    print(df['categories'][0])
    categories = df['categories'].str.split(';', expand=True)
    print("Size of the created data frame: ", categories.shape)
    print(categories.head())
    #
    #extract the names of the new columns from the first row of the dataframe
    row = categories.iloc[0,:]
    category_colnames = row.str.rpartition('-')[0]
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    print(categories.head())
    #
    #Convert category values to numbers 0 or 1 (trailing characters of the string)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.rpartition('-')[2]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    print(categories.head())
    #
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    #then merge df and categories dataframes
    df = pd.concat([df, categories], axis=1)
    print(df.head())
    #
    # remove duplicates.
    # Use the "original" column to check for duplicates
    # check the number of duplicates
    #
    df.duplicated(subset='original', keep='first').value_counts()
    # drop duplicates
    df = df.drop_duplicates(subset='original', keep='first')
    print("size of the cleaned data base: ", df.shape)
    #
    return df
   
def save_data(df, database_filename):
    
    # Save the cleaned data base to an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    
def main():
    
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
from src.scraper import Scraper
import pandas as pd


def fetch_data():
    with Scraper() as scraper:
        scraper.fetch_immo_list(property_type='house', max_pages=5)
        # scraper.fetch_details_soup_multithread(property_type='house')
    # inspect()


if __name__ == '__main__':
    fetch_data()

def read_data():
    with open('data.csv', 'r', encoding='utf-8') as f:
        res = sum(1 for line in f)
        print("Number of lines:", res)

def drop_duplicates():
    df = pd.read_csv("data.csv")
    df.drop_duplicates(inplace=True)
    df.to_csv("data.csv", index=False)
    print("dropped duplicates")
    read_data()

def add_type_col():
    df = pd.read_csv('data.csv')

    # Define subtypes that correspond to "apartment"
    apartment_subtypes = {
        "apartment",
        "ground_floor",
        "duplex",
        "triplex",
        "flat_studio",
        "penthouse",
        "loft",
        "kot",
        "service_flat"
    }

    # Create the "type" column based on the "subtype" column
    df['type'] = df['subtype'].apply(lambda x: 'apartment' if x in apartment_subtypes else 'house')

    # Optional: Save the updated DataFrame to a new CSV
    df.to_csv('cleaned_data.csv', index=False)

def inspect():
    df = pd.read_csv('cleaned_data.csv')
    print(df.info())
    print(df.describe())
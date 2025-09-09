from scraper.src.scraper import Scraper
import pandas as pd


def fetch_data():
    with Scraper() as scraper:
        # scraper.store_links_in_db()
        scraper.fetch_and_store_details_from_db(property_type='house')


if __name__ == '__main__':
    fetch_data()
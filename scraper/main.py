from scraper.src.scraper import Scraper

def fetch_data(property_type="apartment"):
    with Scraper() as scraper:
        scraper.store_links_in_db(property_type=property_type)
        scraper.fetch_and_store_details_from_db(property_type=property_type)


if __name__ == '__main__':
    fetch_data()
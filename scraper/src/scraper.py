import requests
from typing import Optional
from bs4 import BeautifulSoup
import logging

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import time
from random import randint
import re
import cfscrape
from concurrent.futures import ThreadPoolExecutor, as_completed


class Scraper:
    def _fetch_links_for_postal_code(self, postal_code, start_page, max_pages, property_type, headers):
        links = set()
        for page_nr in range(start_page, start_page + max_pages):
            url = self.get_url(page_nr, postal_code, property_type=property_type)
            try:
                response = self.session.get(url, headers=headers, timeout=15)
                html = response.text
                if "Please enable JS" in html:
                    break
                soup = BeautifulSoup(html, "html.parser")
                links_elements = soup.find_all("a", string=re.compile("details", re.IGNORECASE))
                page_links = set()
                for link in links_elements:
                    href = link.get("href")
                    if href:
                        page_links.add(href)
                if not page_links:
                    break
                new_links = page_links - links
                if not new_links:
                    break
                links.update(new_links)
            except Exception:
                break
        return list(links)

    def _extract_property_details(self, url, property_type, key_list):
        try:
            scraper = cfscrape.create_scraper()
            content = scraper.get(url, timeout=15).content
            soup = BeautifulSoup(content, "html.parser")
            p = soup.find("p")
            if p and "Please enable JS and disable any ad blocker" in p.text:
                return None, 'error'
            address_block = soup.find("div", class_="detail__header_address")
            street = address_block.find("span").text if address_block else ""
            price_block = soup.find("span", class_="detail__header_price_data")
            if price_block:
                price = price_block.get_text(strip=True)
                price = int(re.sub(r"[^\d]", "", price))
            else:
                return None, 'error'
            postal_code, city = self.extract_postal_code_and_city(url)
            subtype = self.extract_subtype(url)
            type = property_type
            details = {}
            for data_row in soup.find_all("div", class_="data-row"):
                wrapper = data_row.find("div", class_="data-row-wrapper") if data_row else None
                if not wrapper:
                    continue
                for div in wrapper.find_all("div", recursive=False):
                    key_tag = div.find("h4")
                    value_tag = div.find("p")
                    if key_tag and value_tag:
                        key = key_tag.get_text(strip=True)
                        value = value_tag.get_text(strip=True)
                        if key in key_list:
                            details[key] = value
            result = {k: None for k in key_list}
            result["url"] = url
            result["address"] = street
            result["price"] = price
            result["postal_code"] = postal_code
            result["city"] = city
            result["subtype"] = subtype
            result["type"] = type
            for k, v in details.items():
                result[k] = v
            result["scrape_timestamp"] = datetime.utcnow()
            return result, 'done'
        except Exception as e:
            print(f"❌ Error on {url}: {e}")
            return None, 'error'

    def store_links_in_db(self, start_page=1, max_pages=50, max_workers=5, property_type="apartment", db_url=None):
        """
        Scrape links and store them in a DB table with status and timestamp.
        """
        if db_url is None:
            db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
        engine = create_engine(db_url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        postal_codes = (
            pd.read_csv("src/georef-belgium-postal-codes@public.csv", delimiter=";")["Post code"].astype(str).tolist()
        )
        all_links = []
        now = datetime.now()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_links_for_postal_code, postal_code, start_page, max_pages, property_type, headers): postal_code
                for postal_code in postal_codes
            }
            for future in as_completed(futures):
                links = future.result()
                all_links.extend(links)
        links_df = pd.DataFrame(
            {
                "url": all_links,
                "property_type": property_type,
                "status": "pending",
                "scrape_timestamp": now,
            }
        )
        try:
            existing_links = pd.read_sql_table("links", engine)
            links_df = pd.concat([existing_links, links_df], ignore_index=True)
            links_df = links_df.drop_duplicates(subset=["url"], keep="first")
        except Exception:
            pass
        links_df.to_sql("links", engine, if_exists="replace", index=False)
        print(f"✅ All links saved to DB table 'links'")

    def fetch_and_store_details_from_db(self, property_type="apartment", db_url=None):
        """
        Fetch details for links with status 'pending', store in details table, update link status.
        """
        if db_url is None:
            db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
        engine = create_engine(db_url)
        key_list = [
            "url",
            "address",
            "price",
            "State of the property",
            "Build Year",
            "Availability",
            "Number of bedrooms",
            "Surface bedroom 1",
            "Surface bedroom 2",
            "Surface bedroom 3",
            "Livable surface",
            "Furnished",
            "Surface of living-room",
            "Cellar",
            "Diningroom",
            "Surface of the diningroom",
            "Kitchen equipment",
            "Surface kitchen",
            "Number of bathrooms",
            "Number of showers",
            "Number of toilets",
            "Type of heating",
            "Type of glazing",
            "Entry phone",
            "Elevator",
            "Access for disabled",
            "Orientation of the front facade",
            "Floor of appartment",
            "Number of facades",
            "Number of floors",
            "Garden",
            "Surface garden",
            "Terrace",
            "Surface terrace",
            "Total land surface",
            "Sewer Connection",
            "Gas",
            "Running water",
            "Swimming pool",
            "Specific primary energy consumption",
            "Validity date EPC/PEB",
            "CO2 emission",
            "Certification - Electrical installation",
            "Flooding Area type",
            "Demarcated flooding area",
            "postal_code",
            "city",
        ]
        try:
            links_df = pd.read_sql_query(
                f"SELECT url FROM links WHERE status='pending' AND property_type='{property_type}'",
                engine,
            )
            urls = links_df["url"].tolist()
        except Exception:
            print("No pending links found in DB. Run store_links_in_db first.")
            return
        try:
            existing_df = pd.read_sql_table(f"{property_type}_data", engine)
        except Exception:
            existing_df = pd.DataFrame(columns=key_list)
        def fetch_and_process(url):
            processed_urls = set(existing_df["url"].dropna().unique())
            if url in processed_urls:
                with engine.begin() as conn:
                    conn.execute(f"UPDATE links SET status='done' WHERE url=%s", (url,))
                return
            result, status = self._extract_property_details(url, property_type, key_list)
            if result:
                with engine.begin() as conn:
                    pd.DataFrame([result]).to_sql(f"{property_type}_data", conn, if_exists="append", index=False)
            with engine.begin() as conn:
                conn.execute(f"UPDATE links SET status=%s WHERE url=%s", (status, url))
            return result
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(fetch_and_process, url): url for url in urls}
            for future in as_completed(future_to_url):
                future.result()
        print(f"✅ All details processed and stored for property_type={property_type}")

    @staticmethod
    def extract_postal_code_and_city(url):
        """
        Extracts the postal code and city from a URL like:
        https://immovlan.be/en/detail/apartment/for-sale/1190/vorst/vbd23488
        Returns (postal_code, city) as strings, or (None, None) if not found.
        """
        match = re.search(r"/for-sale/(\d{4})/([\w-]+)/", url)
        if match:
            return match.group(1), match.group(2)
        return None, None

    @staticmethod
    def extract_subtype(url):
        """
        Extracts the subtype from a URL like:
        https://immovlan.be/en/detail/apartment/for-sale/1190/vorst/vbd23488
        Returns subtype as string, or None if not found.
        """
        match = re.search(r"/detail/([\w_]+)/for-sale/", url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def get_url(page_nr, postal_code, property_type="apartment"):
        return f"https://immovlan.be/en/real-estate?transactiontypes=for-sale&propertytypes={property_type}&municipals={postal_code}&page={page_nr}&noindex=1"

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.session: Optional[requests.Session] = None

    def __enter__(self) -> "Scraper":
        self.session = requests.Session()
        logging.info("Session created")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logging.error(f"Exception: {exc_type}, {exc_val}, {exc_tb}")
        if self.session:
            self.session.close()
            logging.info("Session closed")

    def fetch_immo_list(
        self, start_page=1, max_pages=50, max_workers=5, property_type="apartment"
    ):
        """
        For each postal code, fetch pages sequentially until no more results or max_pages is reached.
        Process postal codes concurrently for efficiency.
        Args:
            start_page (int): First page number to fetch (inclusive).
            max_pages (int): Maximum number of pages to fetch per postal code.
            max_workers (int): Number of concurrent threads for postal codes.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        def fetch_pages_for_postal_code(postal_code):
            links = set()
            for page_nr in range(start_page, start_page + max_pages):
                url = self.get_url(page_nr, postal_code, property_type=property_type)
                print(f"Requesting URL: {url}")
                try:
                    response = self.session.get(url, headers=headers, timeout=15)
                    print(f"Page {page_nr} status code: {response.status_code}")
                    html = response.text
                    if "Please enable JS" in html:
                        print(
                            f"Blocked: Cloudflare challenge detected on page {page_nr}."
                        )
                        break
                    soup = BeautifulSoup(html, "html.parser")
                    links_elements = soup.find_all(
                        "a", string=re.compile("details", re.IGNORECASE)
                    )
                    page_links = set()
                    for link in links_elements:
                        href = link.get("href")
                        if href:
                            print(f"Page {page_nr}: Found link: {href}")
                            page_links.add(href)
                    if not page_links:
                        print(
                            f"No more links found for postal code {postal_code} at page {page_nr}. Stopping."
                        )
                        break
                    # Stop if all links are already seen (avoid infinite loop on repeated last page)
                    new_links = page_links - links
                    if not new_links:
                        print(
                            f"No new links found for postal code {postal_code} at page {page_nr}. Stopping."
                        )
                        break
                    links.update(new_links)
                except Exception as e:
                    print(
                        f"Request failed for postal code {postal_code} page {page_nr}: {e}"
                    )
                    break
            return list(links)

        all_links = []
        postal_codes = (
            pd.read_csv("src/georef-belgium-postal-codes@public.csv", delimiter=";")[
                "Post code"
            ]
            .astype(str)
            .tolist()
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_pages_for_postal_code, postal_code): postal_code
                for postal_code in postal_codes
            }
            for future in as_completed(futures):
                links = future.result()
                all_links.extend(links)

        print(f"Total links found: {len(all_links)}")
        pandas_df = pd.DataFrame(all_links, columns=["url"])
        # append links to the existing CSV file, then drop duplicates
        try:
            existing_df = pd.read_csv(f"{property_type}_links.csv")
            all_links_df = pd.concat([existing_df, pandas_df], ignore_index=True)
        except FileNotFoundError:
            print("links.csv not found, creating a new one")
            all_links_df = pandas_df
        # Drop duplicate URLs, keeping the first occurrence
        all_links_df = all_links_df.drop_duplicates(subset=["url"])
        print(f"Saving {len(all_links_df)} unique links to {property_type}_links.csv")
        all_links_df.to_csv(f"{property_type}_links.csv", index=False)

    def fetch_details_soup_multithread(self, property_type="apartment", db_url=None):
        # read from links.csv
        try:
            df = pd.read_csv(f"{property_type}_links.csv")
            # deduplicate urls
            df = df.drop_duplicates(subset=["url"])
            urls = df["url"].tolist()  # limit for testing if needed
        except FileNotFoundError:
            print("links.csv not found, please run fetch_list first")
            return

        key_list = [
            "address",
            "price",
            "State of the property",
            "Build Year",
            "Availability",
            "Number of bedrooms",
            "Surface bedroom 1",
            "Surface bedroom 2",
            "Surface bedroom 3",
            "Livable surface",
            "Furnished",
            "Surface of living-room",
            "Cellar",
            "Diningroom",
            "Surface of the diningroom",
            "Kitchen equipment",
            "Surface kitchen",
            "Number of bathrooms",
            "Number of showers",
            "Number of toilets",
            "Type of heating",
            "Type of glazing",
            "Entry phone",
            "Elevator",
            "Access for disabled",
            "Orientation of the front facade",
            "Floor of appartment",
            "Number of facades",
            "Number of floors",
            "Garden",
            "Surface garden",
            "Terrace",
            "Surface terrace",
            "Total land surface",
            "Sewer Connection",
            "Gas",
            "Running water",
            "Swimming pool",
            "Specific primary energy consumption",
            "Validity date EPC/PEB",
            "CO2 emission",
            "Certification - Electrical installation",
            "Flooding Area type",
            "Demarcated flooding area",
            "postal_code",
            "city",
        ]
        # DB setup
        if db_url is None:
            db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
        engine = create_engine(db_url)
        # Check if table exists, else create empty DataFrame
        try:
            existing_df = pd.read_sql_table(f"{property_type}_data", engine)
        except Exception:
            print(f"{property_type}_data table not found, will create new.")
            existing_df = pd.DataFrame(columns=key_list)

        def fetch_and_process(url):
            processed_urls = set(existing_df["url"].dropna().unique())

            if url in processed_urls:
                print(f"⏩ Skipping already-processed URL: {url}")
                return
            # Skip project detail pages, too much different data structure
            if "projectdetail" in url:
                return

            print(f"Fetching: {url}")
            try:
                scraper = cfscrape.create_scraper()
                content = scraper.get(url, timeout=15).content
                soup = BeautifulSoup(content, "html.parser")

                # Check for fallback page
                p = soup.find("p")
                if p and "Please enable JS and disable any ad blocker" in p.text:
                    print(f"⚠️ JS block detected on {url}")
                    return None

                # Extract address
                address_block = soup.find("div", class_="detail__header_address")
                street = address_block.find("span").text if address_block else ""

                # Extract price
                price_block = soup.find("span", class_="detail__header_price_data")
                if price_block:
                    price = price_block.get_text(strip=True)
                    # parse price to int
                    price = int(re.sub(r"[^\d]", "", price))
                else:
                    raise ValueError("Price not found")

                # Extract postal code and city from URL
                postal_code, city = self.extract_postal_code_and_city(url)

                subtype = self.extract_subtype(url)
                type = property_type

                print(city, street, postal_code, price)
                # Extract all key-value pairs from data-row sections
                details = {}
                for data_row in soup.find_all("div", class_="data-row"):
                    wrapper = (
                        data_row.find("div", class_="data-row-wrapper")
                        if data_row
                        else None
                    )
                    if not wrapper:
                        continue  # Just skip this block
                    for div in wrapper.find_all("div", recursive=False):
                        key_tag = div.find("h4")
                        value_tag = div.find("p")
                        if key_tag and value_tag:
                            key = key_tag.get_text(strip=True)
                            value = value_tag.get_text(strip=True)
                            if key in key_list:
                                details[key] = value

                # Combine all extracted info into one dictionary
                result = {k: None for k in key_list}
                result["url"] = url
                result["address"] = street
                result["price"] = price
                result["postal_code"] = postal_code
                result["city"] = city
                result["subtype"] = subtype
                result["type"] = type
                for k, v in details.items():
                    result[k] = v
                # Add timestamp for versioning
                result["scrape_timestamp"] = datetime.utcnow()
                results.append(result)

            except Exception as e:
                print(f"❌ Error on {url}: {e}")
                return None

        # --- Run multithreaded scraping ---
        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                # Todo: limit to a slice for testing
                executor.submit(fetch_and_process, url): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    results.append(result)
                    print(f"✅ Saved data for: {result['url']}")

        # --- Save results to DB ---
        if results:
            df_result = pd.DataFrame(results)
            # Ensure all columns from key_list + scrape_timestamp are present and in order
            for col in key_list + ["scrape_timestamp"]:
                if col not in df_result.columns:
                    df_result[col] = None
            df_result = df_result[key_list + ["scrape_timestamp"]]
            # Append to DB table
            df_result.to_sql(
                f"{property_type}_data", engine, if_exists="append", index=False
            )
            print(f"✅ All results saved to DB table {property_type}_data")
        else:
            print("⚠️ No results were collected.")

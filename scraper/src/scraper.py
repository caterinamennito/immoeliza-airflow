import requests
from typing import Optional
from bs4 import BeautifulSoup
import logging

import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    DateTime,
    Text,
    UniqueConstraint,
    insert,
    select,
    update,
)
from datetime import datetime
import time
from random import randint
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.db_utils import get_engine
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert


class Scraper:
    # Mapping from scraped keys to DB column names (lower_case_with_underscore)
    SCRAPED_TO_DB = {
        "url": "url",
        "address": "address",
        "price": "price",
        "State of the property": "state_of_the_property",
        "Build Year": "build_year",
        "Availability": "availability",
        "Number of bedrooms": "number_of_bedrooms",
        "Surface bedroom 1": "surface_bedroom_1",
        "Surface bedroom 2": "surface_bedroom_2",
        "Surface bedroom 3": "surface_bedroom_3",
        "Livable surface": "livable_surface",
        "Furnished": "furnished",
        "Surface of living-room": "surface_of_living_room",
        "Cellar": "cellar",
        "Diningroom": "diningroom",
        "Surface of the diningroom": "surface_of_the_diningroom",
        "Kitchen equipment": "kitchen_equipment",
        "Surface kitchen": "surface_kitchen",
        "Number of bathrooms": "number_of_bathrooms",
        "Number of showers": "number_of_showers",
        "Number of toilets": "number_of_toilets",
        "Type of heating": "type_of_heating",
        "Type of glazing": "type_of_glazing",
        "Entry phone": "entry_phone",
        "Elevator": "elevator",
        "Access for disabled": "access_for_disabled",
        "Orientation of the front facade": "orientation_of_the_front_facade",
        "Floor of appartment": "floor_of_appartment",
        "Number of facades": "number_of_facades",
        "Number of floors": "number_of_floors",
        "Garden": "garden",
        "Surface garden": "surface_garden",
        "Terrace": "terrace",
        "Surface terrace": "surface_terrace",
        "Total land surface": "total_land_surface",
        "Sewer Connection": "sewer_connection",
        "Gas": "gas",
        "Running water": "running_water",
        "Swimming pool": "swimming_pool",
        "Specific primary energy consumption": "specific_primary_energy_consumption",
        "Validity date EPC/PEB": "validity_date_epc_peb",
        "CO2 emission": "co2_emission",
        "Certification - Electrical installation": "certification_electrical_installation",
        "Flooding Area type": "flooding_area_type",
        "Demarcated flooding area": "demarcated_flooding_area",
        "postal_code": "postal_code",
        "city": "city",
    }

    # All DB columns (in order)
    KEY_LIST = list(SCRAPED_TO_DB.values())

    @staticmethod
    def get_links_table(metadata, engine=None):
        from sqlalchemy import UniqueConstraint
        table_name = "links"
        if engine is not None:
            inspector = engine.dialect.get_inspector(engine)
            if inspector.has_table(table_name):
                Table(table_name, metadata, autoload_with=engine).drop(engine)
        return Table(
            table_name,
            metadata,
            Column("url", String, primary_key=True),
            Column("property_type", String),
            Column("status", String),
            Column("scrape_timestamp", DateTime),
            UniqueConstraint("url", name="uq_links_url"),
            extend_existing=True,
        )
    
    @classmethod
    def get_details_table(cls, metadata, property_type, engine=None):
        from sqlalchemy import UniqueConstraint
        table_name = "property_details"
        if engine is not None:
            inspector = engine.dialect.get_inspector(engine)
            if inspector.has_table(table_name):
                Table(table_name, metadata, autoload_with=engine).drop(engine)
        return Table(
            table_name,
            metadata,
            Column("url", String, primary_key=True),
            *(Column(k, Text) for k in cls.KEY_LIST if k != "url"),
            Column("type", String),
            Column("scrape_timestamp", DateTime),
            Column("subtype", String),
            UniqueConstraint("url", name="uq_property_details_url"),
            extend_existing=True,
        )

    # Immovlan has a bug where after page 1000, it starts returning the same links again. To avoid this, I added an iteration over postal codes.
    def _fetch_links_for_postal_code(
        self, postal_code, start_page, max_pages, property_type, headers
    ):
        print(f"[START] Postal code {postal_code}")
        links = set()
        for page_nr in range(start_page, start_page + max_pages):
            url = self.get_url(page_nr, postal_code, property_type=property_type)
            print(f"  [Postal {postal_code}] Fetching page {page_nr}: {url}")
            try:
                response = self.session.get(url, headers=headers, timeout=15)
                html = response.text
                if "Please enable JS" in html:
                    print(
                        f"  [Postal {postal_code}] Blocked by JS challenge on page {page_nr}"
                    )
                    break
                soup = BeautifulSoup(html, "html.parser")
                links_elements = soup.find_all(
                    "a", string=re.compile("details", re.IGNORECASE)
                )
                page_links = set()
                for link in links_elements:
                    href = link.get("href")
                    if href and "projectdetail" not in href:
                        page_links.add(href)
                print(
                    f"    [Postal {postal_code}] Found {len(page_links)} links on page {page_nr}"
                )
                if not page_links:
                    print(
                        f"    [Postal {postal_code}] No more links on page {page_nr}, stopping."
                    )
                    break
                new_links = page_links - links
                if not new_links:
                    print(
                        f"    [Postal {postal_code}] No new links on page {page_nr}, stopping."
                    )
                    break
                links.update(new_links)
            except Exception as e:
                print(f"  [Postal {postal_code}] Exception on page {page_nr}: {e}")
                break
        print(f"[DONE] Postal code {postal_code}: {len(links)} links found.")
        return list(links)

    def _extract_property_details(self, url, property_type, key_list=None):
        if key_list is None:
            key_list = self.KEY_LIST
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
            response = requests.get(url, headers=headers, timeout=15)
            content = response.content
            soup = BeautifulSoup(content, "html.parser")
            p = soup.find("p")
            if p and "Please enable JS and disable any ad blocker" in p.text:
                return None, "error"
            address_block = soup.find("div", class_="detail__header_address")
            street = address_block.find("span").text if address_block else ""
            price_block = soup.find("span", class_="detail__header_price_data")
            if price_block:
                price = price_block.get_text(strip=True)
                price = int(re.sub(r"[^\d]", "", price))
            else:
                return None, "error"
            postal_code, city = self.extract_postal_code_and_city(url)
            subtype = self.extract_subtype(url)
            type = property_type
            details = {}
            for data_row in soup.find_all("div", class_="data-row"):
                wrapper = (
                    data_row.find("div", class_="data-row-wrapper")
                    if data_row
                    else None
                )
                if not wrapper:
                    continue
                for div in wrapper.find_all("div", recursive=False):
                    key_tag = div.find("h4")
                    value_tag = div.find("p")
                    if key_tag and value_tag:
                        key = key_tag.get_text(strip=True)
                        value = value_tag.get_text(strip=True)
                        if key in self.SCRAPED_TO_DB:
                            details[self.SCRAPED_TO_DB[key]] = value
            result = {k: None for k in self.KEY_LIST}
            result["url"] = url
            result["address"] = street
            result["price"] = price
            result["postal_code"] = postal_code
            result["city"] = city
            result["subtype"] = subtype
            result["type"] = type
            for k, v in details.items():
                result[k] = v
            result["scrape_timestamp"] = datetime.now()
            return result, "done"
        except Exception as e:
            print(f"❌ Error on {url}: {e}{result}")
            return None, "error"

    def store_links_in_db(
        self,
        start_page=1,
        max_pages=50,
        max_workers=5,
        property_type="apartment",
        db_url=None,
    ):
        """
        Scrape links and store them in a DB table with status and timestamp.
        Table is auto-created if it doesn't exist.
        """

        engine = get_engine(db_url)

        metadata = MetaData()
        links_table = self.get_links_table(metadata)
        metadata.create_all(engine, checkfirst=True)  # auto-create if missing

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        postal_codes = (
            pd.read_csv(
                "scraper/src/georef-belgium-postal-codes@public.csv", delimiter=";"
            )["Post code"]
            .astype(str)
            .tolist()
        )

        all_links = []
        now = datetime.now()
        batch_size = 20

        for i in range(0, len(postal_codes), batch_size):
            batch = postal_codes[i : i + batch_size]
            print(f"\n[Batch {i//batch_size+1}] Processing postal codes: {batch}")
            batch_links = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._fetch_links_for_postal_code,
                        postal_code,
                        start_page,
                        max_pages,
                        property_type,
                        headers,
                    ): postal_code
                    for postal_code in batch
                }
                for future in as_completed(futures):
                    links = future.result()
                    batch_links.extend(links)

            all_links.extend(batch_links)

            # Convert batch results into records
            links_df = pd.DataFrame(
                {
                    "url": batch_links,
                    "property_type": property_type,
                    "status": "pending",
                    "scrape_timestamp": now,
                }
            )
            records = links_df.to_dict(orient="records")

            if not records:
                print(f"[Batch {i//batch_size+1}] ⚠️ No records to insert, skipping")
                continue

            # Insert into DB with dedupe at DB level
            stmt = insert(links_table).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=["url"])

            with engine.begin() as conn:
                conn.execute(stmt)

            print(
                f"[Batch {i//batch_size+1}] ✅ {len(batch_links)} links inserted into 'links'"
            )

        print(f"\n✅ All batches complete. Total links processed: {len(all_links)}")

    def fetch_and_store_details_from_db(self, property_type="apartment", db_url=None):
        """
        Fetch details for links with status 'pending', store in details table, update link status.
        """

        engine = get_engine(db_url)
        metadata = MetaData()
        links_table = self.get_links_table(metadata)
        details_table = self.get_details_table(metadata, property_type)
        metadata.create_all(engine, checkfirst=True)

        # Fetch pending URLs
        with engine.connect() as conn:
            result = conn.execute(
                select(links_table.c.url).where(
                    (links_table.c.status == "pending")
                    & (links_table.c.property_type == property_type)
                )
            )
            urls = [row[0] for row in result]

        if not urls:
            print("⚠️ No pending links found in DB. Run store_links_in_db first.")
            return

        # Fetch already processed URLs
        with engine.connect() as conn:
            try:
                result = conn.execute(select(details_table.c.url))
                processed_urls = {row[0] for row in result}
            except Exception:
                processed_urls = set()

        def fetch_and_process(url):
            if url in processed_urls:
                with engine.begin() as conn:
                    conn.execute(
                        update(links_table)
                        .where(links_table.c.url == url)
                        .values(status="done")
                    )
                return

            # Scrape property details
            result_dict, status = self._extract_property_details(
                url, property_type, self.KEY_LIST
            )

            if result_dict:
                stmt = insert(details_table).values(result_dict)
                stmt = stmt.on_conflict_do_nothing(index_elements=["url"])
                with engine.begin() as conn:
                    conn.execute(stmt)

            print(f"✅ Processed {url} with status {status}")

            # Update link status
            with engine.begin() as conn:
                conn.execute(
                    update(links_table)
                    .where(links_table.c.url == url)
                    .values(status=status)
                )

        # Process in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_and_process, url): url for url in urls}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Error processing {futures[future]}: {e}")

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

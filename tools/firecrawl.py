"""
title: Firecrawl Crawler and Scraper
description: Web crawler and scraper using FireCrawl to fetch web pages
author: Sam McLeod, @signorecello
author_url: https://openwebui.com/u/signorecelloo/
funding_url: https://github.com/signorecello/openwebui-extras
requirements: asyncio, pydantic, bs4, langchain_community, typing, asyncio, firecrawl-py
version: 0.3.0
license: MIT
"""

import re
import logging
from typing import Callable, Any
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from langchain_community.document_loaders import FireCrawlLoader
import asyncio


# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EventEmitter:
    def __init__(
        self, event_emitter: Callable[[dict], Any] = None, show_logs: bool = True
    ):
        """
        :param event_emitter: Function to emit events to the chat.
        :param show_logs: Toggle to enable or disable event emitting (for debugging).
        """
        self.event_emitter = event_emitter
        self.show_logs = show_logs

    async def progress_update(self, description):
        if self.show_logs:
            await self.emit(description)

    async def error_update(self, description):
        if self.show_logs:
            await self.emit(description, "error", True)

    async def success_update(self, description):
        if self.show_logs:
            await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            event_data = {
                "type": "status",
                "data": {
                    "status": status,
                    "description": description,
                    "done": done,
                },
            }

            # Handle both async and sync event emitters
            if asyncio.iscoroutinefunction(self.event_emitter):
                await self.event_emitter(event_data)
            else:
                self.event_emitter(event_data)


class Tools:
    class Valves(BaseModel):
        FIRECRAWL_API_KEY: str | None = Field(
            default="api_key",
            description="Firecrawl API key for web crawling. Can be whatever if using self-hosted instance.",
        )
        FIRECRAWL_API_URL: str | None = Field(
            default="http://firecrawl-api:3002",
            description="Optional Firecrawl API URL for self-hosted instances",
        )
        SHOW_LOGS: bool = Field(
            default=True,
            description="Toggle Event Emitters. If False, no status updates are shown.",
        )
        LIMIT: int = Field(default=50, description="Max number of pages to crawl")
        MAX_DEPTH: int = Field(
            default=2,
            description="Maximum crawling depth for nested pages",
            alias="maxDepth",
        )
        DEFAULT_FORMAT: list[str] = Field(
            default=["markdown"], description="Default format for content return"
        )

    class UserValves(BaseModel):
        CLEAN_CONTENT: bool = Field(
            default=True,
            description="Remove links and image urls from scraped content. This reduces the number of tokens.",
        )

    def __init__(self, valves: Valves = None, user_valves: UserValves = None):
        self.valves = valves or self.Valves()
        self.user_valves = user_valves or self.UserValves()
        logger.debug(f"Initialized Tools with valves: {self.valves}")

    async def web_scrape_async(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        params: dict | None = None,
    ) -> dict:
        """
        Scrape and process a web page asynchronously using FireCrawl.
        Returns a dictionary with content, metadata and any errors.
        """
        logger.debug(f"Starting web scrape for URL: {url}")
        emitter = EventEmitter(__event_emitter__, self.valves.SHOW_LOGS)
        await emitter.progress_update(f"Scraping {url}")
        errors = []

        try:
            # Prepare params with defaults from valves
            scrape_params = {"scrapeOptions": {"formats": self.valves.DEFAULT_FORMAT}}
            # Update with any user-provided params
            if params:
                scrape_params.update(params)
                logger.debug(f"Using custom params for scrape: {scrape_params}")

            logger.debug(
                f"Initializing FireCrawlLoader with API URL: {self.valves.FIRECRAWL_API_URL}"
            )
            loader = FireCrawlLoader(
                api_key=(
                    self.valves.FIRECRAWL_API_KEY
                    if self.valves.FIRECRAWL_API_KEY
                    else "api_key"
                ),
                api_url=(
                    self.valves.FIRECRAWL_API_URL
                    if self.valves.FIRECRAWL_API_URL
                    else None
                ),
                url=url,
                mode="scrape",
                params=scrape_params,
            )

            logger.debug("Starting document load")
            docs = loader.load()
            logger.debug(f"Received {len(docs)} documents")

            content = "\n\n".join([doc.page_content for doc in docs])
            logger.debug(f"Combined content length: {len(content)} characters")

            if self.user_valves.CLEAN_CONTENT:
                logger.debug("Cleaning content of URLs")
                await emitter.progress_update("Received content, cleaning up ...")
                content = clean_urls(content)
                logger.debug(f"Cleaned content length: {len(content)} characters")

            # Extract metadata
            title = extract_title(content)
            links = extract_links(content)
            images = extract_images(content)

            logger.debug(
                f"Extracted metadata - title: {title}, links: {len(links)}, images: {len(images)}"
            )
            await emitter.success_update(
                f"Successfully Scraped {title if title else url}"
            )

            return {
                "url": url,
                "title": title,
                "content": content,
                "links": links,
                "images": images,
                "errors": errors,
            }

        except Exception as e:
            error_msg = f"Error scraping web page {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            await emitter.error_update(error_msg)
            return {
                "url": url,
                "title": None,
                "content": "",
                "links": [],
                "images": [],
                "errors": errors,
            }

    async def crawl_website_async(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
        params: dict | None = None,
    ) -> list[dict]:
        """
        Crawl an entire website and its subpages using FireCrawl.
        Returns a list of dictionaries containing content and metadata for each page.
        """
        logger.debug(f"Starting website crawl for URL: {url}")
        emitter = EventEmitter(__event_emitter__, self.valves.SHOW_LOGS)
        await emitter.progress_update(f"Starting crawl of {url}")
        all_errors = []

        try:
            # Prepare params with defaults
            crawl_params = {
                "limit": self.valves.LIMIT,
                "maxDepth": self.valves.MAX_DEPTH,
                "scrapeOptions": {"formats": self.valves.DEFAULT_FORMAT},
            }
            if params:
                crawl_params.update(params)
                logger.debug(f"Using custom params for crawl: {crawl_params}")

            logger.debug(
                f"Initializing FireCrawlLoader for crawl with params: {crawl_params}"
            )
            loader = FireCrawlLoader(
                api_key=self.valves.FIRECRAWL_API_KEY,
                api_url=(
                    self.valves.FIRECRAWL_API_URL
                    if self.valves.FIRECRAWL_API_URL
                    else None
                ),
                url=url,
                mode="crawl",
                params=crawl_params,
            )

            logger.debug("Starting crawl document load")
            docs = loader.load()
            logger.debug(f"Received {len(docs)} documents from crawl")

            results = []
            for i, doc in enumerate(docs, 1):
                logger.debug(f"Processing document {i}/{len(docs)}")
                content = doc.page_content
                if self.user_valves.CLEAN_CONTENT:
                    content = clean_urls(content)

                source_url = doc.metadata.get("source", "Unknown")
                logger.debug(f"Document {i} source URL: {source_url}")

                # Extract metadata for each page
                title = extract_title(content)
                links = extract_links(content)
                images = extract_images(content)

                results.append(
                    {
                        "url": source_url,
                        "title": title,
                        "content": content,
                        "links": links,
                        "images": images,
                        "errors": [],  # No errors for this page
                    }
                )

            await emitter.success_update(
                f"Successfully crawled {url} and found {len(docs)} pages"
            )
            return results

        except Exception as e:
            error_msg = f"Error crawling website {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            all_errors.append(error_msg)
            await emitter.error_update(error_msg)
            return [
                {
                    "url": url,
                    "title": None,
                    "content": "",
                    "links": [],
                    "images": [],
                    "errors": all_errors,
                }
            ]


def clean_urls(text) -> str:
    """Clean URLs from a string containing structured text."""
    return re.sub(r"\((http[^)]+)\)", "", text)


def extract_title(text) -> str | None:
    """Extract the title from a string containing structured text."""
    match = re.search(r"Title: (.*)\n", text)
    return match.group(1).strip() if match else None


def extract_links(text) -> list[str]:
    """Extract links from a string containing structured text."""
    soup = BeautifulSoup(text, "html.parser")
    return [a.get("href") for a in soup.find_all("a", href=True)]


def extract_images(text) -> list[str]:
    """Extract images from a string containing structured text."""
    soup = BeautifulSoup(text, "html.parser")
    return [img.get("src") for img in soup.find_all("img", src=True)]

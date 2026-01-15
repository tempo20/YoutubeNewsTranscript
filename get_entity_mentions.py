import feedparser
import trafilatura
import spacy
from collections import defaultdict
import re
from pathlib import Path
import csv
# import io
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


CONTEXT_SENTENCES = 1
TICKER_LIST_PATH = Path("tickers.csv")  # optional: columns ticker,name

# Match standalone ticker-like tokens (2–5 uppercase letters, optional leading $),
# avoiding being part of a longer all-caps word or alphanumeric string.
TICKER_RE = re.compile(r"(?<![A-Z0-9])\$?[A-Z]{2,5}(?![A-Z0-9])")
TICKER_STOP = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "CAN", "CO", "FOR",
    "FROM", "HAS", "HAVE", "IN", "IS", "IT", "ITS", "NOT", "OF", "ON", "OR",
    "THE", "TO", "WAS", "WERE", "WILL", "WITH",
}

# Extra tickers we may care about even if not present in tickers.csv
EXTRA_TICKERS = {
    "SPX", "SPY", "QQQ", "DIA", "IWM",  # major indices / ETFs
}

# Simple heuristic to detect ticker-like tokens (used when ORG names are actually tickers)
TICKER_TOKEN_RE = re.compile(r"^[A-Z]{1,5}$")

ENTITY_ALIASES = {
    # companies
    "meta": "META",
    "facebook": "META",

    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "google llc": "GOOGL",

    "apple": "AAPL",
    "apple inc": "AAPL",

    "amazon": "AMZN",
    "amazon.com": "AMZN",

    "microsoft": "MSFT",
    "microsoft corp": "MSFT",

    "chevron": "CVX",
    "chevron corp": "CVX",

    "tesla": "TSLA",
    "tesla motors": "TSLA",

    "netflix": "NFLX",

    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",

    "bank of america": "BAC",
    "bofa": "BAC",

    # institutions
    "fed": "Federal Reserve",
    "federal reserve": "Federal Reserve",
    "doj": "Department of Justice",
    "department of justice": "Department of Justice",
    "supreme court": "Supreme Court",
    "cnn": "CNN",
}

SECTOR_KEYWORDS = {
    "Technology": [
        "tech", "software", "technology", "cloud", "ai", "artificial intelligence",
        "chip", "semiconductor", "digital", "platform", "app", "data", "cyber",
        "saas", "hardware", "server", "datacenter",
    ],
    "Finance": [
        "bank", "financial", "finance", "investment", "trading", "market",
        "stock", "equity", "bond", "credit", "lending", "mortgage",
        "treasury", "yield", "rates", "interest rate", "brokerage",
    ],
    "Healthcare": [
        "health", "medical", "pharmaceutical", "drug", "biotech", "hospital",
        "treatment", "patient", "fda", "clinical", "therapy", "vaccine", "pharma",
    ],
    "Energy": [
        "oil", "gas", "energy", "petroleum", "renewable", "solar", "wind",
        "electric", "power", "fuel", "drilling", "crude", "utility", "utilities",
    ],
    "Retail": [
        "retail", "store", "shopping", "consumer", "e-commerce", "online shopping",
        "merchandise", "sales", "retailer",
    ],
    "Automotive": [
        "car", "automotive", "vehicle", "auto", "truck", "electric vehicle",
        "ev", "manufacturing", "tesla",
    ],
    "Real Estate": [
        "real estate", "property", "housing", "construction", "mortgage",
        "development", "reit",
    ],
    "Telecommunications": [
        "telecom", "communication", "wireless", "5g", "network", "internet",
        "broadband",
    ],
    "Aerospace": [
        "aerospace", "aircraft", "defense", "boeing", "space", "missile",
    ],
    "Consumer Goods": [
        "consumer goods", "packaged goods", "cpg", "beverage", "food",
    ],
}


def debug_video_entities(video) -> None:
    """
    Helper for manual debugging: prints how a single video is parsed into
    tickers, companies, sectors, and scores.
    Not used in the main pipeline; call from a notebook when tuning rules.
    """
    parts = analyze_video_entities_split(video)
    print(f"Title      : {video.get('title', '')}")
    print(f"Title sent.: {video.get('title_sentiment')}")
    print(f"Summary    : {video.get('transcript_summary', '')[:200]}...")
    print(f"Summary sent.: {video.get('transcript_sentiment')}")
    print()
    for part_name, (tickers, companies, sectors, score) in parts.items():
        print(f"[{part_name.upper()}] score={score}")
        print("  Tickers :", sorted(tickers))
        print("  Companies:", sorted(companies))
        print("  Sectors :", sorted(sectors))
        print()

def normalize_company_name(name):
    """
    Normalize company names for mapping:
    - lowercase
    - strip common legal suffixes
    - remove punctuation
    """
    n = (name or "").lower()

    # Strip common suffixes
    suffixes = [
        " inc.", " inc", " corporation", " corp.", " corp", " co.", " co",
        " ltd.", " ltd", " plc", " llc", " holdings", " holding",
    ]
    for suf in suffixes:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break

    n = re.sub(r"[^\w\s]", "", n)
    return n.strip()

def extract_article_text(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_formatting=False
    )
    return text

def load_ticker_map(path: Path):
    ticker_to_name = {}
    name_to_ticker = {}
    if not path.exists():
        return ticker_to_name, name_to_ticker

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            name = (row.get("name") or "").strip()
            if not ticker or not name:
                continue
            ticker_to_name[ticker] = name
            name_to_ticker[normalize_company_name(name)] = ticker

    return ticker_to_name, name_to_ticker


ticker_to_name, name_to_ticker = load_ticker_map(TICKER_LIST_PATH)


def fetch_articles(feed_url, max_items=30):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:max_items]:
        text = extract_article_text(entry.link)
        if not text:
            continue
        articles.append({
            "title": entry.title,
            "url": entry.link,
            "published": entry.get("published"),
            "text": text,
        })
    return articles


def get_tickers(text):
    tickers = set()
    for m in TICKER_RE.findall(text or ""):
        t = m.replace("$", "").upper()
        if t in TICKER_STOP:
            continue
        # If we have an explicit ticker list, require membership or allow a small extra set
        if ticker_to_name and t not in ticker_to_name and t not in EXTRA_TICKERS:
            continue
        tickers.add(t)

    return list(tickers)


def is_ticker(name: str) -> bool:
    """Return True if the string looks like a stock ticker (short all-caps token)."""
    if not name:
        return False
    n = (name or "").upper()
    return bool(TICKER_TOKEN_RE.fullmatch(n))

def is_company_name(name: str) -> bool:
    """Return True if the string looks like a company name (not a ticker)."""
    if not name:
        return False
    # Company names are typically longer than tickers or contain spaces/words
    # If it's not ticker-like, it's likely a company name
    return not is_ticker(name) and name.upper() not in ticker_to_name

def get_companies(doc):
    mapped = []

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        key = normalize_company_name(ent.text)
        if key in name_to_ticker:
            mapped.append(name_to_ticker[key])   # return ticker
        else:
            mapped.append(ent.text)
    return mapped


def get_sectors(text_lower):
    return [
        sector for sector, keywords in SECTOR_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]

def normalize_entity(name: str) -> str:
    if not name:
        return name

    n = name.strip().lower()

    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n)

    if n in ENTITY_ALIASES:
        return ENTITY_ALIASES[n]

    return n.upper() if n.isupper() else n.title()

def sentiment_to_score(sentiment):
    if not sentiment:
        return None
    label = str(sentiment.get('label', '')).upper()
    score = float(sentiment.get('score', 0))
    if 'POS' in label:
        return score
    if 'NEG' in label:
        return -score
    return 0.0

def extract_video_text(video, prefer_summary=True):
    title = video.get('title', '')
    transcript = ''
    if prefer_summary and video.get('transcript_summary'):
        transcript = video['transcript_summary']
    elif video.get('transcript_text'):
        transcript = video['transcript_text']
    combined = f"{title} {transcript}".strip()
    return combined

def analyze_video_entities_split(video):
    title = video.get('title', '') or ''
    summary = video.get('transcript_summary', '') or ''

    title_doc = nlp(title) if title else None
    summary_doc = nlp(summary) if summary else None

    title_tickers = set(get_tickers(title)) if title else set()
    title_companies = set(get_companies(title_doc)) if title_doc else set()
    title_sectors = set(get_sectors(title.lower())) if title else set()

    summary_tickers = set(get_tickers(summary)) if summary else set()
    summary_companies = set(get_companies(summary_doc)) if summary_doc else set()
    summary_sectors = set(get_sectors(summary.lower())) if summary else set()

    title_score = sentiment_to_score(video.get('title_sentiment'))
    summary_score = sentiment_to_score(video.get('transcript_sentiment'))

    return {
        "title": (title_tickers, title_companies, title_sectors, title_score),
        "summary": (summary_tickers, summary_companies, summary_sectors, summary_score),
    }

def aggregate_youtube_entities(videos):

    def new_bucket():
        return {
            "title_mentions": 0,
            "title_scores": [],
            "summary_mentions": 0,
            "summary_scores": [],
        }

    stock_stats = defaultdict(new_bucket)
    company_stats = defaultdict(new_bucket)
    sector_stats = defaultdict(new_bucket)

    for video in videos:
        parts = analyze_video_entities_split(video)

        for part_name, (tickers, companies, sectors, score) in parts.items():

            # Per-part de-duplication so the same logical stock/company
            # isn't counted twice (e.g. both ticker and company name).
            seen_stocks = set()
            seen_companies = set()

            for t in tickers:
                # Tickers come in as symbols; normalize and treat as stocks.
                sym = (t or "").upper()
                if not is_ticker(sym):
                    # Fallback: let normalize_entity handle unexpected forms
                    sym = normalize_entity(t)
                if sym in seen_stocks:
                    continue
                seen_stocks.add(sym)
                stock_stats[sym][f"{part_name}_mentions"] += 1
                if score is not None:
                    stock_stats[sym][f"{part_name}_scores"].append(score)

            for c in companies:
                c = normalize_entity(c)

                # Check company first, then stock:
                # 1) If it's a company name (not ticker-like), treat as company.
                # 2) If it's NOT a company name (i.e., it's ticker-like), treat as stock.
                if is_company_name(c):
                    # It's a company name
                    if c in seen_companies:
                        continue
                    seen_companies.add(c)
                    company_stats[c][f"{part_name}_mentions"] += 1
                    if score is not None:
                        company_stats[c][f"{part_name}_scores"].append(score)
                else:
                    # It's ticker-like, treat as stock
                    sym = c.upper()
                    if sym in seen_stocks:
                        continue
                    seen_stocks.add(sym)
                    stock_stats[sym][f"{part_name}_mentions"] += 1
                    if score is not None:
                        stock_stats[sym][f"{part_name}_scores"].append(score)

            for s in sectors:
                s = normalize_entity(s)
                sector_stats[s][f"{part_name}_mentions"] += 1
                if score is not None:
                    sector_stats[s][f"{part_name}_scores"].append(score)

    def finalize(stats, *, treat_as_stocks: bool = False):
        rows = []
        for name, data in stats.items():
            display_name = name
            if treat_as_stocks:
                # For stocks, display all-lower/TitleCase tickers as uppercase (e.g. cvx -> CVX)
                # Heuristic: short alphabetic token that could be a ticker.
                if re.fullmatch(r"[A-Za-z]{1,5}", name or ""):
                    display_name = name.upper()
            rows.append({
                "name": display_name,

                "title_mentions": data["title_mentions"],
                "avg_title_sentiment": (
                    sum(data["title_scores"]) / len(data["title_scores"])
                    if data["title_scores"] else None
                ),

                "summary_mentions": data["summary_mentions"],
                "avg_summary_sentiment": (
                    sum(data["summary_scores"]) / len(data["summary_scores"])
                    if data["summary_scores"] else None
                ),
            })

        rows.sort(key=lambda x: (x["title_mentions"] + x["summary_mentions"]), reverse=True)
        return rows

    return {
        "stocks": finalize(stock_stats, treat_as_stocks=True),
        "companies": finalize(company_stats),
        "sectors": finalize(sector_stats),
    }
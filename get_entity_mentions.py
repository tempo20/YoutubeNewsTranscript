import feedparser
import trafilatura
import spacy
from collections import Counter, defaultdict
import re
from pathlib import Path
import csv
import math
# import io
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline, logging as transformers_logging
transformers_logging.set_verbosity_error()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer for per-entity sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

CONTEXT_SENTENCES = 2  # Base context window
MIN_CONTEXT_SENTENCES = 2  # Minimum sentences to extract
MAX_CONTEXT_SENTENCES = 3  # Maximum sentences to extract
MIN_CONTEXT_WORDS = 10  # Minimum words needed for meaningful sentiment
MIN_CONTEXT_CHARS = 50  # Minimum characters needed for meaningful sentiment
TICKER_LIST_PATH = Path("tickers.csv")  # optional: columns ticker,name
SP500_TICKER_PATH = Path("500ticker.csv")  # S&P 500 tickers: column Symbol

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

# Ambiguous tickers that look like common English words - require strong evidence
# These are real S&P 500 tickers but often appear as regular words in transcripts
AMBIGUOUS_TICKERS = {
    "SO",    # Southern Company - but also "so" as conjunction
    "ON",    # ON Semiconductor - but also "on" as preposition
    "IN",    # (not S&P 500, but commonly matched)
    "IT",    # Gartner - but also "it" as pronoun
    "AS",    # (not S&P 500, but commonly matched)
    "AT",    # (not S&P 500, but commonly matched)
    "OR",    # (not S&P 500, but commonly matched)
    "BY",    # (not S&P 500, but commonly matched)
    "BE",    # (not S&P 500, but commonly matched)
    "AM",    # Antero Midstream - but also "am" (I am)
    "ARE",   # (not S&P 500, but commonly matched)
    "ALL",   # Allstate - but also "all" as determiner
    "NOW",   # ServiceNow - but also "now" as adverb
    "DAY",   # Dayforce - but also "day" as noun
    "LOW",   # Lowe's - but also "low" as adjective
    "TECH",  # Bio-Techne - but also "tech" as abbreviation for technology
    "WELL",  # Welltower - but also "well" as adverb/interjection
}

# Ambiguous tickers that should ONLY be accepted when company name evidence exists
STRICT_COMPANY_ONLY = {"SO", "NOW", "DAY", "TECH", "LOW", "WELL"}

# Finance-related context words that indicate a ticker mention is likely genuine
# If any of these appear within +-5 tokens of an ambiguous ticker, accept it
FINANCE_CONTEXT_WORDS = {
    # Trading/investing terms
    "stock", "stocks", "share", "shares", "equity", "equities",
    "ticker", "tickers", "symbol", "symbols",
    "buy", "sell", "hold", "long", "short", "trade", "trading",
    "invest", "investor", "investors", "investment",
    # Financial metrics
    "earnings", "revenue", "profit", "loss", "margin", "eps",
    "guidance", "forecast", "outlook", "estimate", "estimates",
    "dividend", "dividends", "yield", "payout",
    "price", "target", "valuation", "pe", "ratio",
    # Market terms
    "market", "markets", "nyse", "nasdaq", "exchange",
    "rally", "surge", "drop", "fall", "rise", "gain", "decline",
    "bullish", "bearish", "upside", "downside",
    # Company/corporate terms
    "ceo", "cfo", "company", "corp", "corporation",
    "quarter", "quarterly", "annual", "fiscal",
    "report", "reports", "reported", "reporting",
    # Analyst terms
    "analyst", "analysts", "upgrade", "downgrade", "rating",
    "overweight", "underweight", "outperform",
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

# Tokens to ignore when building name-based aliases
ALIAS_EXCLUDE_WORDS = {
    "inc", "corp", "corporation", "company", "companies", "ltd", "llc",
    "plc", "holdings", "holding", "group", "the", "and", "of", "class",
}
TOKEN_ALIAS_MIN_LEN = 5

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
    tickers, companies, sectors, and per-entity sentiment scores.
    Not used in the main pipeline; call from a notebook when tuning rules.
    """
    parts = analyze_video_entities_split(video)
    print(f"Title      : {video.get('title', '')}")
    summary = video.get('transcript_summary') or video.get('transcript_text', '')
    if summary:
        print(f"Summary    : {summary[:200]}...")
    else:
        print(f"Summary    : (no summary or transcript)")
    print()
    for part_name, part_data in parts.items():
        print(f"[{part_name.upper()}]")
        tickers = list(part_data.get('tickers', {}).keys())
        companies = list(part_data.get('companies', {}).keys())
        sectors = list(part_data.get('sectors', {}).keys())
        print("  Tickers :", sorted(tickers))
        print("  Companies:", sorted(companies))
        print("  Sectors :", sorted(sectors))
        # Show sample sentiment data
        for entity_type in ['tickers', 'companies', 'sectors']:
            entities = part_data.get(entity_type, {})
            if entities:
                print(f"  {entity_type.capitalize()} sentiments:")
                for entity_name, sentiment_list in list(entities.items())[:3]:  # Show first 3
                    avg_sent = sum(s for _, s, _, _ in sentiment_list) / len(sentiment_list) if sentiment_list else None
                    print(f"    {entity_name}: {avg_sent:.3f} ({len(sentiment_list)} mentions)")
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


def extract_name_tokens(name: str) -> set[str]:
    """
    Extract significant tokens from a company name for aliasing.
    - Lowercase
    - Remove punctuation
    - Exclude common suffixes/stopwords
    - Keep tokens with length >= TOKEN_ALIAS_MIN_LEN
    """
    tokens = set()
    for word in (name or "").lower().split():
        clean = re.sub(r"[^\w]", "", word)
        if len(clean) >= TOKEN_ALIAS_MIN_LEN and clean not in ALIAS_EXCLUDE_WORDS:
            tokens.add(clean)
    return tokens

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
    """Load ticker to name mapping from CSV with ticker,name columns."""
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


def load_sp500_tickers(path: Path):
    """Load S&P 500 tickers from CSV with Symbol column."""
    valid_tickers = set()
    ticker_to_name = {}
    name_to_ticker = {}
    token_counts = Counter()
    rows = []
    
    if not path.exists():
        return valid_tickers, ticker_to_name, name_to_ticker

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("Symbol") or "").strip().upper()
            security = (row.get("Security") or "").strip()
            
            if not symbol:
                continue
            rows.append((symbol, security))
            # Count tokens for uniqueness pass
            for tok in extract_name_tokens(security):
                token_counts[tok] += 1

    for symbol, security in rows:
        valid_tickers.add(symbol)
        if security:
            norm_name = normalize_company_name(security)
            ticker_to_name[symbol] = security
            name_to_ticker[norm_name] = symbol
            # Add unique, significant tokens as aliases when unambiguous
            for tok in extract_name_tokens(security):
                if token_counts[tok] == 1 and tok not in name_to_ticker:
                    name_to_ticker[tok] = symbol

    return valid_tickers, ticker_to_name, name_to_ticker


# Load S&P 500 tickers (primary source for stock filtering)
sp500_tickers, sp500_ticker_to_name, sp500_name_to_ticker = load_sp500_tickers(SP500_TICKER_PATH)

# Also load from tickers.csv if it exists (for backward compatibility)
ticker_to_name, name_to_ticker = load_ticker_map(TICKER_LIST_PATH)

# Merge the mappings (S&P 500 takes precedence)
if sp500_ticker_to_name:
    ticker_to_name.update(sp500_ticker_to_name)
if sp500_name_to_ticker:
    name_to_ticker.update(sp500_name_to_ticker)


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
    """
    Extract tickers from text, only including those from S&P 500 list (500ticker.csv).
    Applies multi-layer validation for ambiguous tickers (SO, ON, IT, etc.).
    """
    if not text:
        return []
    
    tickers = set()
    for match in TICKER_RE.finditer(text):
        t = match.group().replace("$", "").upper()
        if t in TICKER_STOP:
            continue
        # Only include tickers from S&P 500 list (500ticker.csv)
        # Also allow EXTRA_TICKERS (indices/ETFs) if needed
        if t not in sp500_tickers and t not in EXTRA_TICKERS:
            continue
        # Validate ambiguous tickers (require finance context or company name evidence)
        if not is_valid_ticker_mention(text, match.start(), match.end(), t):
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


def get_company_name_tokens(ticker: str) -> set:
    """
    Get significant tokens from a company name for evidence matching.
    Returns lowercase tokens that are length >= 5 and not common suffixes.
    """
    company_name = ticker_to_name.get(ticker, "")
    if not company_name:
        return set()
    
    # Common suffixes to exclude
    exclude_words = {
        "inc", "corp", "corporation", "company", "companies", "ltd", "llc",
        "plc", "holdings", "holding", "group", "the", "and", "of"
    }
    
    tokens = set()
    for word in company_name.lower().split():
        # Remove punctuation
        clean_word = re.sub(r"[^\w]", "", word)
        # Only keep significant words (length >= 5, not a common suffix)
        if len(clean_word) >= 5 and clean_word not in exclude_words:
            tokens.add(clean_word)
    
    return tokens


def has_finance_context_nearby(text: str, match_start: int, match_end: int, window_tokens: int = 5, buffer_chars: int = 200) -> bool:
    """
    Check if any finance-related context words appear within ±window_tokens of the match position.
    
    Args:
        text: The full text
        match_start: Start position of the ticker match
        match_end: End position of the ticker match
        window_tokens: Number of tokens before/after to check (default 5)
    
    Returns:
        True if finance context words found nearby, False otherwise
    """
    if not text:
        return False
    
    # Simple tokenization: split on non-alphanumeric characters
    # Get text around the match with some buffer (shorter buffer for stricter matching if desired)
    start_pos = max(0, match_start - buffer_chars)
    end_pos = min(len(text), match_end + buffer_chars)
    context_text = text[start_pos:end_pos].lower()
    
    # Tokenize the context
    tokens = re.findall(r"[a-z]+", context_text)
    
    # Find the approximate position of our match in the tokens
    # We need to find tokens near our match position
    match_text_lower = text[match_start:match_end].lower()
    
    # Find all occurrences of the ticker in the token list
    match_indices = []
    for i, token in enumerate(tokens):
        if token == match_text_lower or token == match_text_lower.replace("$", ""):
            match_indices.append(i)
    
    # If we can't find the match in tokens, check all tokens in context
    if not match_indices:
        # Fallback: just check if any finance word is in the context
        for token in tokens:
            if token in FINANCE_CONTEXT_WORDS:
                return True
        return False
    
    # Check tokens within window of each match
    for match_idx in match_indices:
        start_window = max(0, match_idx - window_tokens)
        end_window = min(len(tokens), match_idx + window_tokens + 1)
        window_tokens_list = tokens[start_window:end_window]
        
        for token in window_tokens_list:
            if token in FINANCE_CONTEXT_WORDS:
                return True
    
    return False


def has_company_name_evidence(text: str, ticker: str) -> bool:
    """
    Check if the company name (or significant tokens from it) appears in the text.
    
    Args:
        text: The full text to search
        ticker: The ticker symbol to find company name for
    
    Returns:
        True if company name evidence found, False otherwise
    """
    if not text or not ticker:
        return False
    
    text_lower = text.lower()
    
    # Get the full company name
    company_name = ticker_to_name.get(ticker, "")
    if company_name:
        # Check for full company name (normalized)
        normalized_name = normalize_company_name(company_name)
        if normalized_name and normalized_name in text_lower:
            return True
        
        # Check for significant tokens from company name
        name_tokens = get_company_name_tokens(ticker)
        for token in name_tokens:
            if token in text_lower:
                return True
    
    return False


def is_valid_ticker_mention(text: str, match_start: int, match_end: int, ticker: str) -> bool:
    """
    Validate whether a ticker mention is likely genuine or just a common word.
    
    For non-ambiguous tickers (AAPL, MSFT, etc.): always returns True.
    For ambiguous tickers (SO, ON, IT, etc.): requires evidence:
        - Finance-related words within ±5 tokens, OR
        - Company name appears somewhere in the text
    
    Args:
        text: The full text containing the ticker
        match_start: Start position of the ticker match
        match_end: End position of the ticker match  
        ticker: The ticker symbol (uppercase)
    
    Returns:
        True if the ticker mention should be accepted, False if it should be rejected
    """
    # Non-ambiguous tickers are always valid
    if ticker not in AMBIGUOUS_TICKERS:
        return True
    
    # For ambiguous tickers, require evidence
    # Some noisy ambiguous tickers must have company name evidence only
    if ticker in STRICT_COMPANY_ONLY:
        return has_company_name_evidence(text, ticker)
    
    # Default ambiguous logic: finance context (narrower window) OR company evidence
    if has_finance_context_nearby(text, match_start, match_end, window_tokens=3, buffer_chars=120):
        return True
    
    if has_company_name_evidence(text, ticker):
        return True
    
    # No evidence found - reject this ambiguous ticker
    return False


def get_companies(doc):
    """
    Extract company names from spaCy doc, mapping known companies to tickers.
    Guards against ambiguous tickers: only maps to an ambiguous ticker if the
    ORG text actually matches/contains the company name tokens.
    """
    mapped = []

    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        key = normalize_company_name(ent.text)
        if key in name_to_ticker:
            ticker = name_to_ticker[key]
            # Guard for ambiguous tickers: only accept if ORG text matches company name
            if ticker in AMBIGUOUS_TICKERS:
                # Check if the ORG text actually contains company name tokens
                # (not just a coincidental match like "SO" being tagged as ORG)
                name_tokens = get_company_name_tokens(ticker)
                ent_text_lower = ent.text.lower()
                has_name_evidence = any(token in ent_text_lower for token in name_tokens)
                if not has_name_evidence:
                    # The ORG entity doesn't look like the company name, skip mapping
                    mapped.append(ent.text)  # Keep as company name, not ticker
                    continue
            mapped.append(ticker)
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
    """
    Convert FinBERT sentiment result to numeric score.
    FinBERT returns: 'positive', 'negative', or 'neutral' (lowercase)
    """
    if not sentiment:
        return None
    label = str(sentiment.get('label', '')).lower().strip()
    score = float(sentiment.get('score', 0))
    
    if label == 'positive':
        return score  # Return positive score (0.0 to 1.0)
    elif label == 'negative':
        return -score  # Return negative score (-1.0 to 0.0)
    elif label == 'neutral':
        # For neutral, return a small value based on confidence
        # High confidence neutral (score > 0.8) = 0.0
        # Lower confidence might indicate mixed sentiment
        if score > 0.8:
            return 0.0
        else:
            # Low confidence neutral might be slightly positive or negative
            # Return a small value scaled by (1 - confidence)
            return (1.0 - score) * 0.1  # Small positive bias for low-confidence neutral
    else:
        # Unknown label, default to 0.0
        return 0.0


def get_sentence_boundaries(text):
    """
    Split text into sentences using spaCy.
    Returns list of (sentence_text, start_char, end_char) tuples.
    """
    if not text or not text.strip():
        return []
    
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentences.append((sent.text, sent.start_char, sent.end_char))
    return sentences


def extract_entity_contexts(text, entity_mentions, context_sentences=CONTEXT_SENTENCES):
    """
    Extract context around each entity mention with dynamic window sizing.
    Starts with base context_sentences, expands up to MAX_CONTEXT_SENTENCES if needed.
    
    Args:
        text: The text to search in
        entity_mentions: List of (entity_name, start_pos, end_pos) tuples or 
                       list of entity names (will search for positions)
        context_sentences: Base number of sentences before and after to include
    
    Returns:
        List of (entity_name, context_text, mention_position, context_length) tuples
    """
    if not text or not entity_mentions:
        return []
    
    # Split text into sentences with positions
    sentences = get_sentence_boundaries(text)
    if not sentences:
        # If sentence splitting fails, use entire text as fallback
        # This handles very short texts or edge cases
        entity_contexts = []
        for entity_info in entity_mentions:
            if isinstance(entity_info, tuple) and len(entity_info) >= 3:
                entity_name = entity_info[0]
            else:
                entity_name = str(entity_info)
            entity_contexts.append((entity_name, text, 0, len(text)))
        return entity_contexts
    
    # Find which sentence each entity mention is in
    entity_contexts = []
    processed_mentions = set()  # Track (entity, sentence_idx) to avoid duplicates
    
    for entity_info in entity_mentions:
        if isinstance(entity_info, tuple) and len(entity_info) >= 3:
            entity_name, start_pos, end_pos = entity_info[0], entity_info[1], entity_info[2]
            # Validate positions
            if start_pos < 0 or end_pos > len(text) or start_pos >= end_pos:
                continue
        else:
            # If just entity name, find all occurrences
            entity_name = str(entity_info)
            if not entity_name:
                continue
            # Find all positions of entity in text (case-insensitive)
            pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            # Use first match for now (could be extended to handle all matches)
            start_pos = matches[0].start()
            end_pos = matches[0].end()
        
        # Find which sentence contains this mention
        mention_sentence_idx = None
        for idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
            if sent_start <= start_pos < sent_end:
                mention_sentence_idx = idx
                break
        
        # If mention not found in any sentence, use entire text as fallback
        if mention_sentence_idx is None:
            entity_contexts.append((entity_name, text, 0, len(text)))
            continue
        
        # Check if we've already processed this entity in this sentence
        mention_key = (entity_name.upper(), mention_sentence_idx)
        if mention_key in processed_mentions:
            continue
        processed_mentions.add(mention_key)
        
        # Dynamic context window: start with base, expand if needed
        current_window = context_sentences
        context_text = None
        context_length = 0
        
        # Try expanding window from MIN to MAX until we get sufficient context
        for window_size in range(MIN_CONTEXT_SENTENCES, MAX_CONTEXT_SENTENCES + 1):
            start_sentence = max(0, mention_sentence_idx - window_size)
            end_sentence = min(len(sentences), mention_sentence_idx + window_size + 1)
            
            context_sentences_list = sentences[start_sentence:end_sentence]
            candidate_text = " ".join([sent[0] for sent in context_sentences_list])
            candidate_words = len(candidate_text.split())
            candidate_chars = len(candidate_text.strip())
            
            # Check if this window size provides sufficient context
            if candidate_words >= MIN_CONTEXT_WORDS and candidate_chars >= MIN_CONTEXT_CHARS:
                context_text = candidate_text
                context_length = candidate_chars
                current_window = window_size
                break
        
        # If no window size worked, use full text as fallback
        if context_text is None or len(context_text.strip()) < MIN_CONTEXT_CHARS:
            context_text = text
            context_length = len(context_text)
            current_window = MAX_CONTEXT_SENTENCES  # Mark as using full text
        
        entity_contexts.append((entity_name, context_text, mention_sentence_idx, context_length))
    
    return entity_contexts


def analyze_entity_sentiment(context_text):
    """
    Analyze sentiment for a specific entity context using FinBERT.
    This is the only sentiment analysis performed - there is no video-level sentiment.
    
    Args:
        context_text: The context text around an entity mention
    
    Returns:
        Sentiment score (float) or None if analysis fails
        - Positive sentiment: returns positive float (0.0 to 1.0)
        - Negative sentiment: returns negative float (-1.0 to 0.0)
        - Neutral sentiment: returns 0.0 or small value
    """
    if not context_text or not context_text.strip():
        return None
    
    # Ensure minimum context length for meaningful sentiment analysis
    context_text = context_text.strip()
    if len(context_text) < 10:
        # Too short, likely not meaningful
        return None
    
    try:
        # Truncate to max length for sentiment analyzer (512 chars)
        # But ensure we have enough context
        truncated = context_text[:512] if len(context_text) > 512 else context_text
        
        # If context is very short, try to expand it or skip
        if len(truncated.split()) < 3:
            return None
        
        result = sentiment_analyzer(truncated)[0]
        return sentiment_to_score(result)
    except Exception as e:
        # Fallback: return None if sentiment analysis fails
        return None

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
    """
    Analyze entities in video title and transcript, extracting per-entity sentiment.
    Uses transcript_summary if available, otherwise falls back to transcript_text.
    
    Returns:
        Dictionary with 'title' and 'summary' keys, each containing:
        {
            'tickers': {ticker: [(context, sentiment, position, context_length), ...]},
            'companies': {company: [(context, sentiment, position, context_length), ...]},
            'sectors': {sector: [(context, sentiment, position, context_length), ...]}
        }
    """
    title = video.get('title', '') or ''
    # Use transcript_summary if available, otherwise use transcript_text
    summary = video.get('transcript_summary', '') or video.get('transcript_text', '') or ''

    title_doc = nlp(title) if title else None
    summary_doc = nlp(summary) if summary else None

    # Extract entities with positions
    def extract_tickers_with_positions(text):
        """
        Extract tickers with their positions in text. Only includes S&P 500 tickers.
        Applies multi-layer validation for ambiguous tickers (SO, ON, IT, etc.).
        """
        tickers_with_pos = []
        if not text:
            return tickers_with_pos
        for match in TICKER_RE.finditer(text):
            t = match.group().replace("$", "").upper()
            if t in TICKER_STOP:
                continue
            # Only include tickers from S&P 500 list (500ticker.csv)
            # Also allow EXTRA_TICKERS (indices/ETFs) if needed
            if t not in sp500_tickers and t not in EXTRA_TICKERS:
                continue
            # Validate ambiguous tickers (require finance context or company name evidence)
            if not is_valid_ticker_mention(text, match.start(), match.end(), t):
                continue
            tickers_with_pos.append((t, match.start(), match.end()))
        return tickers_with_pos

    def extract_companies_with_positions(doc, text):
        """
        Extract companies with their positions in text.
        Guards against ambiguous tickers: only maps to an ambiguous ticker if the
        ORG text actually matches/contains the company name tokens.
        """
        companies_with_pos = []
        if not doc:
            return companies_with_pos
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            key = normalize_company_name(ent.text)
            if key in name_to_ticker:
                ticker = name_to_ticker[key]
                # Guard for ambiguous tickers: only accept if ORG text matches company name
                if ticker in AMBIGUOUS_TICKERS:
                    # Check if the ORG text actually contains company name tokens
                    name_tokens = get_company_name_tokens(ticker)
                    ent_text_lower = ent.text.lower()
                    has_name_evidence = any(token in ent_text_lower for token in name_tokens)
                    if not has_name_evidence:
                        # The ORG entity doesn't look like the company name, keep as company
                        companies_with_pos.append((ent.text, ent.start_char, ent.end_char))
                        continue
                mapped_name = ticker
            else:
                mapped_name = ent.text
            companies_with_pos.append((mapped_name, ent.start_char, ent.end_char))
        return companies_with_pos

    def extract_sectors_with_positions(text_lower, text):
        """Extract sectors - for sectors, we use the whole text as context."""
        sectors_found = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    # For sectors, we'll use the entire text as context
                    # since sector keywords might appear multiple times
                    # Use a special marker position to indicate full text context
                    sectors_found.append((sector, -1, -1))  # -1 indicates full text
                    break  # Only add each sector once per text
        return sectors_found

    # Extract entities with positions
    title_tickers_pos = extract_tickers_with_positions(title) if title else []
    title_companies_pos = extract_companies_with_positions(title_doc, title) if title_doc else []
    title_sectors_pos = extract_sectors_with_positions(title.lower(), title) if title else []

    summary_tickers_pos = extract_tickers_with_positions(summary) if summary else []
    summary_companies_pos = extract_companies_with_positions(summary_doc, summary) if summary_doc else []
    summary_sectors_pos = extract_sectors_with_positions(summary.lower(), summary) if summary else []

    # Extract contexts and analyze sentiment for each entity
    def process_entities(entity_positions, text, part_name, is_sectors=False):
        """Process entities: extract contexts and analyze sentiment."""
        entity_sentiments = defaultdict(list)
        
        if not entity_positions or not text:
            return entity_sentiments
        
        # For sectors, use full text as context (they're keyword-based, not position-based)
        if is_sectors:
            for entity_name, start_pos, end_pos in entity_positions:
                # Use entire text as context for sectors
                context_text = text
                context_length = len(context_text)
                sentiment = analyze_entity_sentiment(context_text)
                if sentiment is not None:
                    entity_sentiments[entity_name].append((context_text, sentiment, 0, context_length))
                # If sentiment analysis fails, skip this entity mention (no fallback)
        else:
            # For tickers and companies, extract surrounding sentence contexts
            # Extract contexts for all entity mentions
            contexts = extract_entity_contexts(text, entity_positions, context_sentences=CONTEXT_SENTENCES)
            
            # Analyze sentiment for each context
            for entity_name, context_text, position, context_length in contexts:
                sentiment = analyze_entity_sentiment(context_text)
                if sentiment is not None:
                    entity_sentiments[entity_name].append((context_text, sentiment, position, context_length))
                # If sentiment analysis fails, skip this entity mention (no fallback)
        
        return entity_sentiments

    # Process title entities
    title_tickers_sent = process_entities(title_tickers_pos, title, 'title', is_sectors=False)
    title_companies_sent = process_entities(title_companies_pos, title, 'title', is_sectors=False)
    title_sectors_sent = process_entities(title_sectors_pos, title, 'title', is_sectors=True)

    # Process summary entities
    summary_tickers_sent = process_entities(summary_tickers_pos, summary, 'transcript', is_sectors=False)
    summary_companies_sent = process_entities(summary_companies_pos, summary, 'transcript', is_sectors=False)
    summary_sectors_sent = process_entities(summary_sectors_pos, summary, 'transcript', is_sectors=True)

    return {
        "title": {
            'tickers': dict(title_tickers_sent),
            'companies': dict(title_companies_sent),
            'sectors': dict(title_sectors_sent),
        },
        "summary": {
            'tickers': dict(summary_tickers_sent),
            'companies': dict(summary_companies_sent),
            'sectors': dict(summary_sectors_sent),
        },
    }

def aggregate_youtube_entities(videos):

    def new_bucket():
        return {
            "title_mentions": 0,
            "title_sentiment_data": [],  # List of (sentiment, weight) tuples
            "summary_mentions": 0,
            "summary_sentiment_data": [],  # List of (sentiment, weight) tuples
        }

    stock_stats = defaultdict(new_bucket)
    company_stats = defaultdict(new_bucket)
    sector_stats = defaultdict(new_bucket)

    # Weight factors: title mentions are weighted higher than summary mentions
    TITLE_WEIGHT = 2.0
    SUMMARY_WEIGHT = 1.0

    for video in videos:
        parts = analyze_video_entities_split(video)

        for part_name, part_data in parts.items():
            # part_data is now a dict with 'tickers', 'companies', 'sectors'
            # Each contains {entity_name: [(context, sentiment, position, context_length), ...]}

            # Per-part de-duplication so the same logical stock/company
            # isn't counted twice (e.g. both ticker and company name).
            seen_stocks = set()
            seen_companies = set()

            # Process tickers
            for ticker_name, sentiment_list in part_data.get('tickers', {}).items():
                sym = (ticker_name or "").upper()
                if not is_ticker(sym):
                    sym = normalize_entity(ticker_name)
                if sym in seen_stocks:
                    continue
                seen_stocks.add(sym)
                
                stock_stats[sym][f"{part_name}_mentions"] += len(sentiment_list)
                # Add sentiment data with weights
                for context_text, sentiment, position, context_length in sentiment_list:
                    # Weight by context length (longer contexts may be more reliable)
                    # and by part type (title vs summary)
                    base_weight = TITLE_WEIGHT if part_name == "title" else SUMMARY_WEIGHT
                    # Normalize context_length weight (use log scale to avoid extreme weights)
                    length_weight = math.log(max(context_length, 10)) / math.log(100)  # Normalize to ~0-1 range
                    weight = base_weight * (1 + 0.5 * length_weight)  # Add 50% bonus for longer contexts
                    stock_stats[sym][f"{part_name}_sentiment_data"].append((sentiment, weight))

            # Process companies
            for company_name, sentiment_list in part_data.get('companies', {}).items():
                c = normalize_entity(company_name)

                # Check company first, then stock:
                # 1) If it's a company name (not ticker-like), treat as company.
                # 2) If it's ticker-like AND in S&P 500, treat as stock.
                # 3) If it's ticker-like but NOT in S&P 500, treat as company (e.g., "CNBC").
                sym = c.upper()
                if is_company_name(c):
                    # It's a company name
                    if c in seen_companies:
                        continue
                    seen_companies.add(c)
                    company_stats[c][f"{part_name}_mentions"] += len(sentiment_list)
                    for context_text, sentiment, position, context_length in sentiment_list:
                        base_weight = TITLE_WEIGHT if part_name == "title" else SUMMARY_WEIGHT
                        length_weight = math.log(max(context_length, 10)) / math.log(100)
                        weight = base_weight * (1 + 0.5 * length_weight)
                        company_stats[c][f"{part_name}_sentiment_data"].append((sentiment, weight))
                elif sym in sp500_tickers or sym in EXTRA_TICKERS:
                    # It's ticker-like AND a valid S&P 500 ticker, treat as stock
                    if sym in seen_stocks:
                        continue
                    seen_stocks.add(sym)
                    stock_stats[sym][f"{part_name}_mentions"] += len(sentiment_list)
                    for context_text, sentiment, position, context_length in sentiment_list:
                        base_weight = TITLE_WEIGHT if part_name == "title" else SUMMARY_WEIGHT
                        length_weight = math.log(max(context_length, 10)) / math.log(100)
                        weight = base_weight * (1 + 0.5 * length_weight)
                        stock_stats[sym][f"{part_name}_sentiment_data"].append((sentiment, weight))
                else:
                    # It's ticker-like but NOT in S&P 500 (e.g., "CNBC", "GLP")
                    # Treat as company instead
                    if c in seen_companies:
                        continue
                    seen_companies.add(c)
                    company_stats[c][f"{part_name}_mentions"] += len(sentiment_list)
                    for context_text, sentiment, position, context_length in sentiment_list:
                        base_weight = TITLE_WEIGHT if part_name == "title" else SUMMARY_WEIGHT
                        length_weight = math.log(max(context_length, 10)) / math.log(100)
                        weight = base_weight * (1 + 0.5 * length_weight)
                        company_stats[c][f"{part_name}_sentiment_data"].append((sentiment, weight))

            # Process sectors
            for sector_name, sentiment_list in part_data.get('sectors', {}).items():
                s = normalize_entity(sector_name)
                sector_stats[s][f"{part_name}_mentions"] += len(sentiment_list)
                for context_text, sentiment, position, context_length in sentiment_list:
                    base_weight = TITLE_WEIGHT if part_name == "title" else SUMMARY_WEIGHT
                    length_weight = math.log(max(context_length, 10)) / math.log(100)
                    weight = base_weight * (1 + 0.5 * length_weight)
                    sector_stats[s][f"{part_name}_sentiment_data"].append((sentiment, weight))

    def finalize(stats, *, treat_as_stocks: bool = False):
        rows = []
        for name, data in stats.items():
            display_name = name
            if treat_as_stocks:
                # For stocks, display all-lower/TitleCase tickers as uppercase (e.g. cvx -> CVX)
                # Heuristic: short alphabetic token that could be a ticker.
                if re.fullmatch(r"[A-Za-z]{1,5}", name or ""):
                    display_name = name.upper()
            
            # Calculate weighted average for title sentiment
            title_sentiment = None
            if data["title_sentiment_data"]:
                total_weighted = sum(sentiment * weight for sentiment, weight in data["title_sentiment_data"])
                total_weight = sum(weight for _, weight in data["title_sentiment_data"])
                if total_weight > 0:
                    title_sentiment = total_weighted / total_weight

            # Calculate weighted average for summary sentiment
            summary_sentiment = None
            if data["summary_sentiment_data"]:
                total_weighted = sum(sentiment * weight for sentiment, weight in data["summary_sentiment_data"])
                total_weight = sum(weight for _, weight in data["summary_sentiment_data"])
                if total_weight > 0:
                    summary_sentiment = total_weighted / total_weight

            rows.append({
                "name": display_name,
                "title_mentions": data["title_mentions"],
                "avg_title_sentiment": round(title_sentiment, 4) if title_sentiment is not None else None,
                "summary_mentions": data["summary_mentions"],
                "avg_summary_sentiment": round(summary_sentiment, 4) if summary_sentiment is not None else None,
            })

        rows.sort(key=lambda x: (x["title_mentions"] + x["summary_mentions"]), reverse=True)
        return rows

    return {
        "stocks": finalize(stock_stats, treat_as_stocks=True),
        "companies": finalize(company_stats),
        "sectors": finalize(sector_stats),
    }
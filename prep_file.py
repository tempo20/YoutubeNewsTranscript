import json
from pathlib import Path

def filter_result(result, min_mentions=1, min_sentiment_threshold=0.01):
    """
    Filter result to only include items with:
    - Total mentions > min_mentions, OR
    - Absolute sentiment > min_sentiment_threshold
    
    Parameters:
    -----------
    result : dict
        The result dictionary with stocks, companies, sectors
    min_mentions : int
        Minimum total mentions required (default: 1)
    min_sentiment_threshold : float
        Minimum absolute sentiment value (default: 0.01)
    
    Returns:
    --------
    dict : Filtered result with same structure
    """
    def should_include(item):
        total_mentions = item.get("title_mentions", 0) + item.get("summary_mentions", 0)
        
        # Check mentions
        if total_mentions > min_mentions:
            return True
        
        # Check sentiment (handle None values)
        title_sent = item.get("avg_title_sentiment")
        summary_sent = item.get("avg_summary_sentiment")
        
        if title_sent is not None and abs(title_sent) > min_sentiment_threshold:
            return True
        
        if summary_sent is not None and abs(summary_sent) > min_sentiment_threshold:
            return True
        
        return False
    
    filtered = {}
    for category in ["stocks", "companies", "sectors"]:
        if category in result:
            filtered[category] = [item for item in result[category] if should_include(item)]
        else:
            filtered[category] = []
    
    return filtered

def convert_entity_mentions_to_text(json_path="entity_mentions.json", output_path=None, print_output=True):
    """
    Convert entity_mentions.json to a readable text file.
    
    Parameters:
    -----------
    json_path : str or Path
        Path to the input JSON file (default: "entity_mentions.json")
    output_path : str or Path, optional
        Path to the output text file. If None, uses json_path with .txt extension
    print_output : bool
        Whether to print the output to console (default: True)
    
    Returns:
    --------
    str : The formatted text output
    """
    
    def format_sentiment(score):
        """Format sentiment score for display."""
        if score is None:
            return "N/A"
        return f"{score:+.4f}"

    def format_mentions(item):
        """Format a single item's mention data."""
        total_mentions = item.get("title_mentions", 0) + item.get("summary_mentions", 0)
        title_sent = format_sentiment(item.get("avg_title_sentiment"))
        summary_sent = format_sentiment(item.get("avg_summary_sentiment"))
        
        lines = [
            f"  Name: {item['name']}",
            f"  Total Mentions: {total_mentions}",
            f"    - Title Mentions: {item.get('title_mentions', 0)} (Sentiment: {title_sent})",
            f"    - Summary Mentions: {item.get('summary_mentions', 0)} (Sentiment: {summary_sent})"
        ]
        return "\n".join(lines)
    
    # Load the JSON data
    json_path = Path(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build the output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("ENTITY MENTIONS REPORT")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Stocks Section
    output_lines.append("STOCKS")
    output_lines.append("-" * 80)
    if data.get("stocks"):
        for i, stock in enumerate(data["stocks"], 1):
            output_lines.append(f"\n{i}. {format_mentions(stock)}")
    else:
        output_lines.append("  No stocks found.")
    output_lines.append("")
    output_lines.append("")

    # Companies Section
    output_lines.append("COMPANIES")
    output_lines.append("-" * 80)
    if data.get("companies"):
        for i, company in enumerate(data["companies"], 1):
            output_lines.append(f"\n{i}. {format_mentions(company)}")
    else:
        output_lines.append("  No companies found.")
    output_lines.append("")
    output_lines.append("")

    # Sectors Section
    output_lines.append("SECTORS")
    output_lines.append("-" * 80)
    if data.get("sectors"):
        for i, sector in enumerate(data["sectors"], 1):
            output_lines.append(f"\n{i}. {format_mentions(sector)}")
    else:
        output_lines.append("  No sectors found.")

    output_lines.append("")
    output_lines.append("=" * 80)

    # Join all lines
    output_text = "\n".join(output_lines)
    
    # Print if requested
    if print_output:
        print(output_text)
    
    # Save to file if output_path is provided
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Successfully saved to {output_path}")
    else:
        # Auto-generate output path from input path
        output_path = json_path.with_suffix('.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Successfully saved to {output_path}")
    
    return output_text
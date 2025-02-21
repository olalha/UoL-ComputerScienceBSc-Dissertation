import openpyxl
from pathlib import Path
from typing import Optional

from utils.settings_manager import get_setting

def parse_rulebook_excel(rulebook_name: str, collection_mode: str) -> Optional[dict]:
    """
    Parses an Excel workbook defining the topic sentiment distribution for a text corpuse
    with additional parameters. This Excel workbook needs to follow the structure of the
    template workbook.
    
    The work can be formatted in one of two ways based on the collection_mode parameter:
    - "word": Defines exact word count ranges for each topic-sentiment pair.
    - "chunk": Defines number of chunks for each topic-sentiment pair.

    This function loads an Excel workbook from the specified path and extracts:
      - 'review_item': from cell A1
      - 'total': from cell F1 (must be an int > 0)
        - either defines total word count or number of chunks depending on collection_mode
      - 'content_rules': a dictionary mapping topic names to a dictionary containing:
          - 'total_proportion': float from col B
          - 'sentiment_proportion': a tuple of three floats from cols C, D, and E (which must sum to 1)
          - 'chunk_min_wc': int from col H (must be > 0)
          - 'chunk_max_wc': int from col I (must be > chunk_min_wc)
          - 'chunk_pref': mapped from col J if present
            - Represents the preferred number of chunks for this topic-sentiment pair or word count 
              size of each chunk depending on collection_mode
          - 'chunk_wc_distribution': mapped from col K if present
      - 'collection_ranges': a list of dictionaries representing ranges (either word count or chunk count),
          each with keys:
              - 'range': a tuple (start, end) where end > start
              - 'target_fraction': fraction for this range (must sum to 1)

    Args:
        rulebook_name (str): The file name of the Excel workbook, located in the "rulebooks" directory.
        collection_mode (str): The mode for collection COLLECTION_RANGES, either "word" or "chunk".

    Returns:
        Optional[dict]:
            On success, returns a dictionary with keys:
              - 'review_item'
              - 'total'
              - 'content_rules'
              - 'collection_ranges'
            Returns None if an error occurs.
    """
    
    # Validate collection_mode.
    if collection_mode not in ('word', 'chunk'):
        print("parse_rulebook_excel: Invalid collection_mode (must be 'word' or 'chunk').")
        return None
    
    try:
        # Get the file path relative to the script location
        file_path = Path(__file__).parent / "rulebooks" / rulebook_name

        if not file_path.exists():
            print(f"parse_rulebook_excel: File does not exist: {file_path}")
            return None
        if file_path.suffix.lower() not in {'.xlsx', '.xlsm'}:
            print(f"parse_rulebook_excel: File is not an Excel workbook: {file_path}")
            return None

        # Load the workbook and ensure the "MAIN" sheet is present
        wb = openpyxl.load_workbook(file_path)
        if "MAIN" not in wb.sheetnames:
            print(f"parse_rulebook_excel: 'MAIN' sheet not found in workbook: {file_path}")
            return None
        ws = wb["MAIN"]

        # Retrieve review_item from cell A1 and validate it
        review_item = ws['A1'].value
        if not isinstance(review_item, str):
            print("parse_rulebook_excel: Invalid value in cell A1 for review_item.")
            return None

        # Retrieve total from cell F1 and validate it
        total = ws['F1'].value
        if not isinstance(total, int) or total <= 0:
            print("parse_rulebook_excel: Invalid value in cell F1 for total.")
            return None

        content_rules = {}
        # Iterate over rows starting at row 5; include columns A-E and H-K.
        for row_num, row in enumerate(ws.iter_rows(min_row=5, max_col=11, values_only=True), start=5):
            # Column A: topic (index 0)
            topic = row[0]
            # Column B: total_proportion (index 1)
            total_proportion = row[1]
            # Columns C-E: sentiment proportions (indices 2, 3, 4)
            c = row[2]
            d = row[3]
            e = row[4]
            # Column H: chunk_min_wc (index 7)
            chunk_min_wc = row[7]
            # Column I: chunk_max_wc (index 8)
            chunk_max_wc = row[8]
            # Columns J: chunk_count_pref (index 9)
            chunk_pref_raw = row[9]
            # Columns K: chunk_wc_distribution (index 10)
            chunk_wc_distribution_raw = row[10]

            # Terminate iteration if the topic cell is empty
            if topic is None:
                break
            if not isinstance(topic, str):
                print(f"parse_rulebook_excel: Non-string value for topic at row {row_num}.")
                return None

            # Validate total_proportion
            if not isinstance(total_proportion, (int, float)):
                print(f"parse_rulebook_excel: Non-numeric value for proportion at row {row_num}.")
                return None
            if not (0 <= total_proportion <= 1):
                print(f"parse_rulebook_excel: Value out of [0,1] range at row {row_num} for proportion.")
                return None

            # Validate sentiment proportions
            if any(not isinstance(val, (int, float)) for val in (c, d, e)):
                print(f"parse_rulebook_excel: Non-numeric value for sentiment at row {row_num}.")
                return None
            if not all(0 <= val <= 1 for val in (c, d, e)):
                print(f"parse_rulebook_excel: Values out of [0,1] range at row {row_num} for sentiment")
                return None
            if abs((c + d + e) - 1) > 1e-9:
                print(f"parse_rulebook_excel: Sentiment does not sum to 1 at row {row_num}.")
                return None

            # Validate chunk_min_wc: must be non-None, numeric, an integer and > 0
            if chunk_min_wc is None or not isinstance(chunk_min_wc, int):
                print(f"parse_rulebook_excel: Non-integer value for chunk_min_wc at row {row_num}.")
                return None
            if chunk_min_wc <= 0:
                print(f"parse_rulebook_excel: Value for chunk_min_wc must be greater than 0 at row {row_num}.")
                return None

            # Validate chunk_max_wc: must be non-None, numeric, an integer and > chunk_min_wc
            if chunk_max_wc is None or not isinstance(chunk_max_wc, int):
                print(f"parse_rulebook_excel: Non-integer value for chunk_max_wc at row {row_num}.")
                return None
            if chunk_max_wc <= chunk_min_wc:
                print(f"parse_rulebook_excel: Value for chunk_max_wc I must be greater than chunk_min_wc at row {row_num}.")
                return None
            
            # Default chunk_pref to 0.50 if no value is present
            chunk_pref = 0.50
            # If collection_mode is "word", process chunk_pref as the preffered number of chunks
            if collection_mode == "word":
                if chunk_pref_raw is not None:
                    chunk_count_pref_str = str(chunk_pref_raw).strip()
                    mapping = {
                        "Lowest Number": get_setting('CHUNK_COUNT_MAPPING', 'Lowest_Number'),
                        "Low Number": get_setting('CHUNK_COUNT_MAPPING', 'Low_Number'),
                        "Mean Number": get_setting('CHUNK_COUNT_MAPPING', 'Mean_Number'),
                        "High Number": get_setting('CHUNK_COUNT_MAPPING', 'High_Number'),
                        "Highest Number": get_setting('CHUNK_COUNT_MAPPING', 'Highest_Number')
                    }
                    chunk_pref = mapping.get(chunk_count_pref_str, 0.5)
            
            # If collection_mode is "chunk", process chunk_pref as the preffered chunk word count size
            else:
                if chunk_pref_raw is not None:
                    chunk_count_pref_str = str(chunk_pref_raw).strip()
                    mapping = {
                        "Very Small": get_setting('CHUNK_SIZE_MAPPING', 'Very_Small'),
                        "Small": get_setting('CHUNK_SIZE_MAPPING', 'Small'),
                        "Medium": get_setting('CHUNK_SIZE_MAPPING', 'Medium'),
                        "Large": get_setting('CHUNK_SIZE_MAPPING', 'Large'),
                        "Very Large": get_setting('CHUNK_SIZE_MAPPING', 'Very_Large')
                    }
                    chunk_pref = mapping.get(chunk_count_pref_str, 0.5)
                    

            # Process chunk_wc_distribution: map expected strings to values.
            # Default to "Average Variation" if no value is present.
            chunk_wc_distribution = 5.0
            if chunk_wc_distribution_raw is not None:
                chunk_wc_distribution_str = str(chunk_wc_distribution_raw).strip()
                mapping_dist = {
                    "Low Variation": get_setting('CHUNK_VAR_MAPPING', 'Low_Variation'),
                    "Average Variation": get_setting('CHUNK_VAR_MAPPING', 'Average_Variation'),
                    "High Variation": get_setting('CHUNK_VAR_MAPPING', 'High_Variation')
                }
                chunk_wc_distribution = mapping_dist.get(chunk_wc_distribution_str, 5.0)

            # Only include rows with a positive total_proportion value.
            if total_proportion > 0:
                content_rules[topic] = {
                    'total_proportion': total_proportion,
                    'sentiment_proportion': (c, d, e),
                    'chunk_min_wc': chunk_min_wc,
                    'chunk_max_wc': chunk_max_wc,
                    'chunk_pref': chunk_pref,
                    'chunk_wc_distribution': chunk_wc_distribution
                }

        # Verify that the sum of total_proportion values equals 1
        total_prob = sum(rule['total_proportion'] for rule in content_rules.values())
        if abs(total_prob - 1) > 1e-9:
            print("parse_rulebook_excel: Sum of column B values does not equal 1.")
            return None

        # Process COLLECTION_RANGES sheet for collection_ranges
        if "COLLECTION_RANGES" not in wb.sheetnames:
            print(f"parse_rulebook_excel: 'COLLECTION_RANGES' sheet not found in workbook: {file_path}")
            return None

        COLLECTION_RANGES_ws = wb["COLLECTION_RANGES"]
        collection_ranges = []
        for row_num, row in enumerate(COLLECTION_RANGES_ws.iter_rows(min_row=4, max_col=3, values_only=True), start=4):
            # Expecting: Column A: start, Column B: end, Column C: proportion
            start_val, end_val, prop_val = row[0], row[1], row[2]
            # Terminate iteration if the start cell is empty
            if start_val is None:
                break

            # Validate that all three cells are present and numeric
            if start_val is None or end_val is None or prop_val is None:
                print(f"parse_rulebook_excel: Missing value in COLLECTION_RANGES sheet at row {row_num}.")
                return None
            if not isinstance(start_val, int) or not isinstance(end_val, int) or not isinstance(prop_val, (int, float)):
                print(f"parse_rulebook_excel: Incorrect value in COLLECTION_RANGES sheet at row {row_num}.")
                return None

            # Validate that end is greater than or equal to start
            if end_val < start_val:
                print(f"parse_rulebook_excel: In COLLECTION_RANGES sheet at row {row_num}, end ({end_val}) must be greater than start ({start_val}).")
                return None

            # If this is not the first range, ensure the current start is previous end + 1
            if collection_ranges:
                previous_end = collection_ranges[-1]['range'][1]
                if start_val != previous_end + 1:
                    print(f"parse_rulebook_excel: In COLLECTION_RANGES sheet at row {row_num}, start ({start_val}) must be equal to previous end + 1 ({previous_end + 1}).")
                    return None

            collection_ranges.append({
                'range': (int(start_val), int(end_val)),
                'target_fraction': float(prop_val)
            })

        # Validate that the sum of all proportions equals 1
        total_fraction = sum(item['target_fraction'] for item in collection_ranges)
        if abs(total_fraction - 1) > 1e-9:
            print("parse_rulebook_excel: Sum of fractions in COLLECTION_RANGES sheet does not equal 1.")
            return None

        return {
            'review_item': review_item,
            'total': total,
            'content_rules': content_rules,
            'collection_ranges': collection_ranges
        }

    except Exception as e:
        print(f"parse_rulebook_excel: {e}")
        return None

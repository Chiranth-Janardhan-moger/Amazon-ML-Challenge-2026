"""
Advanced Feature Engineering for Pairwise Entity Resolution.
Incorporates:
- Acronym detection (e.g. ZB <-> Zander Blue)
- Number-word normalization (e.g. 11th <-> Eleventh)
- State & province code expansion (e.g. MH <-> Maharashtra, DC <-> District of Columbia)
- Compact domain / DBA continuous matching
- Blank/degenerate name detection with address invariance
- Numerical house/building invariant matching
"""

import re
from typing import List, Set, Tuple
import numpy as np
from rapidfuzz import fuzz
from .matcher import clean_brand_name, extract_normalized_numbers, GENERIC_STOP

# State & province code expansion maps
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
}

INDIA_STATES = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana', 'hp': 'himachal pradesh',
    'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra',
    'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha',
    'pb': 'punjab', 'rj': 'rajasthan', 'sk': 'sikkim', 'tn': 'tamil nadu', 'ts': 'telangana',
    'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand', 'wb': 'west bengal',
    'dl': 'delhi', 'jk': 'jammu and kashmir', 'la': 'ladakh', 'py': 'puducherry', 'ch': 'chandigarh'
}

ALL_STATES = {**US_STATES, **INDIA_STATES}

NUM_WORDS = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th',
    'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
    'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th',
    'fifteenth': '15th', 'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
}

FEATURE_NAMES = [
    "brand_token_set_ratio",
    "brand_token_sort_ratio",
    "brand_ratio",
    "brand_compact_match",
    "brand_len_ratio",
    "is_acronym_match",
    "is_blank_name",
    "addr_token_set_ratio",
    "addr_token_sort_ratio",
    "addr_ratio",
    "addr_norm_sort_ratio",
    "addr_num_agree",
    "addr_num_jaccard",
    "addr_len_ratio",
    "is_source3",
    "blocking_score"
]


def normalize_address_text(addr: str) -> str:
    """Normalize address with state code expansions and number-word conversions."""
    if not addr or addr == 'None':
        return ""
    a = addr.lower()
    for word, repl in NUM_WORDS.items():
        a = re.sub(rf'\b{word}\b', repl, a)
    for code, full in ALL_STATES.items():
        a = re.sub(rf'\b{code}\b', full, a)
    return a


def check_acronym(name1: str, name2: str) -> float:
    """Check if one name is an acronym of the other (e.g. ZB <-> Zander Blue)."""
    words1 = [w for w in re.findall(r'[a-z0-9]+', name1.lower()) if w not in GENERIC_STOP]
    words2 = [w for w in re.findall(r'[a-z0-9]+', name2.lower()) if w not in GENERIC_STOP]
    if not words1 or not words2:
        return 0.0

    acr1 = ''.join([w[0] for w in words1 if w])
    acr2 = ''.join([w[0] for w in words2 if w])
    clean1 = ''.join(words1)
    clean2 = ''.join(words2)

    if len(clean1) in (2, 3, 4) and (clean1 == acr2 or clean1 in words2):
        return 1.0
    if len(clean2) in (2, 3, 4) and (clean2 == acr1 or clean2 in words1):
        return 1.0
    return 0.0


def is_blank_or_punct(name: str) -> float:
    """Detect if name is blank, whitespace, or just punctuation dots."""
    clean = re.sub(r'[^a-z0-9]', '', name.lower())
    return 1.0 if len(clean) == 0 else 0.0


def extract_pairwise_features(
    name1: str,
    addr1: str,
    name2: str,
    addr2: str,
    cand_id: str = "",
    blocking_score: float = 0.0
) -> List[float]:
    """
    Extract a 16-dimensional dense numerical feature vector for a candidate pair.
    """
    b1_str, b1_compact = clean_brand_name(name1)
    b2_str, b2_compact = clean_brand_name(name2)

    # 1. Brand similarities
    if b1_str and b2_str:
        b_set = fuzz.token_set_ratio(b1_str, b2_str) / 100.0
        b_sort = fuzz.token_sort_ratio(b1_str, b2_str) / 100.0
        b_ratio = fuzz.ratio(b1_str, b2_str) / 100.0
    else:
        raw1, raw2 = name1.lower(), name2.lower()
        b_set = fuzz.token_set_ratio(raw1, raw2) / 100.0
        b_sort = fuzz.token_sort_ratio(raw1, raw2) / 100.0
        b_ratio = fuzz.ratio(raw1, raw2) / 100.0

    # Compact / domain containment
    if b1_compact and b2_compact and (b1_compact in b2_compact or b2_compact in b1_compact) and min(len(b1_compact), len(b2_compact)) >= 4:
        compact_match = 1.0
    else:
        compact_match = 0.0

    # Brand length ratio
    l1, l2 = len(b1_str), len(b2_str)
    brand_len_ratio = min(l1, l2) / max(l1, l2, 1)

    # Acronym & blank name signals
    acronym_match = check_acronym(name1, name2)
    blank_name = is_blank_or_punct(name2)

    # 2. Address similarities
    a1_low = addr1.lower() if addr1 and addr1 != 'None' else ''
    a2_low = addr2.lower() if addr2 and addr2 != 'None' else ''
    
    if a1_low and a2_low:
        a_set = fuzz.token_set_ratio(a1_low, a2_low) / 100.0
        a_sort = fuzz.token_sort_ratio(a1_low, a2_low) / 100.0
        a_ratio = fuzz.ratio(a1_low, a2_low) / 100.0
        
        # Normalized address similarity (with state expansions & number-words)
        a1_norm = normalize_address_text(addr1)
        a2_norm = normalize_address_text(addr2)
        a_norm_sort = fuzz.token_sort_ratio(a1_norm, a2_norm) / 100.0
    else:
        a_set = 0.0
        a_sort = 0.0
        a_ratio = 0.0
        a_norm_sort = 0.0

    # Address numbers
    nums1 = extract_normalized_numbers(addr1)
    nums2 = extract_normalized_numbers(addr2)

    if nums1 and nums2:
        num_agree = 1.0 if bool(nums1 & nums2) else 0.0
        num_jaccard = len(nums1 & nums2) / len(nums1 | nums2)
    elif not nums1 and not nums2:
        num_agree = 0.5
        num_jaccard = 0.5
    else:
        num_agree = 0.2
        num_jaccard = 0.0

    # Address length ratio
    al1, al2 = len(a1_low), len(a2_low)
    addr_len_ratio = min(al1, al2) / max(al1, al2, 1)

    # Source origin flag
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return [
        b_set,
        b_sort,
        b_ratio,
        compact_match,
        brand_len_ratio,
        acronym_match,
        blank_name,
        a_set,
        a_sort,
        a_ratio,
        a_norm_sort,
        num_agree,
        num_jaccard,
        addr_len_ratio,
        is_s3,
        float(blocking_score)
    ]

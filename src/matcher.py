"""
Upgraded Precision-First Pairwise Matcher for Amazon ML Challenge 2026.
Incorporate 5 key noise corruption patterns discovered in competition ground truth:
1. Acronym Detection (e.g. ZB <-> Zander Blue Co)
2. State & Province Expansions (e.g. MH <-> Maharashtra, DC <-> District of Columbia)
3. Number-Word Alignment (e.g. 11th <-> Eleventh, 1st <-> First)
4. Blank / Degenerate Name Protection with Physical Location Anchoring
5. Continuous Domain / DBA Containment
"""

import re
from typing import Dict, Set, Tuple
from rapidfuzz import fuzz

RE_NUM = re.compile(r'\b\d+\b')
RE_DOMAIN = re.compile(r'\.(com|org|net|in|fr|co|io)\b', re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_COMPACT = re.compile(r'[^a-z0-9]')

GENERIC_STOP = {
    'inc', 'corp', 'corporation', 'incorporated', 'llc', 'llp',
    'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie',
    'fils', 'centre', 'center', 'groupe', 'group', 'services',
    'solutions', 'enterprises', 'industries', 'partners',
    'and', 'the', 'of', 'in', 'at', 'dba', 'formerly', 'fka', 'aka', '&'
}

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


def clean_brand_name(name: str) -> Tuple[str, str]:
    """Extract normalized brand name string and compact continuous string."""
    name_clean = RE_DOMAIN.sub('', name.lower())
    words = [
        w for w in RE_NON_ALPHANUM.sub(' ', name_clean).split()
        if w not in GENERIC_STOP and len(w) >= 2
    ]
    brand_str = ' '.join(words)
    compact_str = RE_COMPACT.sub('', brand_str)
    return brand_str, compact_str


def extract_normalized_numbers(text: str) -> Set[str]:
    """Extract numbers stripped of leading zeros (e.g. '01604' -> '1604')."""
    if not text or text == 'None':
        return set()
    return {str(int(n)) for n in RE_NUM.findall(text)}


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


class PairwiseMatcher:
    """Computes fine-grained similarity and selects high-confidence matches."""

    def __init__(self, brand_threshold: float = 0.70, addr_threshold: float = 0.50):
        self.brand_threshold = brand_threshold
        self.addr_threshold = addr_threshold

    def is_match(
        self,
        name1: str,
        addr1: str,
        name2: str,
        addr2: str
    ) -> bool:
        """
        Evaluate whether two entity records refer to the same real-world business.
        """
        b1_str, b1_compact = clean_brand_name(name1)
        b2_str, b2_compact = clean_brand_name(name2)

        # 1. Number Alignment (House / Building / Street numbers)
        nums1 = extract_normalized_numbers(addr1)
        nums2 = extract_normalized_numbers(addr2)

        if nums1 and nums2:
            num_agree = bool(nums1 & nums2)
        else:
            num_agree = True  # Neutral when numbers are absent

        if not num_agree:
            return False

        # 2. Normalized Address Similarity
        a1_norm = normalize_address_text(addr1)
        a2_norm = normalize_address_text(addr2)

        if a1_norm and a2_norm:
            addr_sim = max(
                fuzz.token_set_ratio(addr1.lower(), addr2.lower()),
                fuzz.token_sort_ratio(a1_norm, a2_norm)
            ) / 100.0
        else:
            addr_sim = 0.0

        # 3. Brand Name Similarity
        if b1_compact and b2_compact and (b1_compact in b2_compact or b2_compact in b1_compact) and min(len(b1_compact), len(b2_compact)) >= 4:
            brand_sim = 1.0
        elif b1_str and b2_str:
            brand_sim = max(
                fuzz.token_set_ratio(b1_str, b2_str),
                fuzz.token_sort_ratio(b1_str, b2_str),
                fuzz.ratio(b1_str, b2_str)
            ) / 100.0
        else:
            brand_sim = fuzz.token_set_ratio(name1.lower(), name2.lower()) / 100.0

        is_acr = check_acronym(name1, name2) == 1.0
        is_blank = is_blank_or_punct(name2) == 1.0

        # Decision Logic:
        # Rule A: Standard high-confidence match
        if brand_sim >= self.brand_threshold and addr_sim >= self.addr_threshold:
            return True

        # Rule B: Exact physical address agreement with moderate brand overlap
        if addr_sim >= 0.80 and brand_sim >= 0.48:
            return True

        # Rule C: Acronym match (e.g. ZB <-> Zander Blue) with strong address agreement
        if is_acr and addr_sim >= 0.75:
            return True

        # Rule D: Blank/punctuation name (. .) with near-exact address
        if is_blank and addr_sim >= 0.88:
            return True

        # Rule E: Near-exact brand (>= 0.95) with missing or partial address
        if brand_sim >= 0.95 and (addr_sim >= 0.35 or not addr1 or not addr2 or addr1 == 'None' or addr2 == 'None'):
            return True

        return False

    def match_candidates(
        self,
        candidates_per_s1: Dict[str, Set[str]],
        s1_records: Dict[str, Tuple[str, str]],
        cand_records: Dict[str, Tuple[str, str]]
    ) -> Dict[str, Set[str]]:
        """
        Filter candidate pairs down to final high-confidence matches.
        """
        final_matches: Dict[str, Set[str]] = {}

        for s1_id, candidate_ids in candidates_per_s1.items():
            if not candidate_ids or s1_id not in s1_records:
                final_matches[s1_id] = set()
                continue

            n1, a1 = s1_records[s1_id]
            matched_for_s1 = set()

            for cid in candidate_ids:
                if cid not in cand_records:
                    continue

                n2, a2 = cand_records[cid]
                if self.is_match(n1, a1, n2, a2):
                    matched_for_s1.add(cid)

            final_matches[s1_id] = matched_for_s1

        return final_matches

"""
High-throughput text normalization and feature extraction for Business Entity Resolution.
Supports multilingual entities across US, India, and France.
"""

import re
from typing import Tuple, Set

# Universal legal forms & stopwords (US, India, France)
LEGAL_STOPWORDS = {
    'inc', 'corp', 'corporation', 'incorporated', 'llc', 'llp',
    'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'sarl', 'sas', 'sa', 'sci', 'eurl', 'snc', 'gie',
    'and', 'the', 'of', 'in', 'at', 'dba', 'formerly', 'fka', 'aka',
    'com', 'org', 'net', 'in', 'fr', 'co', 'io'
}

# Common street and layout designations across English & French
COMMON_ADDRESS_TOKENS = {
    # English
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'dr', 'drive', 'ln', 'lane',
    'blvd', 'boulevard', 'fl', 'floor', 'unit', 'suite', 'ste', 'apt', 'apartment',
    'po', 'box', 'near', 'opp', 'behind', 'block', 'plot', 'no', 'flat', 'cross',
    'main', 'nagar', 'colony', 'layout', 'city', 'town', 'dist', 'district', 'state',
    # French
    'rue', 'avenue', 'boulevard', 'bd', 'allée', 'chemin', 'route', 'impasse',
    'place', 'cours', 'quai', 'voie', 'passage', 'étage', 'batiment', 'bâtiment',
    'residence', 'résidence', 'cedex'
}

# Regex pre-compiled for performance
RE_DOMAIN = re.compile(r'\.(com|org|net|in|fr|co|io)\b', re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_COMPACT = re.compile(r'[^a-z0-9]')


def normalize_business_record(name: str, address: str) -> Tuple[Set[str], Set[str], Set[str], str]:
    """
    Extract normalized token sets and representations from business record fields.

    Returns:
        name_tokens: Set of significant alphanumeric tokens in business name
        address_numbers: Set of extracted numerical digits with leading zeros stripped
        address_tokens: Set of significant non-generic address tokens
        compact_name: Alphanumeric continuous string for substring/acronym matching
    """
    if not name or name == 'nan':
        name = ''
    if not address or address == 'nan':
        address = ''

    # Clean name
    name_clean = RE_DOMAIN.sub('', name.lower())
    name_clean = RE_NON_ALPHANUM.sub(' ', name_clean)
    name_tokens = {t for t in name_clean.split() if len(t) >= 2 and t not in LEGAL_STOPWORDS}

    # Extract clean address numbers (normalized: 01604 -> '1604')
    address_lower = address.lower()
    raw_nums = RE_NUMBERS.findall(address_lower)
    address_numbers = {str(int(n)) for n in raw_nums}

    # Extract significant address words (street names, cities, landmarks)
    address_words = RE_NON_ALPHANUM.sub(' ', address_lower)
    address_tokens = {
        w for w in address_words.split()
        if len(w) >= 3 and w not in COMMON_ADDRESS_TOKENS and not w.isdigit()
    }

    # Compact name for concatenated domain/DBA matching
    compact_name = RE_COMPACT.sub('', name_clean)

    return name_tokens, address_numbers, address_tokens, compact_name

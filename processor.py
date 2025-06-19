# Import necessary libraries
import pandas as pd
import numpy as np
import re
import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import combinations
from rapidfuzz import fuzz
from tqdm import tqdm
import warnings
import string
warnings.filterwarnings('ignore')



def extract_emails(value):
    if pd.isnull(value):
        return None
    # Split using comma or semicolon with optional whitespace
    raw_emails = re.split(r'\s*[,;]\s*', value)
    # Strip whitespace from each email and filter out empty strings
    emails = [email.strip() for email in raw_emails if email.strip()]
    return emails

def preprocess_df(df):

    df["clean_name"] = df["first_name"] + " " + df["last_name"]
    df["list_mails"] = df["email_address"].apply(extract_emails)

    # filter out the null companys, building control emails,
    
    return df

def preprocess_text(text):
    """Preprocess text similar to the original code"""
    if pd.isnull(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Remove common company suffixes
    suffixes = [
        'llc', 'inc', 'ltd', 'corp', 'co', 'company',
        'incorporated', 'limited', 'corporation', 'plc', 'gmbh', 'srl',
        'sa', 'ag', 'kg', 'oy', 'ab', 'as', 'pte', 'pte ltd', 'llp', 'lp'
    ]
    pattern = r'\b(' + '|'.join(suffixes) + r')\b'
    text = re.sub(pattern, '', text)
    
    # Expand common abbreviations
    abbreviations = {
        'intl': 'international',
        'tech': 'technology',
        'mfg': 'manufacturing',
        'svc': 'service',
        'svcs': 'services',
        'mgmt': 'management',
        'grp': 'group',
        'inst': 'institute',
        'univ': 'university',
        'dept': 'department',
        'deptt': 'department',
        'co': 'company',
        'cos': 'companies',
        'corp': 'corporation',
        'assn': 'association',
        'assoc': 'association',
        'org': 'organization',
        'hosp': 'hospital',
        'med': 'medical',
        'ctr': 'center',
        'cnt': 'center',
        'cntre': 'centre'
    }
    abbr_pattern = re.compile(r'\b(' + '|'.join(abbreviations.keys()) + r')\b')
    text = abbr_pattern.sub(lambda m: abbreviations[m.group()], text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def standardize_phone_number(phone):
    """Normalize phone number by removing non-digit characters"""
    if pd.isnull(phone):
        return ''
    phone = re.sub(r'\D', '', str(phone))  # Remove non-digit characters
    return phone

def double_metaphone(name):
    """Generate phonetic codes for blocking"""
    tokens = name.split()
    metaphone_tokens = [jellyfish.metaphone(token) for token in tokens]
    return ' '.join(metaphone_tokens)

def get_alphabetical_label(index):
    """Convert index to alphabetical label (A, B, C, ..., AA, AB, etc.)"""
    letters = string.ascii_uppercase
    result = []
    while index >= 0:
        result.append(letters[index % 26])
        index = index // 26 - 1
    return ''.join(reversed(result))

def emails_have_overlap(email_list1, email_list2):
    """Check if two email lists have any common emails"""
    if not email_list1 or not email_list2:
        return False
    
    # Convert to sets and check intersection
    set1 = set([email.lower().strip() for email in email_list1 if email and pd.notnull(email)])
    set2 = set([email.lower().strip() for email in email_list2 if email and pd.notnull(email)])
    
    return len(set1.intersection(set2)) > 0

def find_company_matches(df):
    """Find similar companies using sentence transformers and clustering"""
    print("Step 1: Finding similar companies...")
    
    # Preprocess company names
    df['cleaned_company_name'] = df['customer_name'].apply(preprocess_text)
    
    # Generate phonetic codes for blocking
    df['phonetic_code'] = df['cleaned_company_name'].apply(double_metaphone)
    
    # Build blocks based on phonetic codes
    df['blocking_key'] = df['phonetic_code'].apply(lambda x: x[:3])
    blocks = df.groupby('blocking_key')
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode all company names
    print("Encoding company names...")
    df['company_embedding'] = list(model.encode(df['cleaned_company_name'].tolist(), show_progress_bar=True))
    
    # Initialize graph for clustering
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    # Compare records within blocks
    print("Comparing companies within blocks...")
    for block_key, block_df in tqdm(blocks, total=len(blocks)):
        if len(block_df) < 2:
            continue
        
        records = block_df.to_dict('records')
        indices = block_df.index.tolist()
        
        for i, (idx1, rec1) in enumerate(zip(indices, records)):
            for j, (idx2, rec2) in enumerate(zip(indices[i+1:], records[i+1:]), i+1):
                name1 = rec1['cleaned_company_name']
                name2 = rec2['cleaned_company_name']
                
                # Skip if either name is empty
                if not name1 or not name2:
                    continue
                
                # Phonetic Similarity
                phonetic1 = rec1['phonetic_code']
                phonetic2 = rec2['phonetic_code']
                phonetic_sim = fuzz.ratio(phonetic1, phonetic2) / 100
                
                # Embedding Similarity
                embedding1 = rec1['company_embedding']
                embedding2 = rec2['company_embedding']
                name_cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
                
                # String Similarity Measures
                name_fuzz_ratio = fuzz.token_set_ratio(name1, name2) / 100
                partial_ratio = fuzz.partial_ratio(name1, name2) / 100
                levenshtein_dist = jellyfish.levenshtein_distance(name1, name2)
                max_len = max(len(name1), len(name2))
                levenshtein_ratio = (max_len - levenshtein_dist) / max_len if max_len > 0 else 0
                
                # Token-Based Matching
                tokens1 = name1.split()
                tokens2 = name2.split()
                token_matches = sum(1 for token in tokens1 if token in tokens2)
                token_match_ratio = token_matches / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
                
                # Aggregate Similarity Score
                aggregate_score = (
                    0.35 * name_cosine_sim +
                    0.25 * name_fuzz_ratio +
                    0.15 * phonetic_sim +
                    0.1 * partial_ratio +
                    0.1 * token_match_ratio +
                    0.05 * levenshtein_ratio
                )
                
                # Decision Threshold for company matching
                if aggregate_score >= 0.75:
                    G.add_edge(idx1, idx2, weight=aggregate_score)
    
    # Identify company clusters
    print("Identifying company clusters...")
    company_clusters = list(nx.connected_components(G))
    
    # Assign company group IDs
    company_group_mapping = {}
    for i, cluster in enumerate(company_clusters):
        company_group_id = f"{i+1}{get_alphabetical_label(0)}"  # 1A, 2A, 3A, etc.
        for idx in cluster:
            company_group_mapping[idx] = company_group_id
    
    # Handle ungrouped companies (single companies)
    for idx in df.index:
        if idx not in company_group_mapping:
            company_group_id = f"{len(company_clusters)+1}{get_alphabetical_label(0)}"
            company_group_mapping[idx] = company_group_id
            company_clusters.append({idx})
    
    df['company_group_id'] = df.index.map(company_group_mapping)
    
    return df

def find_primary_matches(df):
    """Find exact name matches within each company group"""
    print("Step 2: Finding primary matches (name matches within companies)...")
    
    # Normalize names for case-insensitive matching
    df['normalized_clean_name'] = df['clean_name'].str.lower().str.strip()
    
    primary_group_mapping = {}
    primary_group_counter = 0
    
    # Group by company groups
    for company_group_id in df['company_group_id'].unique():
        company_df = df[df['company_group_id'] == company_group_id]
        
        # Group by exact name matches within this company
        name_groups = company_df.groupby('normalized_clean_name')
        
        for name, name_group in name_groups:
            if len(name_group) > 1:  # Only create groups for actual matches
                primary_group_id = f"{get_alphabetical_label(primary_group_counter)}{1}"  # A1, B1, C1, etc.
                for idx in name_group.index:
                    primary_group_mapping[idx] = primary_group_id
                primary_group_counter += 1
            else:
                # Single names get their own group
                primary_group_id = f"{get_alphabetical_label(primary_group_counter)}{1}"
                primary_group_mapping[name_group.index[0]] = primary_group_id
                primary_group_counter += 1
    
    df['primary_group_id'] = df.index.map(primary_group_mapping)
    
    return df

def find_secondary_matches(df):
    """Find email and phone matches within each primary group"""
    print("Step 3: Finding secondary matches (email/phone matches within primary groups)...")
    
    # Normalize phone numbers
    df['normalized_phnumber'] = df['direct_phone'].apply(standardize_phone_number)
    
    secondary_group_mapping = {}
    secondary_group_counter = 0
    
    # Group by primary groups
    for primary_group_id in df['primary_group_id'].unique():
        primary_df = df[df['primary_group_id'] == primary_group_id]
        
        if len(primary_df) <= 1:
            # Single records get their own secondary group
            secondary_group_id = f"{secondary_group_counter+1}{get_alphabetical_label(0)}{1}"  # 1A1, 2A1, etc.
            for idx in primary_df.index:
                secondary_group_mapping[idx] = secondary_group_id
            secondary_group_counter += 1
            continue
        
        # Create graph for email/phone matching within this primary group
        G = nx.Graph()
        G.add_nodes_from(primary_df.index)
        
        records = primary_df.to_dict('records')
        indices = primary_df.index.tolist()
        
        # Compare all pairs within this primary group
        for i, (idx1, rec1) in enumerate(zip(indices, records)):
            for j, (idx2, rec2) in enumerate(zip(indices[i+1:], records[i+1:]), i+1):
                
                # Check email overlap
                email_match = False
                if rec1['list_mails'] and rec2['list_mails']:
                    # Assuming list_mails is already a list or convert from string representation
                    if isinstance(rec1['list_mails'], str):
                        emails1 = eval(rec1['list_mails']) if rec1['list_mails'].startswith('[') else [rec1['list_mails']]
                    else:
                        emails1 = rec1['list_mails']
                    
                    if isinstance(rec2['list_mails'], str):
                        emails2 = eval(rec2['list_mails']) if rec2['list_mails'].startswith('[') else [rec2['list_mails']]
                    else:
                        emails2 = rec2['list_mails']
                    
                    email_match = emails_have_overlap(emails1, emails2)
                
                # Check phone number match
                phone_match = (rec1['normalized_phnumber'] == rec2['normalized_phnumber'] and 
                             rec1['normalized_phnumber'] != '' and rec2['normalized_phnumber'] != '')
                
                # Add edge if either email or phone matches
                if email_match or phone_match:
                    G.add_edge(idx1, idx2)
        
        # Find connected components (secondary groups)
        secondary_clusters = list(nx.connected_components(G))
        
        for cluster in secondary_clusters:
            secondary_group_id = f"{secondary_group_counter+1}{get_alphabetical_label(0)}{1}"  # 1A1, 2A1, etc.
            for idx in cluster:
                secondary_group_mapping[idx] = secondary_group_id
            secondary_group_counter += 1
    
    df['secondary_group_id'] = df.index.map(secondary_group_mapping)
    
    return df

def create_match_summary(df):
    """Create summary columns for matches"""
    print("Step 4: Creating match summary...")
    
    # Add customer_id column if it doesn't exist (using index)
    if 'S_id' not in df.columns:
        df['S_id'] = df.index
    
    # Count matches for each level
    company_counts = df.groupby('company_group_id').size()
    primary_counts = df.groupby('primary_group_id').size()
    secondary_counts = df.groupby('secondary_group_id').size()
    
    df['company_match_count'] = df['company_group_id'].map(company_counts)
    df['primary_match_count'] = df['primary_group_id'].map(primary_counts)
    df['secondary_match_count'] = df['secondary_group_id'].map(secondary_counts)
    
    # Create match indicators
    df['has_company_matches'] = df['company_match_count'] > 1
    df['has_primary_matches'] = df['primary_match_count'] > 1
    df['has_secondary_matches'] = df['secondary_match_count'] > 1
    
    # Create match_check column - "yes" if has name matches OR email/phone matches, "no" otherwise
    df['match_check'] = df.apply(
        lambda row: "yes" if (row['has_primary_matches'] or row['has_secondary_matches']) else "no",
        axis=1
    )
    
    # Create matching IDs lists
    print("Creating matching ID lists...")
    
    # Primary matching IDs - list of customer IDs in the same primary group
    primary_group_to_ids = df.groupby('primary_group_id')['S_id'].apply(list).to_dict()
    df['matching_names_ids'] = df['primary_group_id'].map(primary_group_to_ids)
    
    # Secondary matching IDs - list of customer IDs in the same secondary group
    secondary_group_to_ids = df.groupby('secondary_group_id')['S_id'].apply(list).to_dict()
    df['secondary_matching_ids'] = df['secondary_group_id'].map(secondary_group_to_ids)
    
    return df

def main_matching_pipeline(df):
    """Main pipeline to run all matching steps"""
    print("Starting three-layer customer matching pipeline...")
    
    # Ensure required columns exist
    required_columns = ['first_name', 'last_name', 'customer_name', 'direct_phone', 'email_address']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Step 1: Find company matches
    df = find_company_matches(df)
    
    # Step 2: Find primary matches (names within companies)
    df = find_primary_matches(df)
    
    # Step 3: Find secondary matches (email/phone within primary groups)
    df = find_secondary_matches(df)
    
    # Step 4: Create summary statistics
    df = create_match_summary(df)
    
    # Sort results by company group, then primary group, then secondary group
    df = df.sort_values(['company_group_id', 'primary_group_id', 'secondary_group_id']).reset_index(drop=True)
    
    print("Matching pipeline completed!")
    print(f"Total records: {len(df)}")
    print(f"Company groups: {df['company_group_id'].nunique()}")
    print(f"Primary groups: {df['primary_group_id'].nunique()}")
    print(f"Secondary groups: {df['secondary_group_id'].nunique()}")
    print(f"Records with company matches: {df['has_company_matches'].sum()}")
    print(f"Records with primary matches: {df['has_primary_matches'].sum()}")
    print(f"Records with secondary matches: {df['has_secondary_matches'].sum()}")
    
    return df

# Example usage:
# df_result = main_matching_pipeline(your_dataframe)
# df_result.to_excel('customer_matches_result.xlsx', index=False)

# Output columns include:
# - company_group_id: Company grouping (1A, 2A, etc.)
# - primary_group_id: Name matching within companies (A1, B1, etc.)  
# - secondary_group_id: Email/phone matching within primary groups (1A1, 2A1, etc.)
# - matching_names_ids: List of customer IDs with matching names in same company
# - secondary_matching_ids: List of customer IDs with matching email/phone in same primary group
# - Match counts and indicators for each layer
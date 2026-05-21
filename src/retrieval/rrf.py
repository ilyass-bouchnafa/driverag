# src/retrieval/rrf.py
from collections import defaultdict
from typing import List, Dict

def reciprocal_rank_fusion(
    results_list: List[List[Dict]], 
    k: int = 60, 
    top_n: int = 20
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) - Professional grade result fusion.
    Very effective when combining multiple retrieval strategies (multi-query + HyDE).
    """
    score_dict = defaultdict(float)
    chunk_dict = {}

    for ranked_list in results_list:
        for rank, chunk in enumerate(ranked_list, start=1):
            # Unique chunk identifier
            chunk_id = (
                f"{chunk['metadata'].get('source', 'unknown')}_"
                f"p{chunk['metadata'].get('page', 0)}_"
                f"c{chunk['metadata'].get('chunk_index', 0)}"
            )
            
            score_dict[chunk_id] += 1.0 / (k + rank)
            
            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = chunk

    # Sort by RRF score
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for chunk_id, rrf_score in sorted_items[:top_n]:
        chunk = chunk_dict[chunk_id].copy()
        chunk["score"] = round(rrf_score, 4)
        final_results.append(chunk)

    return final_results
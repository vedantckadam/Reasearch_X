import numpy as np
from typing import Dict, List, Any, Optional

def calculate_temporal_analysis(paper_data: List[Dict[str, Any]], num_topics: int) -> Dict[str, Any]:
    """
    paper_data should be a list of dicts:
    [
        {"paper_id": "...", "year": 2021, "topics": [0.8, 0.1, ...]}, ...
    ]
    """
    total_papers = len(paper_data)
    papers_with_year = [p for p in paper_data if p.get("year") is not None]
    
    result = {
        "total_papers": total_papers,
        "papers_with_year": len(papers_with_year),
        "missing_year": total_papers - len(papers_with_year),
        "years": {},
        "topics": {}
    }
    
    if not papers_with_year:
        return result
        
    # Group by year
    years_dict = {}
    for p in papers_with_year:
        yr = p["year"]
        if yr not in years_dict:
            years_dict[yr] = []
        years_dict[yr].append(p)
        
    for yr, papers in years_dict.items():
        pc = len(papers)
        yr_data = {
            "year": yr,
            "paper_count": pc,
            "topics": {}
        }
        
        for t in range(num_topics):
            topic_probs = [p["topics"][t] for p in papers]
            dom_count = sum(1 for p in papers if np.argmax(p["topics"]) == t)
            prob_mass = sum(topic_probs)
            yr_data["topics"][t] = {
                "dominant_papers": dom_count,
                "dominant_percentage": dom_count / pc if pc > 0 else 0,
                "probability_mass": prob_mass,
                "average_probability": prob_mass / pc if pc > 0 else 0
            }
            
        result["years"][yr] = yr_data
        
    # Topic trends
    for t in range(num_topics):
        topic_timeline = []
        for yr in sorted(years_dict.keys()):
            yr_data = result["years"][yr]["topics"][t]
            topic_timeline.append({
                "year": yr,
                "paper_count": len(years_dict[yr]),
                "dominant_percentage": yr_data["dominant_percentage"],
                "probability_mass": yr_data["probability_mass"],
                "average_probability": yr_data["average_probability"]
            })
            
        first_yr = min(years_dict.keys())
        last_yr = max(years_dict.keys())
        
        # Trend
        if len(years_dict) >= 3:
            x = np.arange(len(years_dict))
            y = [item["average_probability"] for item in topic_timeline]
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
            
        result["topics"][t] = {
            "first_year_observed": first_yr,
            "last_year_observed": last_yr,
            "number_of_years_present": len(years_dict),
            "trend": trend,
            "timeline": topic_timeline
        }
        
    return result

def calculate_temporal_intersections(paper_data: List[Dict[str, Any]], num_topics: int, threshold: float = 0.20) -> List[Dict[str, Any]]:
    papers_with_year = [p for p in paper_data if p.get("year") is not None]
    
    years_dict = {}
    for p in papers_with_year:
        yr = p["year"]
        if yr not in years_dict:
            years_dict[yr] = []
        years_dict[yr].append(p)
        
    pairs = []
    
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            timeline = []
            
            for yr in sorted(years_dict.keys()):
                papers = years_dict[yr]
                
                # Support count
                support = sum(1 for p in papers if p["topics"][i] >= threshold and p["topics"][j] >= threshold)
                
                # Weighted intersection
                theta_yr = np.array([p["topics"] for p in papers])
                weighted_intersect = float(theta_yr[:, i] @ theta_yr[:, j]) if len(papers) > 0 else 0.0
                
                timeline.append({
                    "year": yr,
                    "paper_count": len(papers),
                    "support_count": support,
                    "weighted_intersection": weighted_intersect
                })
                
            if len(years_dict) >= 3:
                x = np.arange(len(years_dict))
                y = [item["support_count"] for item in timeline]
                slope, _ = np.polyfit(x, y, 1)
                
                if slope > 0.5:
                    trend = "increasing"
                elif slope < -0.5:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
                
            pairs.append({
                "topic_pair": {"topic_a_id": i, "topic_b_id": j},
                "trend": trend,
                "timeline": timeline
            })
            
    return pairs

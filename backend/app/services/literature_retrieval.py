import os
import uuid
import logging
from typing import List
from app.services.openalex import OpenAlexClient
from app.services.semantic_scholar import SemanticScholarClient
from app.services.query_generation import generate_queries
from app.services.deduplication import deduplicate_papers
from app.services.relevance import rank_papers
from app.schemas.paper import PaperSchema
from app.models.paper_db import SessionLocal, PaperDB, Analysis

logger = logging.getLogger(__name__)

# Configurable settings
MAX_PAPERS = int(os.getenv("RETRIEVAL_MAX_PAPERS", 100))
OPENALEX_RESULTS_PER_QUERY = int(os.getenv("OPENALEX_RESULTS_PER_QUERY", 25))
SEMANTIC_SCHOLAR_RESULTS_PER_QUERY = int(os.getenv("SEMANTIC_SCHOLAR_RESULTS_PER_QUERY", 25))

class LiteratureRetrievalOrchestrator:
    def __init__(self):
        self.openalex = OpenAlexClient()
        self.semantic_scholar = SemanticScholarClient()

    def run_retrieval(self, analysis_id: str, seed_text: str):
        session = SessionLocal()
        try:
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not analysis:
                analysis = Analysis(id=analysis_id, status="RETRIEVING", seed_text=seed_text)
                session.add(analysis)
            else:
                analysis.status = "RETRIEVING"
                analysis.seed_text = seed_text
            session.commit()

            logger.info(f"Analysis {analysis_id}: Generating queries...")
            queries = generate_queries(seed_text)
            
            all_papers: List[PaperSchema] = []

            for query in queries:
                logger.info(f"Analysis {analysis_id}: Searching OpenAlex for '{query}'...")
                oa_results = self.openalex.search_works(query, limit=OPENALEX_RESULTS_PER_QUERY)
                all_papers.extend(oa_results)

                logger.info(f"Analysis {analysis_id}: Searching Semantic Scholar for '{query}'...")
                ss_results = self.semantic_scholar.search_works(query, limit=SEMANTIC_SCHOLAR_RESULTS_PER_QUERY)
                all_papers.extend(ss_results)

            analysis.status = "DEDUPLICATING"
            session.commit()
            
            logger.info(f"Analysis {analysis_id}: Deduplicating {len(all_papers)} retrieved papers...")
            deduped_papers = deduplicate_papers(all_papers)
            
            analysis.status = "RANKING"
            session.commit()

            logger.info(f"Analysis {analysis_id}: Ranking {len(deduped_papers)} papers...")
            ranked_papers = rank_papers(seed_text, deduped_papers)
            
            # Filter low-quality results (e.g., extremely low relevance or zeroed homonym noise)
            min_relevance = float(os.getenv("MIN_RELEVANCE_SCORE", "0.02"))
            filtered_papers = [p for p in ranked_papers if p.relevance_score >= min_relevance]
            if not filtered_papers and ranked_papers:
                filtered_papers = [p for p in ranked_papers if p.relevance_score > 0.0]
            
            # Limit to max papers
            final_corpus = filtered_papers[:MAX_PAPERS]

            # Clear previous papers if re-running analysis
            session.query(PaperDB).filter(PaperDB.analysis_id == analysis_id).delete()
            session.commit()
            
            logger.info(f"Analysis {analysis_id}: Storing {len(final_corpus)} papers in DB...")
            
            for p in final_corpus:
                paper_id = str(uuid.uuid4())
                paper_db = PaperDB(
                    id=paper_id,
                    analysis_id=analysis_id,
                    title=p.title,
                    abstract=p.abstract,
                    year=p.year,
                    authors=p.authors,
                    doi=p.doi,
                    sources=p.sources,
                    citation_count=p.citation_count,
                    url=p.url,
                    publication_venue=p.publication_venue,
                    is_open_access=p.is_open_access,
                    relevance_score=p.relevance_score,
                    openalex_id=p.openalex_id,
                    semantic_scholar_id=p.semantic_scholar_id
                )
                session.add(paper_db)
                
            session.commit()
            logger.info(f"Analysis {analysis_id}: Retrieval completed successfully.")

            # Run Topic Modeling
            logger.info(f"Analysis {analysis_id}: Running topic modeling...")
            from app.api.topic_routes import run_topic_modeling_task
            run_topic_modeling_task(analysis_id, num_topics=8)

            # Check if topic modeling succeeded
            session.refresh(analysis)
            if "FAILED" in analysis.status:
                logger.error(f"Analysis {analysis_id}: Topic modeling failed. Stopping pipeline.")
                return

            # Run Latent & Temporal Analysis
            logger.info(f"Analysis {analysis_id}: Running latent and temporal analysis...")
            from app.api.latent_routes import run_latent_analysis_route
            run_latent_analysis_route(analysis_id, session)

            # Run Gap Generation
            logger.info(f"Analysis {analysis_id}: Running gap generation...")
            from app.api.gap_routes import run_gap_analysis_route
            run_gap_analysis_route(analysis_id, session)

            analysis.status = "COMPLETED"
            session.commit()
            logger.info(f"Analysis {analysis_id}: Full pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Analysis {analysis_id}: Pipeline failed: {e}")
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
            if analysis:
                analysis.status = "FAILED"
                session.commit()
        finally:
            session.close()

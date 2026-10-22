import os
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, JSON, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from app.config import load_project_env

load_project_env()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./researchgapx.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="QUEUED")
    seed_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    topics = relationship("TopicAnalysis", back_populates="analysis")

class PaperDB(Base):
    __tablename__ = "papers"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, index=True)
    title = Column(String, nullable=True)
    abstract = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    authors = Column(JSON, nullable=True)
    doi = Column(String, nullable=True)
    sources = Column(JSON, nullable=True)
    citation_count = Column(Integer, default=0)
    url = Column(String, nullable=True)
    publication_venue = Column(String, nullable=True)
    is_open_access = Column(Boolean, default=False)
    relevance_score = Column(Float, default=0.0)
    openalex_id = Column(String, nullable=True)
    semantic_scholar_id = Column(String, nullable=True)

class TopicAnalysis(Base):
    __tablename__ = "topic_analyses"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"))
    num_topics = Column(Integer)
    vocabulary_size = Column(Integer)
    coherence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="topics")

class Topic(Base):
    __tablename__ = "topics"
    id = Column(String, primary_key=True, index=True)
    topic_analysis_id = Column(String, ForeignKey("topic_analyses.id"))
    topic_idx = Column(Integer)
    label = Column(String)
    dominant_papers = Column(Integer)
    average_probability = Column(Float)
    corpus_share = Column(Float)
    top_words = Column(JSON)

class DocumentTopic(Base):
    __tablename__ = "document_topics"
    id = Column(String, primary_key=True, index=True)
    topic_analysis_id = Column(String, ForeignKey("topic_analyses.id"))
    paper_id = Column(String, ForeignKey("papers.id"))
    topic_idx = Column(Integer)
    probability = Column(Float)

class LatentDimensionDB(Base):
    __tablename__ = "latent_dimensions"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    dimension_index = Column(Integer)
    singular_value = Column(Float)
    squared_singular_value = Column(Float)
    latent_energy_ratio = Column(Float)
    cumulative_energy = Column(Float)
    eigenvalue = Column(Float)
    normalized_eigen_contribution = Column(Float)
    cumulative_eigen_contribution = Column(Float)
    is_low_energy = Column(Boolean)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('analysis_id', 'dimension_index', name='uq_latent_dim'),)

class LatentTopicLoadingDB(Base):
    __tablename__ = "latent_topic_loadings"
    id = Column(String, primary_key=True, index=True)
    latent_dimension_id = Column(String, ForeignKey("latent_dimensions.id"), index=True)
    topic_index = Column(Integer)
    loading = Column(Float)
    absolute_loading = Column(Float)
    rank = Column(Integer)
    
    __table_args__ = (UniqueConstraint('latent_dimension_id', 'topic_index', name='uq_latent_loading'),)

class TemporalTopicDB(Base):
    __tablename__ = "temporal_topics"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    year = Column(Integer)
    topic_index = Column(Integer)
    paper_count = Column(Integer)
    dominant_count = Column(Integer)
    dominant_percentage = Column(Float)
    probability_mass = Column(Float)
    average_probability = Column(Float)
    trend = Column(String)
    first_observed_year = Column(Integer, nullable=True)
    last_observed_year = Column(Integer, nullable=True)
    number_of_years_present = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('analysis_id', 'year', 'topic_index', name='uq_temporal_topic'),)

class TemporalIntersectionDB(Base):
    __tablename__ = "temporal_intersections"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    year = Column(Integer)
    topic_i = Column(Integer)
    topic_j = Column(Integer)
    support_count = Column(Integer)
    normalized_weighted_value = Column(Float)
    weighted_intersection = Column(Float)
    threshold = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('analysis_id', 'year', 'topic_i', 'topic_j', name='uq_temporal_intersection'),)

class LatentAnalysisSummaryDB(Base):
    __tablename__ = "latent_analysis_summaries"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    total_dimensions = Column(Integer)
    rank = Column(Integer)
    energy_80_dimensions = Column(JSON)
    energy_90_dimensions = Column(JSON)
    low_energy_threshold = Column(Float)
    reconstruction_error = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class GapCandidateDB(Base):
    __tablename__ = "gap_candidates"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    candidate_id = Column(String, unique=True, index=True)
    topic_indices = Column(JSON)
    topic_labels = Column(JSON)
    latent_dimension_indices = Column(JSON)
    rarity_score = Column(Float, nullable=True)
    intersection_score = Column(Float, nullable=True)
    temporal_score = Column(Float, nullable=True)
    limitation_score = Column(Float, nullable=True)
    semantic_support_score = Column(Float, nullable=True)
    final_gap_score = Column(Float)
    evidence_count = Column(Integer)
    confidence_level = Column(String)
    reasoning_signals = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class GapEvidenceDB(Base):
    __tablename__ = "gap_evidences"
    id = Column(String, primary_key=True, index=True)
    candidate_id = Column(String, ForeignKey("gap_candidates.candidate_id"), index=True)
    paper_id = Column(String, ForeignKey("papers.id"), nullable=True)
    evidence_type = Column(String)
    topic_index = Column(Integer, nullable=True)
    topic_pair = Column(JSON, nullable=True)
    year = Column(Integer, nullable=True)
    evidence_text = Column(String, nullable=True)
    evidence_metadata = Column(JSON)

class RAGChunkDB(Base):
    __tablename__ = "rag_chunks"
    id = Column(String, primary_key=True, index=True)
    analysis_id = Column(String, ForeignKey("analyses.id"), index=True)
    paper_id = Column(String, ForeignKey("papers.id"), index=True)
    chunk_id = Column(String, unique=True, index=True)
    text = Column(String)
    source_field = Column(String)
    year = Column(Integer)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RAGRetrievalDB(Base):
    __tablename__ = "rag_retrievals"
    id = Column(String, primary_key=True, index=True)
    candidate_id = Column(String, ForeignKey("gap_candidates.candidate_id"), index=True)
    chunk_id = Column(String, ForeignKey("rag_chunks.chunk_id"))
    similarity_score = Column(Float)
    retrieval_rank = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class GroundedExplanationDB(Base):
    __tablename__ = "grounded_explanations"
    id = Column(String, primary_key=True, index=True)
    candidate_id = Column(String, ForeignKey("gap_candidates.candidate_id"), index=True)
    model_name = Column(String)
    prompt_version = Column(String)
    candidate_gap_statement = Column(String)
    evidence_summary = Column(String)
    literature_context = Column(String)
    limitation_evidence = Column(String)
    temporal_context = Column(String)
    counterevidence = Column(String)
    evidence_limitations = Column(String)
    suggested_direction = Column(String)
    grounding_status = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

"""Database models and connection management."""

import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class Run(Base):
    """Model for storing run data and cost predictions."""

    __tablename__ = "runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    model = Column(String, nullable=False, index=True)

    # Predicted values
    predicted_input_tokens = Column(Integer, nullable=False)
    predicted_output_tokens = Column(Integer, nullable=False)
    predicted_cost = Column(Float, nullable=False)

    # Actual values (nullable for prediction-only runs)
    actual_input_tokens = Column(Integer, nullable=True)
    actual_output_tokens = Column(Integer, nullable=True)
    actual_cost = Column(Float, nullable=True)

    # Error metrics
    error_percent = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Run(id={self.id[:8]}..., model={self.model}, "
            f"predicted_cost=${self.predicted_cost:.4f}, "
            f"actual_cost=${self.actual_cost:.4f if self.actual_cost else 'N/A'})>"
        )


class CalibrationMetadata(Base):
    """Model for storing calibration metadata and statistics."""

    __tablename__ = "calibration_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, nullable=False, index=True)
    token_bucket = Column(String, nullable=False, index=True)  # e.g., "0-1000", "1000-5000"
    
    # Statistics
    avg_error_percent = Column(Float, nullable=False)
    std_dev_error = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    
    # Learned parameters
    learned_output_ratio = Column(Float, nullable=True)
    
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<CalibrationMetadata(model={self.model}, bucket={self.token_bucket}, "
            f"avg_error={self.avg_error_percent:.2f}%)>"
        )


class DatabaseManager:
    """Manager for database connections and operations."""

    def __init__(self, database_path: str):
        """Initialize the database manager.

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine with connection pooling
        self.engine = create_engine(
            f"sqlite:///{self.database_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def init_db(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_all(self) -> None:
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Yields:
            SQLAlchemy Session object

        Example:
            with db_manager.get_session() as session:
                run = Run(...)
                session.add(run)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_run(
        self,
        model: str,
        predicted_input_tokens: int,
        predicted_output_tokens: int,
        predicted_cost: float,
        actual_input_tokens: Optional[int] = None,
        actual_output_tokens: Optional[int] = None,
        actual_cost: Optional[float] = None,
        error_percent: Optional[float] = None,
    ) -> Run:
        """Create a new run record.

        Args:
            model: Model name
            predicted_input_tokens: Predicted input tokens
            predicted_output_tokens: Predicted output tokens
            predicted_cost: Predicted total cost
            actual_input_tokens: Actual input tokens (optional)
            actual_output_tokens: Actual output tokens (optional)
            actual_cost: Actual total cost (optional)
            error_percent: Error percentage (optional)

        Returns:
            Created Run object
        """
        with self.get_session() as session:
            run = Run(
                model=model,
                predicted_input_tokens=predicted_input_tokens,
                predicted_output_tokens=predicted_output_tokens,
                predicted_cost=predicted_cost,
                actual_input_tokens=actual_input_tokens,
                actual_output_tokens=actual_output_tokens,
                actual_cost=actual_cost,
                error_percent=error_percent,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def get_run_by_id(self, run_id: str) -> Optional[Run]:
        """Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run object or None if not found
        """
        with self.get_session() as session:
            return session.query(Run).filter(Run.id == run_id).first()

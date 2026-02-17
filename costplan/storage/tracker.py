"""Run tracking and calibration data management."""

import statistics
from typing import Optional, List, Dict

from sqlalchemy import func, and_

from costplan.storage.database import DatabaseManager, Run, CalibrationMetadata
from costplan.core.predictor import PredictionResult
from costplan.core.calculator import ActualCostResult, calculate_error_percent
from costplan.config.settings import Settings


class RunTracker:
    """Tracks run data and manages calibration metadata."""

    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the run tracker.

        Args:
            database_manager: Database manager instance (creates default if None)
            settings: Settings instance (creates default if None)
        """
        self.settings = settings or Settings()
        self.db_manager = database_manager or DatabaseManager(
            str(self.settings.get_database_path())
        )
        # Initialize database
        self.db_manager.init_db()

    def close(self) -> None:
        """Release database connections. Call when done (e.g. in tests before deleting temp DB)."""
        self.db_manager.dispose()

    def store_run(
        self,
        prediction: PredictionResult,
        actual: Optional[ActualCostResult] = None,
        model: Optional[str] = None,
    ) -> Run:
        """Store a run record with prediction and actual data.

        Args:
            prediction: Prediction result
            actual: Actual cost result (optional, for prediction-only runs)
            model: Model name (uses prediction.model if None)

        Returns:
            Created Run object
        """
        model_name = model or prediction.model

        # Calculate error if actual data is available
        error_percent = None
        if actual:
            error_percent = calculate_error_percent(
                prediction.predicted_total_cost,
                actual.actual_total_cost,
            )

        # Create run record
        with self.db_manager.get_session() as session:
            run = Run(
                model=model_name,
                predicted_input_tokens=prediction.predicted_input_tokens,
                predicted_output_tokens=prediction.predicted_output_tokens,
                predicted_cost=prediction.predicted_total_cost,
                actual_input_tokens=actual.actual_input_tokens if actual else None,
                actual_output_tokens=actual.actual_output_tokens if actual else None,
                actual_cost=actual.actual_total_cost if actual else None,
                error_percent=error_percent,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            session.expunge(run)

        return run

    def get_recent_runs(
        self,
        limit: int = 10,
        model: Optional[str] = None
    ) -> List[Run]:
        """Get recent runs.

        Args:
            limit: Maximum number of runs to return
            model: Filter by model name (optional)

        Returns:
            List of Run objects
        """
        with self.db_manager.get_session() as session:
            query = session.query(Run).order_by(Run.timestamp.desc())

            if model:
                query = query.filter(Run.model == model)

            runs = query.limit(limit).all()
            # Detach from session
            session.expunge_all()
            return runs

    def get_error_stats(self, model: Optional[str] = None) -> Dict[str, float]:
        """Get error statistics.

        Args:
            model: Filter by model name (optional)

        Returns:
            Dict with avg_error, std_dev, min_error, max_error
        """
        with self.db_manager.get_session() as session:
            query = session.query(Run).filter(Run.error_percent.isnot(None))

            if model:
                query = query.filter(Run.model == model)

            runs = query.all()

            if not runs:
                return {
                    "avg_error": 0.0,
                    "std_dev": 0.0,
                    "min_error": 0.0,
                    "max_error": 0.0,
                    "sample_count": 0,
                }

            errors = [abs(run.error_percent) for run in runs]

            return {
                "avg_error": statistics.mean(errors),
                "std_dev": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                "min_error": min(errors),
                "max_error": max(errors),
                "sample_count": len(errors),
            }

    def get_rolling_error_average(
        self,
        model: str,
        window: Optional[int] = None
    ) -> Optional[float]:
        """Get rolling average error for a model.

        Args:
            model: Model name
            window: Number of recent runs to consider (uses settings default if None)

        Returns:
            Average absolute error percentage or None if no data
        """
        window_size = window or self.settings.calibration_window

        with self.db_manager.get_session() as session:
            runs = (
                session.query(Run)
                .filter(and_(Run.model == model, Run.error_percent.isnot(None)))
                .order_by(Run.timestamp.desc())
                .limit(window_size)
                .all()
            )

            if not runs:
                return None

            errors = [abs(run.error_percent) for run in runs]
            return statistics.mean(errors)

    def get_calibration_data(
        self,
        model: str,
        token_bucket: Optional[str] = None
    ) -> Optional[Dict]:
        """Get calibration data for a model.

        Args:
            model: Model name
            token_bucket: Token bucket (e.g., "0-1000", "1000-5000")

        Returns:
            Dict with calibration metadata or None
        """
        with self.db_manager.get_session() as session:
            query = session.query(CalibrationMetadata).filter(
                CalibrationMetadata.model == model
            )

            if token_bucket:
                query = query.filter(CalibrationMetadata.token_bucket == token_bucket)

            metadata = query.first()

            if not metadata:
                return None

            return {
                "model": metadata.model,
                "token_bucket": metadata.token_bucket,
                "avg_error_percent": metadata.avg_error_percent,
                "std_dev_error": metadata.std_dev_error,
                "sample_count": metadata.sample_count,
                "learned_output_ratio": metadata.learned_output_ratio,
                "last_updated": metadata.last_updated,
            }

    def update_calibration_metadata(
        self,
        model: str,
        token_bucket: str,
        avg_error: float,
        std_dev: float,
        sample_count: int,
        learned_output_ratio: Optional[float] = None,
    ) -> CalibrationMetadata:
        """Update or create calibration metadata.

        Args:
            model: Model name
            token_bucket: Token bucket
            avg_error: Average error percentage
            std_dev: Standard deviation of error
            sample_count: Number of samples
            learned_output_ratio: Learned output ratio (optional)

        Returns:
            Created or updated CalibrationMetadata object
        """
        with self.db_manager.get_session() as session:
            # Try to find existing metadata
            metadata = (
                session.query(CalibrationMetadata)
                .filter(
                    and_(
                        CalibrationMetadata.model == model,
                        CalibrationMetadata.token_bucket == token_bucket,
                    )
                )
                .first()
            )

            if metadata:
                # Update existing
                metadata.avg_error_percent = avg_error
                metadata.std_dev_error = std_dev
                metadata.sample_count = sample_count
                if learned_output_ratio is not None:
                    metadata.learned_output_ratio = learned_output_ratio
                metadata.last_updated = func.now()
            else:
                # Create new
                metadata = CalibrationMetadata(
                    model=model,
                    token_bucket=token_bucket,
                    avg_error_percent=avg_error,
                    std_dev_error=std_dev,
                    sample_count=sample_count,
                    learned_output_ratio=learned_output_ratio,
                )
                session.add(metadata)

            session.commit()
            session.refresh(metadata)
            return metadata

    def get_all_runs_count(self) -> int:
        """Get total number of runs.

        Returns:
            Total run count
        """
        with self.db_manager.get_session() as session:
            return session.query(func.count(Run.id)).scalar()

    def get_total_cost(self, model: Optional[str] = None) -> float:
        """Get total actual cost across all runs.

        Args:
            model: Filter by model name (optional)

        Returns:
            Total cost
        """
        with self.db_manager.get_session() as session:
            query = session.query(func.sum(Run.actual_cost))

            if model:
                query = query.filter(Run.model == model)

            total = query.scalar()
            return total or 0.0

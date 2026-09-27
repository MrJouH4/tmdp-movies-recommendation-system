from pathlib import Path
import pandas as pd

class MovieDataLoader:
    """Load and validate movie data."""

    REQUIRED_COLUMNS = {
        "id",
        "title",
        "genre",
        "overview",
    }

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )
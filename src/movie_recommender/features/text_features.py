from sklearn.feature_extraction.text import TfidfVectorizer

class MovieFeatureBuilder:
    """Build text features from movie metadata."""

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )


    def build_tags(self, df):
        tags = (
            df["genre"].fillna("")
            + " "
            + df["overview"].fillna("")
        )

        return tags

    def fit_transform(self, df):
        tags = self.build_tags(df)
        return self.vectorizer.fit_transform(tags)

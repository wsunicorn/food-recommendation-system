from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class ContentBasedRecommender:
    """Content-based recommender supporting TF-IDF and SBERT embeddings.

    Usage:
        rec = ContentBasedRecommender(method='tfidf')
        rec.fit(titles, descriptions)
        rec.recommend(idx, top_k=10)
    """

    def __init__(self, method: str = 'tfidf', model_name: str = 'all-MiniLM-L6-v2'):
        assert method in ('tfidf', 'sbert')
        self.method = method
        self.model_name = model_name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.embeddings = None

    def fit(self, titles: List[str], descriptions: Optional[List[str]] = None):
        texts = []
        if descriptions is None:
            texts = [str(t) for t in titles]
        else:
            texts = [f"{t} {d}" for t, d in zip(titles, descriptions)]

        if self.method == 'tfidf':
            # tuned TF-IDF: capture bigrams, ignore very rare tokens, increase features
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2), min_df=2)
            self.embeddings = self.vectorizer.fit_transform(texts)
        else:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. Install with pip install sentence-transformers")
            model = SentenceTransformer(self.model_name)
            self.embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def recommend(self, item_index: int, top_k: int = 10, exclude_self: bool = True, titles: Optional[List[str]] = None):
        """Return indices of top_k most similar items to item_index.

        If `titles` is provided (list of strings aligned with the corpus), results will be
        deduplicated by title so that the returned recommendations have unique titles.
        """
        if self.embeddings is None:
            raise RuntimeError("Model not fitted. Call fit() before recommend().")

        if hasattr(self.embeddings, 'toarray'):
            mat = self.embeddings
            sims = cosine_similarity(mat[item_index], mat).flatten()
        else:
            sims = cosine_similarity(self.embeddings[item_index: item_index+1], self.embeddings).flatten()

        if exclude_self:
            sims[item_index] = -1

        # sort candidates by similarity
        order = np.argsort(-sims)

        if titles is None:
            top_idx = order[:top_k]
            return top_idx, sims[top_idx]

        # Deduplicate by title: keep first occurrence per title
        seen = set()
        results = []
        scores = []
        query_title = titles[item_index] if 0 <= item_index < len(titles) else None
        for idx in order:
            if len(results) >= top_k:
                break
            # skip self or identical title to the query
            try:
                t = titles[int(idx)]
            except Exception:
                t = None
            if idx == item_index:
                continue
            if t is None:
                continue
            # skip if same title as query
            if query_title is not None and t == query_title:
                continue
            if t in seen:
                continue
            seen.add(t)
            results.append(int(idx))
            scores.append(float(sims[int(idx)]))

        return np.array(results, dtype=int), np.array(scores, dtype=float)

import math
from typing import List, Tuple, Dict, Set


class RecommenderEngine:
    """User-based collaborative filtering recommendation engine."""

    def __init__(self):
        # user -> item -> rating
        self.user_ratings: Dict[str, Dict[str, float]] = {}
        # item -> user -> rating
        self.item_ratings: Dict[str, Dict[str, float]] = {}
        self.all_items: Set[str] = set()

    def fit(self, ratings: List[Tuple[str, str, float]]):
        """Ingest list of (user_id, item_id, rating) tuples."""
        self.user_ratings.clear()
        self.item_ratings.clear()
        self.all_items.clear()

        for user, item, score in ratings:
            self.all_items.add(item)
            if user not in self.user_ratings:
                self.user_ratings[user] = {}
            self.user_ratings[user][item] = float(score)

            if item not in self.item_ratings:
                self.item_ratings[item] = {}
            self.item_ratings[item][user] = float(score)

    @staticmethod
    def _cosine_sim(profile_a: Dict[str, float], profile_b: Dict[str, float]) -> float:
        common = set(profile_a.keys()) & set(profile_b.keys())
        if not common:
            return 0.0

        dot = sum(profile_a[k] * profile_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in profile_a.values()))
        norm_b = math.sqrt(sum(v * v for v in profile_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_user_similarity(self, user_a: str, user_b: str) -> float:
        if user_a not in self.user_ratings or user_b not in self.user_ratings:
            return 0.0
        return self._cosine_sim(self.user_ratings[user_a], self.user_ratings[user_b])

    def recommend(self, target_user: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # Cold start fallback: recommend most popular items
        if target_user not in self.user_ratings:
            item_avg = [
                (item, sum(scores.values()) / len(scores))
                for item, scores in self.item_ratings.items()
            ]
            item_avg.sort(key=lambda x: x[1], reverse=True)
            return item_avg[:top_k]

        target_items = set(self.user_ratings[target_user].keys())
        unseen_items = self.all_items - target_items
        if not unseen_items:
            return []

        # Calculate similarity with all other users
        similarities = {}
        for other_user in self.user_ratings:
            if other_user != target_user:
                sim = self.get_user_similarity(target_user, other_user)
                if sim > 0:
                    similarities[other_user] = sim

        if not similarities:
            return []

        # Predict ratings for unseen items
        item_scores: Dict[str, float] = {}
        sim_sums: Dict[str, float] = {}

        for other_user, sim in similarities.items():
            for item, rating in self.user_ratings[other_user].items():
                if item in unseen_items:
                    item_scores[item] = item_scores.get(item, 0.0) + (sim * rating)
                    sim_sums[item] = sim_sums.get(item, 0.0) + sim

        rankings = [
            (item, round(item_scores[item] / sim_sums[item], 2))
            for item in item_scores
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:top_k]

from recommender.engine import RecommenderEngine

def test_recommendation_flow():
    engine = RecommenderEngine()
    data = [
        ("alice", "python_book", 5.0),
        ("alice", "ml_guide", 4.5),
        ("bob", "python_book", 5.0),
        ("bob", "docker_course", 4.0),
        ("charlie", "ml_guide", 4.0),
        ("charlie", "docker_course", 5.0)
    ]
    engine.fit(data)

    # Bob and Alice both like python_book, so Alice might be recommended docker_course
    recs = engine.recommend("alice", top_k=2)
    assert len(recs) >= 1
    assert recs[0][0] == "docker_course"

def test_cold_start_fallback():
    engine = RecommenderEngine()
    data = [
        ("u1", "item_a", 5.0),
        ("u2", "item_a", 5.0),
        ("u3", "item_b", 2.0)
    ]
    engine.fit(data)

    # New unobserved user
    recs = engine.recommend("new_user", top_k=1)
    assert len(recs) == 1
    assert recs[0][0] == "item_a"

"""Tests for evaluation."""
import torch
import numpy as np

from mphrqe.evaluation import (MACRO_AVERAGE, MICRO_AVERAGE,
                               RankingMetricAggregator, SetPrecisionAggregator,
                               _Ranks, filter_ranks, score_to_rank_multi_target)


def _test_score_to_rank_multi_target(average: str):
    """Actually test score_to_rank_multi_target."""
    generator = torch.manual_seed(seed=42)
    batch_size = 2
    num_entities = 7
    num_positives = 5
    scores = torch.rand(batch_size, num_entities, generator=generator)
    targets = torch.stack([
        torch.randint(high=batch_size, size=(num_positives,)),
        torch.randint(high=num_entities, size=(num_positives,)),
    ], dim=0)
    easy_targets = torch.empty(2, 0, dtype=torch.long)
    ranks_ = score_to_rank_multi_target(
        scores=scores,
        hard_targets=targets,
        easy_targets=easy_targets,
        average=average,
    )
    _verify_ranks(ranks_, average, num_entities, num_positives)


def _verify_ranks(ranks_: _Ranks, average: str, num_entities: int,
                  num_positives: int):
    for ranks in (ranks_.pessimistic, ranks_.optimistic, ranks_.realistic,
                  ranks_.expected_rank):
        assert ranks is not None
        assert ranks.shape == (num_positives,)
        assert (ranks >= 1).all()
        assert (ranks <= num_entities).all()
    assert (ranks_.optimistic <= ranks_.pessimistic).all()
    assert (ranks_.weight is None) == (average == MICRO_AVERAGE)


def test_score_to_rank_multi_target_micro():
    """Test score_to_rank_multi_target with micro averaging."""
    _test_score_to_rank_multi_target(average=MICRO_AVERAGE)


def test_score_to_rank_multi_target_macro():
    """Test score_to_rank_multi_target with macro averaging."""
    _test_score_to_rank_multi_target(average=MACRO_AVERAGE)


def test_score_to_rank_infinity():
    """Test score to rank with infinity scores."""
    batch_size = 2
    num_entities = 7
    num_positives = 5
    scores = torch.full(size=(batch_size, num_entities),
                        fill_value=float("inf"))
    targets = torch.stack([
        torch.randint(high=batch_size, size=(num_positives,)),
        torch.randint(high=num_entities, size=(num_positives,)),
    ], dim=0)
    easy_targets = torch.empty(2, 0, dtype=torch.long)
    ranks_ = score_to_rank_multi_target(
        scores=scores,
        hard_targets=targets,
        easy_targets=easy_targets,
        average=MACRO_AVERAGE,
    )
    _verify_ranks(ranks_, average=MACRO_AVERAGE, num_entities=num_entities,
                  num_positives=num_positives)


def test_score_to_rank_multi_target_manual():
    """Test score_to_rank_multi_target on a manual curated examples."""
    targets = torch.as_tensor(data=[[0, 0], [0, 1], [1, 0]]).t()
    easy_targets = torch.zeros(2, 0, dtype=torch.long)
    scores = torch.as_tensor(data=[
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 2.0, 3.0, 4.0],
    ])

    # Micro
    expected_expected_rank_micro = torch.as_tensor(data=[2.0, 2.0, 2.5])
    micro_ranks_ = score_to_rank_multi_target(
        scores=scores,
        hard_targets=targets,
        easy_targets=easy_targets,
        average=MICRO_AVERAGE,
    )
    assert torch.allclose(micro_ranks_.expected_rank,
                          expected_expected_rank_micro)
    assert micro_ranks_.weight is None

    # Macro
    expected_expected_rank_macro = torch.as_tensor(data=[2.0, 2.0, 2.5])
    macro_ranks_ = score_to_rank_multi_target(
        scores=scores,
        hard_targets=targets,
        easy_targets=easy_targets,
        average=MACRO_AVERAGE,
    )
    assert torch.allclose(macro_ranks_.expected_rank,
                          expected_expected_rank_macro)
    expected_weight = torch.as_tensor(data=[0.5, 0.5, 1.0])
    assert torch.allclose(macro_ranks_.weight, expected_weight)


def _test_evaluator(average: str):
    # reproducible testing
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    evaluator = RankingMetricAggregator(average=average)
    batch_size = 2
    num_entities = 5
    num_batches = 3
    nnz = 4
    num_non_empty_queries = 0
    for _ in range(num_batches):
        scores = torch.rand(batch_size, num_entities, device=device,
                            generator=generator)
        targets = torch.stack([
            torch.randint(high=batch_size, size=(nnz,), device=device,
                          generator=generator),
            torch.randint(high=num_entities, size=(nnz,), device=device,
                          generator=generator),
        ], dim=0)
        easy_targets = torch.empty(2, 0, dtype=torch.long, device=device)
        num_non_empty_queries += targets[0].unique().shape[0]
        evaluator.process_scores_(
            scores=scores,
            hard_targets=targets,
            easy_targets=easy_targets,
        )
    results = evaluator.finalize()
    if average == MICRO_AVERAGE:
        expected_num_ranks = num_batches * nnz
    elif average == MACRO_AVERAGE:
        expected_num_ranks = num_non_empty_queries
    else:
        raise ValueError(average)
    assert isinstance(results, dict)
    for key, value in results.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        if "num_ranks" in key:
            assert value == expected_num_ranks
        elif "adjusted_mean_rank_index" in key:
            assert -1 <= value <= 1
        elif "adjusted_mean_rank" in key:
            assert 0 < value < 2
        elif "mean_rank" in key:  # mean_rank, expected_mean_rank
            assert 1 <= value <= num_entities
        else:  # mean_reciprocal_rank, hits_at_k
            assert 0 <= value <= 1, key


def test_evaluator_micro_average():
    """Test evaluator with micro averaging."""
    _test_evaluator(average=MICRO_AVERAGE)


def test_evaluator_macro_average():
    """Test evaluator with macro averaging."""
    _test_evaluator(average=MACRO_AVERAGE)


def test_filter_ranks_manually():
    """Test filter_ranks."""
    # corner case: every rank is one, everything in same batch
    num_entities = 5
    ranks = torch.ones(size=(num_entities,), dtype=torch.long)
    batch_id = torch.zeros_like(ranks)

    filtered_rank = filter_ranks(
        ranks=ranks,
        batch_id=batch_id,
    )
    assert (filtered_rank >= 1).all()


def _test_set_based_precision_value(scores: torch.FloatTensor,
                                    hard_targets: torch.LongTensor,
                                    easy_targets: torch.LongTensor,
                                    threshold: float,
                                    value: float):
    """Test whether computed set-based precision is close to value"""
    agg = SetPrecisionAggregator(threshold=threshold)
    agg.process_scores_(scores, hard_targets, easy_targets)

    mean_precision = agg.finalize()['mean_precision']

    assert np.allclose(mean_precision, value)


def test_set_based_precision_half():
    """Test set-based precision in an example where precision is known."""
    # scores: (batch_size, num_entities)
    scores = torch.FloatTensor([[0.0, 0.9, 0.9, 0.9, 0.9],
                                [0.9, 0.9, 0.9, 0.9, 0.0]])

    # First row is batch ID, second row is answer entity e.
    # 0 <= e < num_entities
    hard_answers = torch.LongTensor([[0, 1],
                                     [1, 0]])
    easy_answers = torch.LongTensor([[0, 1],
                                     [2, 1]])

    _test_set_based_precision_value(scores,
                                    hard_answers,
                                    easy_answers,
                                    threshold=0.0,
                                    value=0.5)


def test_set_based_precision_zero():
    """Test set-based precision in an example where precision is zero due to
    not answers predicted at all."""
    # scores: (batch_size, num_entities)
    scores = torch.full((2, 5), fill_value=-1.0)

    # First row is batch ID, second row is answer entity e.
    # 0 <= e < num_entities
    hard_answers = torch.LongTensor([[0, 1],
                                     [1, 0]])
    easy_answers = torch.LongTensor([[0, 1],
                                     [2, 1]])

    _test_set_based_precision_value(scores,
                                    hard_answers,
                                    easy_answers,
                                    threshold=0.0,
                                    value=0.0)

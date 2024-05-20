"""Evaluation utilities."""
import dataclasses
import logging
from abc import abstractmethod
from typing import Collection, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

import numpy
import pandas
import torch
from torch.utils.data import DataLoader
from pykeen.typing import RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC, RANK_TYPES
from tqdm.auto import tqdm

from gqs.loader import QueryGraphBatch
# from .data.mapping import get_entity_mapper
from .loss import QueryEmbeddingLoss
from .models import QueryEmbeddingModel
from .similarity import Similarity
from .typing import FloatTensor, LongTensor

__all__ = [
    "SetPrecisionAggregator",
    "RankingMetricAggregator",
    "evaluate",
]

logger = logging.getLogger(__name__)

MICRO_AVERAGE = "micro"
MACRO_AVERAGE = "macro"


@dataclasses.dataclass
class _Ranks:
    """Rank results."""
    #: The optimistic rank (i.e. best among equal scores)
    optimistic: LongTensor

    #: The pessimistic rank (i.e. worst among equal scores)
    pessimistic: LongTensor

    #: The expected rank
    expected_rank: Optional[FloatTensor] = None

    # weight
    weight: Optional[FloatTensor] = None

    @property
    def realistic(self) -> FloatTensor:
        """Return the realistic rank, i.e. the average of optimistic and pessimistic rank."""
        return 0.5 * (self.optimistic + self.pessimistic).float()

    def __post_init__(self):
        """Error checking."""
        assert (self.optimistic > 0).all()
        assert (self.pessimistic > 0).all()


def compute_ranks_from_scores(
    scores: FloatTensor,
    positive_scores: FloatTensor,
) -> _Ranks:
    """
    Compute (unfiltered) ranks from a batch of scores.

    :param scores: shape: (batch_size, num_choices)
        The scores for all choices.
    :param positive_scores: (batch_size,)
        The scores for the true choice.

    :return:
        A rank object, comprising optimistic and pessimistic rank.
    """
    positive_scores = positive_scores.unsqueeze(dim=-1)
    best_rank = 1 + (scores > positive_scores).sum(dim=-1)
    worst_rank = (scores >= positive_scores).sum(dim=-1)
    return _Ranks(optimistic=best_rank, pessimistic=worst_rank)


def filter_ranks(
    ranks: LongTensor,
    batch_id: LongTensor,
) -> LongTensor:
    """
    Adjust ranks for filtered setting.

    Determines for each rank, how many smaller ranks there are in the same batch and subtracts this number. Notice that
    this requires that ranks contains all ranks for a certain batch which will be considered for filtering!

    :param ranks: shape: (num_choices,)
        The unfiltered ranks.
    :param batch_id: shape: (num_choices,)
        The batch ID for each rank.

    :return: shape: (num_choices,)
        Filtered ranks.
    """
    smaller_rank = ranks.unsqueeze(dim=0) < ranks.unsqueeze(dim=1)
    same_batch = batch_id.unsqueeze(dim=0) == batch_id.unsqueeze(dim=1)
    adjusted_ranks = ranks - (smaller_rank & same_batch).sum(dim=1)
    assert (adjusted_ranks > 0).all()
    return adjusted_ranks


def score_to_rank_multi_target(
    scores: FloatTensor,
    hard_targets: LongTensor,
    easy_targets: LongTensor,
    average: str = MICRO_AVERAGE,
) -> _Ranks:
    """
    Compute ranks, and optional weights for "macro" average.

    :param scores: shape: (batch_size, num_choices)
        The scores for all choices.
    :param hard_targets: shape: (2, num_hard_targets)
        Answers as pairs (batch_id, entity_id) that cannot be obtained with traversal
    :param easy_targets: shape: (2, num_easy_targets)
        Answers as pairs (batch_id, entity_id) that can be obtained with traversal
    :param average:
        'micro':
            Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'macro':
            Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into
            account.

    :return: shape: (num_true_choices,)
        A rank object comprising the filtered optimistic, realistic, pessimistic, and expected ranks, and weights in
        case macro is selected.
    """
    num_hard_targets = hard_targets.shape[1]

    # scores: (batch_size, num_entities)
    # targets: (2, nnz)
    targets = torch.cat((hard_targets, easy_targets), dim=-1)
    batch_id, entity_id = targets
    # get positive scores: shape: (nnz,)
    positive_scores = scores[batch_id, entity_id]

    # get unfiltered ranks: shape: (nnz,)
    ranks = compute_ranks_from_scores(scores=scores[batch_id], positive_scores=positive_scores)

    # filter ranks
    ranks.optimistic = filter_ranks(ranks=ranks.optimistic, batch_id=batch_id)
    ranks.pessimistic = filter_ranks(ranks=ranks.pessimistic, batch_id=batch_id)

    # Compute metrics for hard targets only
    ranks.optimistic = ranks.optimistic[:num_hard_targets]
    ranks.pessimistic = ranks.pessimistic[:num_hard_targets]

    # Compute expected rank with hard answers only
    batch_id = batch_id[:num_hard_targets]

    if average == MICRO_AVERAGE:
        uniq, counts = batch_id.unique(return_counts=True)
    elif average == MACRO_AVERAGE:
        # add sample weights such that all answers for the same query sum up to one
        uniq, inverse, counts = batch_id.unique(return_counts=True, return_inverse=True)
        ranks.weight = counts.float().reciprocal()[inverse]
    else:
        raise ValueError(f"Unknown average={average}")

    # expected filtered rank: shape: (nnz,)
    expected_rank = torch.full(size=(scores.shape[0],), fill_value=scores.shape[1], device=scores.device)
    expected_rank[uniq] -= counts
    expected_rank = 0.5 * (1 + 1 + expected_rank.float())
    expected_rank = expected_rank[batch_id]
    ranks.expected_rank = expected_rank
    return ranks


class ScoreAggregator:
    """An aggregator for scores."""

    @abstractmethod
    def process_scores_(
        self,
        scores: torch.FloatTensor,
        hard_targets: torch.LongTensor,
        easy_targets: torch.LongTensor
    ) -> None:
        """
        Process a batch of scores.

        Updates internal accumulator of ranks.

        :param scores: shape: (batch_size, num_choices)
            The scores for each batch element.
        :param hard_targets: shape: (2, nnz)
            The answer entities, in format (batch_id, entity_id) that cannot be obtained with traversal
        :param easy_targets: shape: (2, nnz)
            The answer entities, in format (batch_id, entity_id) that can be obtained with traversal
        """
        raise NotImplementedError

    def finalize(self) -> Mapping[str, float]:
        """
        Finalize aggregation and extract result.

        :return:
            A mapping from metric names to the scalar metric values.
        """
        raise NotImplementedError


class SetPrecisionAggregator(ScoreAggregator):
    """An aggregator for computing precision based on sets of predicted
    and true (easy and hard) answers.

    :param threshold: Scores above this threshold are considered predicted answers.
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._batch_sizes: List[int] = []
        self._mean_precisions: List[float] = []

    @torch.no_grad()
    def process_scores_(
            self,
            scores: FloatTensor,
            hard_targets: LongTensor,
            easy_targets: LongTensor,
    ) -> None:  # noqa: D102
        batch_size = scores.shape[0]
        batch_id, targets = torch.cat((hard_targets, easy_targets), dim=-1)
        true_answers = torch.zeros_like(scores, dtype=torch.bool)
        true_answers[batch_id, targets] = True

        pred_answers = scores > self.threshold
        num_pred_answers = pred_answers.sum(dim=-1)

        true_positives = torch.logical_and(pred_answers, true_answers)
        precision = true_positives.sum(dim=-1) / num_pred_answers

        # Map NaNs due to no predicted answers to 0 precision
        precision[num_pred_answers == 0] = 0

        self._batch_sizes.append(batch_size)
        self._mean_precisions.append(precision.mean().item())

    def finalize(self) -> Mapping[str, float]:  # noqa: D102
        batch_sizes = torch.tensor(self._batch_sizes)
        num_scores = batch_sizes.sum()
        mean_precision = torch.tensor(self._mean_precisions)
        mean_precision = (mean_precision * batch_sizes).sum() / num_scores

        result = {'num_scores': num_scores.item(),
                  'mean_precision': mean_precision.item()}

        return result


def _weighted_mean(
    tensor: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute weighted mean.

    :param tensor:
        The tensor.
    :param weight:
        An optional weight.

    :return:
        The (weighted) mean. If weight is None, uniform weights are assumed.
    """
    tensor = tensor.float()
    if weight is None:
        return tensor.mean()
    return (tensor * weight).sum() / weight.sum()


class _RankingMetricAggregator:
    """An aggregator for fixed rank-type ranking metrics."""

    def __init__(
        self,
        ks: Collection[int] = (1, 3, 5, 10),
    ):
        self.ks = ks
        self._data: List[Tuple[Union[int, float], ...]] = []

    def process_ranks(self, ranks: torch.Tensor, weight: Optional[torch.Tensor]) -> None:
        """Process a tensor of ranks, with optional weights."""
        assert (ranks > 0).all()
        self._data.append((
            ranks.shape[0] if weight is None else int(weight.sum()),  # assume sum to integer
            _weighted_mean(ranks, weight=weight).item(),
            _weighted_mean(ranks.float().reciprocal(), weight=weight).item(),
            *(
                _weighted_mean(ranks <= k, weight=weight).item()
                for k in self.ks
            ),
        ))

    def finalize(self) -> Mapping[str, float]:
        """Aggregates ranks into various single-figure metrics."""
        df = pandas.DataFrame(self._data, columns=[
            "batch_size",
            "mean_rank",
            "mean_reciprocal_rank",
            *(
                f"hits_at_{k}"
                for k in self.ks
            ),
        ])
        total = df["batch_size"].sum()
        result = dict(
            num_ranks=int(total),
        )
        for column in df.columns:
            if column == "batch_size":
                continue
            value = (df[column] * df["batch_size"]).sum() / total
            if numpy.issubdtype(value.dtype, numpy.integer):
                value = int(value)
            elif numpy.issubdtype(value.dtype, numpy.floating):
                value = float(value)
            result[column] = value
        return result


class RankingMetricAggregator(ScoreAggregator):
    """An aggregator for ranking metrics."""

    def __init__(
        self,
        ks: Collection[int] = (1, 3, 5, 10),
        average: str = MICRO_AVERAGE,
    ):
        """
        Initialize the aggregator.

        :param ks:
            The values for which to compute Hits@k.
        :param average:
            The average mode to use for computing aggregated metrics.
        """
        self.ks = ks
        self.average = average
        self._aggregators: MutableMapping[str, _RankingMetricAggregator] = {
            rank_type: _RankingMetricAggregator(ks=ks)
            for rank_type in RANK_TYPES
        }
        self._expected_ranks: List[torch.Tensor] = []
        self._expected_ranks_weights: List[Optional[torch.Tensor]] = []

    @torch.no_grad()
    def process_scores_(
        self,
        scores: FloatTensor,
        hard_targets: LongTensor,
        easy_targets: LongTensor,
    ) -> None:  # noqa: D102
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"Non-finite scores: {scores}")

        ranks = score_to_rank_multi_target(
            scores=scores,
            hard_targets=hard_targets,
            easy_targets=easy_targets,
            average=self.average,
        )
        self._aggregators[RANK_OPTIMISTIC].process_ranks(ranks.optimistic, weight=ranks.weight)
        self._aggregators[RANK_PESSIMISTIC].process_ranks(ranks.pessimistic, weight=ranks.weight)
        self._aggregators[RANK_REALISTIC].process_ranks(ranks.realistic, weight=ranks.weight)
        assert ranks.expected_rank is not None
        self._expected_ranks.append(ranks.expected_rank.detach().cpu())
        if ranks.weight is not None:
            self._expected_ranks_weights.append(ranks.weight.detach().cpu())

    def finalize(self) -> Mapping[str, float]:  # noqa: D102
        result: dict[str, float] = dict()
        for rank_type, agg in self._aggregators.items():
            for key, value in agg.finalize().items():
                result[f"{rank_type}.{key}"] = value
        self._aggregators.clear()
        # adjusted mean rank (index)
        if len(self._expected_ranks_weights) == 0 or any(w is None for w in self._expected_ranks_weights):
            weights = None
        else:
            weights = torch.cat(cast(List[torch.Tensor], self._expected_ranks_weights))
        expected_mean_rank = _weighted_mean(tensor=torch.cat(self._expected_ranks), weight=weights).item()
        result[f"{RANK_REALISTIC}.expected_mean_rank"] = expected_mean_rank
        result[f"{RANK_REALISTIC}.adjusted_mean_rank"] = result[f"{RANK_REALISTIC}.mean_rank"] / expected_mean_rank
        result[f"{RANK_REALISTIC}.adjusted_mean_rank_index"] = 1 - ((result[f"{RANK_REALISTIC}.mean_rank"] - 1) / (expected_mean_rank - 1))
        return result


@torch.no_grad()
def evaluate(
    data_loader: DataLoader[QueryGraphBatch],
    model: QueryEmbeddingModel,
    similarity: Similarity,
    loss: QueryEmbeddingLoss,
    threshold: float,
) -> Mapping[str, float]:
    """
    Evaluate query embedding model.

    :param data_loader:
        The validation data loader.
    :param model:
        The query embedding model instance.
    :param similarity:
        The similarity instance.
    :param loss:
        The loss instance.
    :param threshold:
        Scores above this threshold are considered answers.

    :return:
        A dictionary of results.
    """
    # set model into evaluation mode
    model.eval()
    ranking_evaluator = RankingMetricAggregator()
    precision_evaluator = SetPrecisionAggregator(threshold)
    validation_loss = torch.zeros(size=tuple(), device=model.device)
    batch: QueryGraphBatch
    for batch in tqdm(data_loader, desc="Evaluation", unit="batch", unit_scale=True):
        # embed query
        x_query = model(batch)
        # compute pairwise similarity to all entities, shape: (batch_size, num_entities)
        scores = similarity(x=x_query, y=model.x_e)
        # now compute the loss based on labels
        validation_loss += loss(scores, batch.hard_targets) * scores.shape[0]

        hard_targets = batch.hard_targets.to(model.device)
        easy_targets = batch.easy_targets.to(model.device)

        ranking_evaluator.process_scores_(scores=scores,
                                          hard_targets=hard_targets,
                                          easy_targets=easy_targets)
        precision_evaluator.process_scores_(scores=scores,
                                            hard_targets=hard_targets,
                                            easy_targets=easy_targets)
    return dict(
        loss=validation_loss.item() / len(data_loader),
        **ranking_evaluator.finalize(),
        **precision_evaluator.finalize()
    )


# @torch.no_grad()
# def evaluate_qualifier_impact(
#     data_loader: DataLoader[QueryGraphBatch],
#     model: QueryEmbeddingModel,
#     similarity: Similarity,
#     ks: Collection[int] = (1, 3, 5, 10),
#     average: str = MICRO_AVERAGE,
#     restrict_relations: Optional[Collection[int]] = None,
# ) -> pandas.DataFrame:
#     """
#     Evaluate the impact of qualifier pairs for each qualifier relation.

#     :param data_loader:
#         The evaluation data loader. batch_size = 1 is required.
#     :param model:
#         The model to evaluate.
#     :param similarity:
#         The similarity used to compute scores between query embedding and entity embeddings.
#     :param ks:
#         The values for which to compute Hits@k.
#     :param average:
#         The average to use for the scores.
#     :param restrict_relations:
#         If given, restrict evaluation to the relations for the given IDs.

#     :return: columns: metric | relation_id | type | value
#         The results as a dataframe.

#     :raise NotImplementedError:
#         For batch_size > 1.
#     """
#     # make our lives easier
#     if data_loader.batch_size is None or data_loader.batch_size > 1:
#         raise NotImplementedError("Batching is not implemented yet! Thus, pass a dataloader with batch_size=1.")

#     # set model into evaluation mode
#     model.eval()

#     # two evaluators for each relation: one having access to full information
#     evaluator: MutableMapping[int, Tuple[RankingMetricAggregator, RankingMetricAggregator]] = dict()

#     # make a (hash) set for faster existence checks
#     if restrict_relations is not None:
#         restrict_relations = set(restrict_relations)

#     batch: QueryGraphBatch
#     for batch in tqdm(data_loader, desc="Qualifier Impact Evaluation", unit="batch", unit_scale=True):
#         # guaranteed to be of batch_size = 1, i.e., contain only a single query

#         #: The targets, in format of pairs (graph_id, entity_id)
#         #: shape: (2, num_targets)
#         targets = batch.targets.to(model.device)

#         # note: since we require batch_size = 1, graph_id will always be zero
#         assert (targets[0] == 0).all()

#         # compute scores with full qualifier access
#         full_scores = similarity(x=model(batch), y=model.x_e)

#         # determine occurring qualifier relations
#         # note: these are the batch-local IDs!
#         local_qualifier_relations_in_batch = batch.qualifier_index[0].unique()

#         # store full qualifier index
#         full_qualifier_index = batch.qualifier_index

#         # restricted evaluation for each occurring relation
#         global_relation_ids = batch.relation_ids[local_qualifier_relations_in_batch].tolist()
#         for local_relation_id, relation_id in zip(
#             local_qualifier_relations_in_batch,
#             global_relation_ids,
#         ):
#             # If evaluation should be restricted to certain relations, skip relations if necessary
#             if restrict_relations is not None and relation_id not in restrict_relations:
#                 continue

#             # make sure that we have ranking aggregators
#             # note: we could initialize them beforehand, but then we would need to know the number of relations
#             evaluator.setdefault(
#                 relation_id, (
#                     RankingMetricAggregator(ks=ks, average=average),  # full
#                     RankingMetricAggregator(ks=ks, average=average),  # restricted
#                 ),
#             )
#             full_evaluator, restricted_evaluator = evaluator[relation_id]

#             # process full scores
#             full_evaluator.process_scores_(scores=full_scores, targets=targets)

#             # remove qualifier pairs for the currently considered relation
#             batch.qualifier_index = full_qualifier_index[:, full_qualifier_index[0] != local_relation_id]

#             # compute scores on restricted batch
#             restricted_scores = similarity(x=model(batch), y=model.x_e)

#             # process restricted scores
#             restricted_evaluator.process_scores_(scores=restricted_scores, targets=targets)

#     return pandas.DataFrame(data=[
#         (metric, relation_id, label, value)
#         for relation_id, evaluator_pair in evaluator.items()
#         for label, evaluator_ in zip(["full", "restricted"], evaluator_pair)
#         for metric, value in evaluator_.finalize().items()
#     ], columns=["metric", "relation_id", "type", "value"])

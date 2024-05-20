"""Graph pooling operations."""
from abc import abstractmethod
from typing import Optional

import torch
from class_resolver import Resolver
from torch import nn
from torch_scatter import scatter_add

from ..typing import FloatTensor

__all__ = [
    "graph_pooling_resolver",
    "GraphPooling",
]


class GraphPooling(nn.Module):
    """A module for graph pooling."""
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor],
    ) -> FloatTensor:
        """
        Obtain graph representations by aggregating node representations.

        :param x_e: shape: (num_nodes, dim)
            The node representations.
        :param graph_ids: shape: (num_nodes,)
            The graph ID for each node.
        :param entity_ids: shape: (num_nodes,)
            The global entity ID for each node.

        :return: shape: (num_graphs, dim)
            The graph representations.
        """
        raise NotImplementedError


class SumGraphPooling(GraphPooling):
    """Aggregation by sum."""

    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor] = None,
    ) -> FloatTensor:  # noqa: D102
        return scatter_add(x_e, index=graph_ids, dim=0)


class TargetPooling(GraphPooling):
    """Aggregation by sum."""

    def __init__(self, target_index: int, **kwargs) -> None:
        """
        Do the target pooling with respect to the given query target index.
        In StarQE, this was set to `get_entity_mapper().highest_entity_index + 1`
        """
        super().__init__()
        self._target_index = target_index

    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor] = None,
    ) -> FloatTensor:  # noqa: D102
        """
        graph_ids: binary mask
        """
        assert entity_ids is not None
        mask = entity_ids == self._target_index
        assert mask.sum() == graph_ids.unique().shape[0], "There should be exactly one target node per graph."
        return x_e[mask]


graph_pooling_resolver: Resolver[GraphPooling] = Resolver.from_subclasses(
    base=GraphPooling,  # type: ignore
    default=SumGraphPooling,
)

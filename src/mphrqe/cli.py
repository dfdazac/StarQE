"""Command line interface."""
import logging
import pathlib
import pickle
import pprint
import secrets
from typing import List, Mapping, Optional

import click
import torch
from class_resolver import Hint
from pykeen.utils import resolve_device
from .evaluation import RANK_REALISTIC
from torch import nn

from gqs.dataset import Dataset
from gqs.sample import resolve_sample
from gqs.loader import get_query_data_loaders

# from .data.config import binary_query_root
# from .data.loader import get_query_data_loaders, resolve_sample
# from .data.mapping import (
#     get_entity_mapper,
#     get_relation_mapper,
# )
from .evaluation import evaluate
from .hpo import optimize
from .layer.aggregation import (
    QualifierAggregation,
    qualifier_aggregation_resolver,
)
from .layer.composition import Composition, composition_resolver
from .layer.pooling import GraphPooling, graph_pooling_resolver
from .layer.util import activation_resolver
from .layer.weighting import MessageWeighting, message_weighting_resolver
from .loss import query_embedding_loss_resolver
from .models import StarEQueryEmbeddingModel
from .similarity import Similarity, similarity_resolver
from .tracking import init_tracker
from .training import optimizer_resolver, train_iter

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)

# Data options
option_dataset = click.option(
    "--dataset",
    type=Dataset,
    help="The name of the dataset. Must match [_a-z]+",
    required=True,
    #    callback=validate_dataset_name,
)
option_train_data = click.option(
    "-tr",
    "--train-data",
    type=str,
    multiple=True,
    help="The selector for the training data, e.g., /2hop/1qual:1000",
    default=[],
)
option_validation_data = click.option(
    "-va",
    "--validation-data",
    type=str,
    multiple=True,
    default=[],
)
option_test_data = click.option(
    "-te",
    "--test-data",
    type=str,
    multiple=True,
    default=[],
)
option_num_workers = click.option(
    "-nw",
    "--num-workers",
    type=int,
    default=0,
)

# Wandb options
option_use_wandb = click.option(
    "-w",
    "--use-wandb",
    is_flag=True,
)
option_wandb_project = click.option(
    "-wp",
    "--wandb-project",
    default=None
)
option_wandb_entity = click.option(
    "-we",
    "--wandb-entity",
    default=None
)
option_wandb_name = click.option(
    "-n",
    "--wandb-name",
    default=None,
)
option_wandb_group = click.option(
    "-wg",
    "--wandb-group",
    default=None,
)

# Logging options
option_log_level = click.option(
    "-ll",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)

# StarE encoder options
option_embedding_dim = click.option(
    "-d",
    "--embedding-dim",
    type=int,
    default=128,
)
option_num_layers = click.option(
    "-nl",
    "--num-layers",
    type=int,
    default=2,
)
option_num_layers_optional = click.option(
    "-nl",
    "--num-layers",
    type=int,
    default=None,
)
option_dropout = click.option(
    "-do",
    "--dropout",
    type=float,
    default=0.3,
)
option_activation = activation_resolver.get_option(
    "-a",
    "--activation",
    default=None,
    as_string=True,
)
option_composition = composition_resolver.get_option(
    "-c",
    "--composition",
    default=None,
    as_string=True,
)
option_qualifier_aggregation = qualifier_aggregation_resolver.get_option(
    "-qa",
    "--qualifier-aggregation",
    default=None,
    as_string=True,
)
option_qualifier_composition = composition_resolver.get_option(
    "-qc",
    "--qualifier-composition",
    default=None,
    as_string=True,
)
option_pooling = graph_pooling_resolver.get_option(
    "-gp",
    "--graph-pooling",
    default=None,
    as_string=True,
)
option_use_bias = click.option(
    "-b",
    "--use-bias",
    type=bool,
    default=False,
)
option_message_weighting = message_weighting_resolver.get_option(
    "-mw",
    "--message-weighting",
    default=None,
    as_string=True,
)
option_edge_dropout = click.option(
    "-ed",
    "--edge-dropout",
    type=float,
    default=0.0,
)
option_repeat_layers_until_diameter = click.option(
    "--repeat-layers-until-diameter",
    is_flag=True,
    default=False,
    help="""If set, the layers used in the message passing will be reused until max(batch.diameter) message passing steps have been performed.
             In effect, the weights will be shared.
            Note that the repetition will be each -num-layers, meaning that if the default is used, two layers are created and this pair will be reused.
        """,
)

option_stop_at_diameter = click.option(
    "--stop-at-diameter",
    is_flag=True,
    default=False,
    help="Stop the message propogation as soon as a number of steps have been performed equal to the diameter fo the query",
)
# Decoder options
option_similarity = similarity_resolver.get_option(
    "-s",
    "--similarity",
    default=None,
    as_string=True,
)

# Training + Optimizer options
option_epochs = click.option(
    "-e",
    "--epochs",
    type=int,
    default=1000,
)
option_learning_rate = click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.001,
)
option_threshold = click.option(
    '-th',
    '--threshold',
    type=float,
    default=0.0,
)
option_train_batch_size = click.option(
    "-tb",
    "--train-batch-size",
    type=int,
    default=32,
)
option_eval_batch_size = click.option(
    "-eb",
    "--eval-batch-size",
    type=int,
    default=32,
)
option_optimizer = optimizer_resolver.get_option(
    "-o",
    "--optimizer",
    default=None,
    as_string=True,
)
option_save = click.option(
    "-s",
    "--save",
    is_flag=True,
)
option_model_path = click.option(
    "-mp",
    "--model-path",
    type=pathlib.Path,
    default=None,
)

# HPO options
option_num_trials = click.option(
    "-nt",
    "--num-trials",
    type=int,
    default=None,
)
option_timeout = click.option(
    "-to",
    "--timeout",
    type=float,
    default=None,
)


@click.group()
def main():
    """The main entry point."""


@main.command(name="train")
# data options
@option_dataset
@option_train_data
@option_validation_data
@option_test_data
@option_num_workers
# wandb options
@option_use_wandb
@option_wandb_project
@option_wandb_entity
@option_wandb_name
@option_wandb_group
# logging options
@option_log_level
# encoder options
@option_embedding_dim
@option_num_layers
@option_dropout
@option_activation
@option_composition
@option_qualifier_aggregation
@option_qualifier_composition
@option_pooling
@option_use_bias
@option_message_weighting
@option_edge_dropout
@option_repeat_layers_until_diameter
@option_stop_at_diameter
# decoder options
@option_similarity
# optimizer + training options
@option_epochs
@option_learning_rate
@option_threshold
@option_train_batch_size
@option_eval_batch_size
@option_optimizer
# save options
@option_save
@option_model_path
def train_cli(
    # data
    dataset: Dataset,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_name: str,
    wandb_group: str,
    # logging
    log_level: str,
    # model
    embedding_dim: int,
    num_layers: int,
    dropout: float,
    activation: Hint[nn.Module],
    composition: Hint[Composition],
    qualifier_aggregation: Hint[QualifierAggregation],
    qualifier_composition: Hint[Composition],
    graph_pooling: Hint[GraphPooling],
    use_bias: bool,
    message_weighting: Hint[MessageWeighting],
    edge_dropout: float,
    repeat_layers_until_diameter: bool,
    stop_at_diameter: bool,
    # optimizer + training
    epochs: int,
    learning_rate: float,
    threshold: float,
    train_batch_size: int,
    eval_batch_size: int,
    optimizer: str,
    # decoder
    similarity: Hint[Similarity],
    # saving
    save: bool,
    model_path: Optional[pathlib.Path],
):
    """Train a single model for the given configuration."""
    # set log level
    logging.basicConfig(level=log_level)

    logger.info("Start loading data.")
    data_loaders, information = get_query_data_loaders(
        dataset=dataset,
        train=map(resolve_sample, train_data),
        validation=map(resolve_sample, validation_data),
        test=map(resolve_sample, test_data),
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    logger.info("Initializing tracker.")
    config = click.get_current_context().params
    result_callback = init_tracker(
        config=config,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        information=information.info,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        is_hpo=False,
    )

    model = StarEQueryEmbeddingModel(
        num_entities=dataset.entity_mapper.number_of_entities_and_reified_relations_without_vars_and_targets() + 3,  # we add a target, a variable and a blank node embedding
        num_relations=dataset.relation_mapper.get_largest_forward_relation_id() + 1,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        composition=composition,
        qualifier_aggregation=qualifier_aggregation,
        qualifier_composition=qualifier_composition,
        use_bias=use_bias,
        message_weighting=message_weighting,
        edge_dropout=edge_dropout,
        repeat_layers_until_diameter=repeat_layers_until_diameter,
        stop_at_diameter=stop_at_diameter,
        graph_pooling=graph_pooling,
        graph_pooling_kwargs=dict(target_index=dataset.entity_mapper.number_of_entities_and_reified_relations_without_vars_and_targets() + 1)
    )
    logger.info(f"Initialized model:\n{model}.")

    for epoch, epoch_result in enumerate(train_iter(
        model=model,
        data_loaders=data_loaders,
        similarity=similarity,
        optimizer=optimizer,
        optimizer_kwargs=dict(
            lr=learning_rate,
        ),
        num_epochs=epochs,
        result_callback=result_callback,
        threshold=threshold,
        early_stopper_kwargs=dict(
            key=("validation", f"{RANK_REALISTIC}.hits_at_10"),
            patience=5,
            relative_delta=0.02,
            larger_is_better=True,
            save_best_model=True,
        ) if validation_data else None,
    )):
        logger.info(f"Epoch: {epoch:5}/{epochs:5}: {epoch_result}")

    if save:
        model_path = model_path or pathlib.Path.home().joinpath(f"{secrets.token_hex()}.pt")
        model_path = model_path.expanduser().resolve()
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(dict(
            model=model,
            config=config,
            similarity=similarity,
            data=information,  # store to log train data
        ), model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path.as_uri()}")

    logger.info("Done")


@main.command(name="evaluate")
# data options
@option_dataset
@option_train_data
@option_validation_data
@option_test_data
@option_num_workers
# wandb options
@option_use_wandb
@option_wandb_project
@option_wandb_entity
@option_wandb_name
@option_wandb_group
# evaluation options
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
)
@option_threshold
# logging options
@option_log_level
# load options
@option_model_path
@click.option("--evaluate-faithfulness", default=False, is_flag=True)
def evaluate_cli(
    # data options
    dataset: Dataset,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_name: str,
    wandb_group: str,
    # evaluation
    batch_size: Optional[int],
    threshold: float,
    # logging
    log_level: str,
    # saving
    model_path: Optional[pathlib.Path],
    evaluate_faithfulness: bool,
):
    """Evaluate a trained model."""
    # set log level
    logging.basicConfig(level=log_level)

    if evaluate_faithfulness:
        assert len(train_data) == 0

    # Resolve device
    device = resolve_device(device=None)
    logger.info(f"Using device: {device}")

    # Load model
    if model_path is None:
        raise ValueError("Model path has to be provided for evaluation.")
    logger.info(f"Loading model from {model_path.as_uri()}")
    data = torch.load(model_path, map_location=device)
    model, config, train_information = [data[k] for k in ("model", "config", "data")]
    logger.info(
        f"Loaded model, trained on \n{pprint.pformat(train_information)}\n"
        f"using configuration \n{pprint.pformat(config)}\n.",
    )
    train_batch_size = int(config["train_batch_size"])
    if batch_size is None:
        logger.info(f"No batch size provided. Using the training batch size: {train_batch_size}")
        batch_size = train_batch_size
    elif batch_size > train_batch_size:
        logger.warning(f"Model was trained with batch size {train_batch_size}, but should be evaluated with a larger one: {batch_size}")

    # Load data
    logger.info("Loading evaluation data.")
    data_loaders, information = get_query_data_loaders(
        dataset=dataset,
        train=map(resolve_sample, train_data) if evaluate_faithfulness else [],
        validation=map(resolve_sample, validation_data),
        test=map(resolve_sample, test_data),
        eval_batch_size=batch_size,
        num_workers=num_workers,
    )
    logger.info(f"Evaluating on: \n{pprint.pformat(information.info)}\n")

    # instantiate decoder
    similarity = similarity_resolver.make(query=data["similarity"])
    logger.info(f"Instantiated similarity {similarity}")

    # instantiate loss
    loss_instance = query_embedding_loss_resolver.make(query=None)
    logger.info(f"Instantiated loss {loss_instance}")

    # Initialize tracking
    result_callback = init_tracker(
        config=config,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        information=information.info,
        is_hpo=False,
    )

    result: dict[str, Mapping[str, float]] = dict()
    for key, data_loader in data_loaders.items():
        if key == "train" and not evaluate_faithfulness:
            continue
        logger.info(f"Evaluating on {key}")
        result[key] = evaluate(
            data_loader=data_loader,
            model=model,
            similarity=similarity,
            loss=loss_instance,
            threshold=threshold,
        )
    logger.info(f"Evaluation result: {pprint.pformat(result, indent=2)}")
    if result_callback:
        result_callback(result)


METRICS = {
    f"{RANK_REALISTIC}.hits_at_10": "H@10",
    f"{RANK_REALISTIC}.mean_reciprocal_rank": "MRR",
    f"{RANK_REALISTIC}.adjusted_mean_rank_index": "AMRI",
}


@main.command(name="optimize")
@option_dataset
@option_train_data
@option_validation_data
@option_test_data
@option_use_wandb
@option_wandb_project
@option_wandb_entity
@option_wandb_name
@option_num_workers
@option_log_level
@option_num_trials
@option_timeout
@option_num_layers_optional
def optimize_cli(
    # data options
    dataset: Dataset,
    train_data: List[str],
    validation_data: List[str],
    test_data: List[str],
    num_workers: int,
    # wandb options
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_name: str,
    # logging options
    log_level: str,
    # optuna options
    num_trials: Optional[int],
    timeout: Optional[float],
    num_layers: Optional[int],
):
    """Optimize hyperparameters using optuna."""
    result = optimize(
        dataset=dataset,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        num_workers=num_workers,
        num_trials=num_trials,
        timeout=timeout,
        log_level=log_level,
        num_layers=num_layers,
    )
    print(f"Best model parameters = {result}")

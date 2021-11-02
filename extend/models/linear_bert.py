import warnings
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Any, Optional

from torch.utils.data.dataloader import default_collate

from transformers.optimization import get_linear_schedule_with_warmup

from flair.data import Sentence, Token
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import networkx as nx
import numpy as np
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel

from extend.data.processing import LinearBertProcessing
from extend.data.utils import Dictionary, overlaps, get_unused_node_name


def find_overlaps(edge_span, G):
    span_to_node = {}
    for node, d in G.nodes(data=True):
        if "span" in d:
            span_to_node[(1,) + tuple(d["span"])] = node
        elif "span2" in d:
            span_to_node[(2,) + tuple(d["span2"])] = node
    overlapping_nodes = []
    for span, node in span_to_node.items():
        if span[0] == edge_span[0] and overlaps(span[1:], edge_span[1:]):
            overlapping_nodes.append(node)

    return overlapping_nodes


class TwoStageLoss(nn.Module):
    def forward(self, node_logits, edge_logits, labels, loss_mask):
        bsz, length = labels.shape
        length -= 1

        true_node_type = labels[:, 0]
        edge_logits = edge_logits[torch.arange(len(labels)), :, true_node_type]

        edge_losses = nn.CrossEntropyLoss(reduction="none")(edge_logits.reshape(bsz*length, -1),
                                                            labels[:, 1:].reshape(bsz*length, -1).squeeze(dim=1))
        edge_loss = edge_losses[loss_mask.reshape_as(edge_losses)].sum()

        node_loss = nn.CrossEntropyLoss(reduction="sum")(node_logits, labels[:, 0])

        return (node_loss + edge_loss) / bsz


class UnconditionalLoss(nn.Module):
    def forward(self, node_logits, edge_logits, labels, loss_mask):
        bsz, length = labels.shape
        length -= 1

        edge_losses = nn.CrossEntropyLoss(reduction="none")(edge_logits.reshape(bsz*length, -1),
                                                            labels[:, 1:].reshape(bsz*length, -1).squeeze(dim=1))
        edge_loss = edge_losses[loss_mask.reshape_as(edge_losses)].sum()

        node_loss = nn.CrossEntropyLoss(reduction="sum")(node_logits, labels[:, 0])

        return (node_loss + edge_loss) / bsz



class BertForGraphGeneration(nn.Module):
    def __init__(self, num_labels, num_node_types, model, condition_on_node_type=True):
        super().__init__()
        self.condition_on_node_type = condition_on_node_type
        self.model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.2)
        self.num_labels = num_labels
        self.num_node_types = num_node_types
        self.node_classifier = nn.Linear(768, self.num_node_types)
        if self.condition_on_node_type:
            self.edge_classifier = nn.Linear(768, self.num_labels * self.num_node_types)
        else:
            self.edge_classifier = nn.Linear(768, self.num_labels)

    def forward(self, batch):
        if "token_type_ids" in batch:
            x = self.model.forward(
                input_ids=batch["input_ids"].long(),
                token_type_ids=batch["token_type_ids"].long(),
                attention_mask=batch["attention_mask"].long(),
            )[0]
        else:
            x = self.model.forward(
                input_ids=batch["input_ids"].long(),
                attention_mask=batch["attention_mask"].long(),
            )[0]

        x = self.dropout(x)
        node_logits = self.node_classifier(x[:, 0])
        bs, length = x.shape[:2]
        edge_logits = self.edge_classifier(x)
        if self.condition_on_node_type:
            edge_logits = edge_logits.reshape(
                (bs, length) + (self.num_node_types, self.num_labels)
            )

        return node_logits, edge_logits[:, 1:]


class LinearBert(pl.LightningModule):
    def __init__(
        self,
        transformer: str,
        num_epochs: int,
        lr: float,
        max_len: int,
        batch_size: int,
        edge_label_dict: Dictionary,
        node_label_dict: Dictionary,
        loss_type: str = "conditional",
        no_pretrained: bool = False,
        train_size=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.i = 0
        self.num_epochs = num_epochs
        self.lr = lr
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_size = train_size
        self.edge_label_dict = edge_label_dict
        self.node_label_dict = node_label_dict
        self.modified_f1 = pl.metrics.F1(num_classes=1)

        if loss_type == "unconditional":
            self.loss = UnconditionalLoss()
        elif loss_type == "conditional":
            self.loss = TwoStageLoss()
        else:
            raise ValueError(f"Invalid loss type {loss_type}")

        self.node_label_dict = node_label_dict
        self.edge_label_dict = edge_label_dict

        self.tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True)
        self.model = BertForGraphGeneration(
            model=transformer,
            num_labels=len(self.edge_label_dict),
            num_node_types=len(self.node_label_dict),
            condition_on_node_type=self.is_loss_conditional(),
        )
        if no_pretrained:
            self.model.model.init_weights()

        self.max_events_per_sentence = 10

    def collate(self, batch: List[Dict]):
        non_collatable_keys = {"G", "text"}
        truncatable_keys = {"input_ids", "token_type_ids", "attention_mask", "labels"}
        collatable = defaultdict(list)
        non_collatable = defaultdict(list)

        max_unpadded_length = 0
        for datapoint in batch:
            unpadded_length = torch.where(
                datapoint["input_ids"] == self.tokenizer.pad_token_id
            )[0][0]
            max_unpadded_length = max(unpadded_length, max_unpadded_length)

        for datapoint in batch:
            for k, v in datapoint.items():
                if k in non_collatable_keys:
                    non_collatable[k].append(v)
                else:
                    if not torch.is_tensor(v):
                        v = torch.tensor(v)
                    if k in truncatable_keys:
                        v = v[: max_unpadded_length + 1]
                    collatable[k].append(v)

        for k, v in collatable.items():
            collatable[k] = torch.stack(v)

        collatable.update(non_collatable)

        return dict(collatable)

    def forward(self, batch):
        return self.model(batch)

    def step(self, batch, batch_idx, prefix):
        batch_node_logits, batch_edge_logits = self.forward(batch)
        bsz, length = batch["labels"].shape
        length -= 1

        loss_mask = torch.ones((bsz, length), device=self.device).bool()
        for i, input_ids in enumerate(batch["input_ids"]):
            pad_start = torch.where(input_ids == 0)[0][0]
            loss_mask[i, pad_start - 1 :] = False
        loss = self.loss(batch_node_logits, batch_edge_logits, batch["labels"], loss_mask)
        node_correct = batch_node_logits.argmax(dim=1) == batch["labels"][:, 0]
        if self.is_loss_conditional():
            edge_logits = batch_edge_logits[torch.arange(bsz), :, batch["labels"][:, 0]]
        else:
            edge_logits = batch_edge_logits
        edge_correct = edge_logits.argmax(dim=2) == batch["labels"][:, 1:]
        aggregated_edge_correct = []
        for e_correct, mask in zip(edge_correct, loss_mask):
            aggregated_edge_correct.append(e_correct[mask].all())
        aggregated_edge_correct = torch.tensor(aggregated_edge_correct)

        is_correct = node_correct.cpu() & aggregated_edge_correct.cpu()

        is_modified_pred = batch_node_logits.argmax(dim=1) != 0
        is_modified_true = batch["labels"][:, 0] != 0

        # if is correctly modified: tp => pred 1 & true 1
        # if should be modified but isnt: fn => pred 0 & true 1
        # if shouldn't be modified but is: fp => pred 1 & true 0
        # if should be modified but is incorrectly modified: fp + fn => (pred 1 & true 0, pred 0 & true 1)

        dummy_preds = []
        dummy_trues = []
        for should_be_mod, was_mod, correct in zip(is_modified_true, is_modified_pred, is_correct):
            if should_be_mod and was_mod and correct:
                dummy_preds.append(1)
                dummy_trues.append(1)
            elif should_be_mod and not was_mod:
                dummy_preds.append(0)
                dummy_trues.append(1)
            elif not should_be_mod and was_mod:
                dummy_preds.append(1)
                dummy_trues.append(0)
            elif should_be_mod and was_mod and not correct:
                dummy_preds.append(1)
                dummy_trues.append(0)

                dummy_preds.append(0)
                dummy_trues.append(1)

        self.log(
            prefix + "_mod_f1",
            self.modified_f1(torch.tensor(dummy_preds, device=self.device),
                             torch.tensor(dummy_trues, device=self.device)),
            prog_bar=True
        )

        self.log(
            prefix + "_acc",
            torch.mean(is_correct.float()),
            prog_bar=True,
        )
        self.log(prefix + "_node_acc", torch.mean(node_correct.float()), prog_bar=False)
        self.log(prefix + "_edge_acc", torch.mean(edge_correct.float()), prog_bar=False)
        self.log(prefix + "_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, prefix="val")

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.modified_f1.reset()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.modified_f1.reset()

    def is_loss_conditional(self):
        return isinstance(self.loss, TwoStageLoss)


    def get_edges_from_labels(self, input_ids, labels, token_starts):
        edge_types = {}

        tags = [self.edge_label_dict.get_item_for_index(i.item()) for i in labels]
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids, skip_special_tokens=False
        )
        sentence = Sentence()
        for token, tag, start in zip(tokens[1:], tags, token_starts[1:]):
            token = Token(token.replace("##", ""), start_position=start)
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Edge", tag)

        i_sep = [tok.idx for tok in sentence if tok.text == "[SEP]"][0]
        for span in sentence.get_spans("Edge"):
            if span[0].idx < i_sep:
                edge_types[(1, span.start_pos, span.end_pos)] = span.tag
            else:
                edge_types[(2, span.start_pos, span.end_pos)] = span.tag

        return edge_types

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=0, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        return (self.train_size / self.batch_size) * self.num_epochs

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_joint_prob_per_node_type(self, node_logits, edge_logits):
        node_probs = torch.log(torch.softmax(node_logits, dim=0))
        edge_probs_per_node_type = torch.softmax(edge_logits, dim=2)
        max_sequences = edge_probs_per_node_type.argmax(dim=2)
        edge_probs_per_node_type = edge_probs_per_node_type.reshape(
            -1, len(self.edge_label_dict)
        )
        max_sequence_probs = edge_probs_per_node_type[
            torch.arange(len(edge_probs_per_node_type)), max_sequences.reshape(-1)
        ]
        max_sequence_probs = max_sequence_probs.reshape_as(max_sequences)
        joint_sequence_probs = torch.sum(torch.log(max_sequence_probs), dim=0)

        out = {}
        out["edge_logits"] = edge_logits.tolist()
        out["node_logits"] = node_logits.tolist()

        joint_probs = node_probs[: len(joint_sequence_probs)] + joint_sequence_probs

        return joint_probs

    def predict(self, text, linearize_graph, G_partial=None, beam_size=None):
        if self.is_loss_conditional():
            return self.predict_two_stage(text, linearize_graph, G_partial, beam_size)
        else:
            if beam_size and beam_size != 1:
                warnings.warn(
                    f"Beam size has no effect for unconditional prediction, but is set to {beam_size}. To avoid this warning, set it to 0 or 1."
                )
            return self.predict_unconditional(text, linearize_graph, G_partial)

    def predict_unconditional(
        self, text, linearize_graph, G_partial=None, beam_size=None
    ):
        G_pred = G_partial.copy() or nx.DiGraph()
        with torch.no_grad():
            for i in range(self.max_events_per_sentence):
                try:
                    example = LinearBertProcessing.build_example(
                        node=None,
                        G_full=G_pred,
                        known_nodes=G_pred.nodes,
                        text1=text,
                        config=self.config,
                        tokenizer=self.tokenizer,
                        edge_label_dict=self.edge_label_dict,
                        node_label_dict=self.node_label_dict,
                        linearize_graph=linearize_graph,
                        allow_truncation=True,
                    )
                except (IndexError, ValueError):
                    continue
                batch = self.collate([example])
                for k, v in batch.items():
                    try:
                        batch[k] = v.to(self.device)
                    except AttributeError:
                        pass

                pad_start = torch.where(batch["input_ids"] == 0)[1][0]
                node_logits, edge_logits = self.forward(batch)
                node_logits = node_logits.squeeze()
                edge_logits = edge_logits.squeeze()
                pred_node_type = self.node_label_dict.get_item_for_index(
                    node_logits.argmax()
                )
                pred_edge_labels = edge_logits.argmax(dim=1)[: pad_start - 1]
                pred_edges = self.get_edges_from_labels(
                    input_ids=example["input_ids"][:pad_start],
                    labels=pred_edge_labels,
                    token_starts=example["offset_mapping"][:pad_start, 0]
                    .long()
                    .tolist(),
                )
                if pred_node_type == "None":
                    break
                else:
                    pred_node_name = get_unused_node_name(node_type=pred_node_type, G=G_pred)
                    G_pred.add_node(
                        pred_node_name,
                        label=pred_node_type,
                        known=True,
                        span2=(-1, -1),
                        step=i
                    )  # span is needed for find_overlaps but will be updated only after a new call to linearization
                    for edge_span, edge_type in pred_edges.items():
                        dst_nodes = find_overlaps(edge_span=edge_span, G=G_pred)
                        is_text_span = edge_span[0] == 1

                        if (
                            is_text_span and not dst_nodes
                        ):  # node has to be created first
                            anchor_text = text[edge_span[1] : edge_span[2]]
                            anchor_name = get_unused_node_name("anchor", G_pred)
                            G_pred.add_node(
                                anchor_name,
                                text=anchor_text,
                                label="anchor",
                                span=edge_span[1:],
                                step=i
                            )
                            dst_nodes.append(anchor_name)

                        for dst_node in dst_nodes:
                            G_pred.add_edge(
                                pred_node_name,
                                dst_node,
                                label=edge_type,
                                known=True,
                            )

        return G_pred

    def predict_two_stage(self, text, linearize_graph, G_partial=None, beam_size=None):
        beam_size = beam_size or 5
        G_partial = G_partial or nx.DiGraph()
        G_beams = [(G_partial, 0.0, None)]
        with torch.no_grad():
            for i in range(self.max_events_per_sentence):
                if all(i[2] == "None" for i in G_beams):
                    break
                G_beams_new = []
                G_beams_valid = []
                beam_examples = []
                for G_pred, prob_G, _ in G_beams:
                    try:
                        example = LinearBertProcessing.build_example(
                            node=None,
                            G_full=G_pred,
                            known_nodes=G_pred.nodes,
                            text1=text,
                            tokenizer=self.tokenizer,
                            edge_label_dict=self.edge_label_dict,
                            node_label_dict=self.node_label_dict,
                            linearize_graph=linearize_graph,
                            allow_truncation=True,
                            max_len=self.max_len
                        )
                    except (IndexError, ValueError):
                        continue
                    beam_examples.append(example)
                    G_beams_valid.append((G_pred, prob_G))

                if not beam_examples:
                    raise ValueError
                batch = self.collate(beam_examples)
                for k, v in batch.items():
                    try:
                        batch[k] = v.to(self.device)
                    except AttributeError:
                        pass

                pad_start = torch.where(batch["input_ids"] == 0)[1][0]
                node_logits, edge_logits = self.forward(batch)
                for node_logit, edge_logit, (G_pred, prob_G), example in zip(
                    node_logits, edge_logits, G_beams_valid, beam_examples
                ):
                    joint_probs = self.get_joint_prob_per_node_type(
                        node_logit, edge_logit[:pad_start]
                    )
                    for pred_node_type_idx, prob_step in enumerate(joint_probs):
                        G_pred_beam_step = G_pred.copy()
                        prob = prob_G + prob_step.item()
                        pred_node_type = self.node_label_dict.get_item_for_index(
                            pred_node_type_idx
                        )
                        pred_edge_labels = edge_logit[:, pred_node_type_idx].argmax(
                            dim=1
                        )[: pad_start - 1]
                        pred_edges = self.get_edges_from_labels(
                            input_ids=example["input_ids"][:pad_start],
                            labels=pred_edge_labels,
                            token_starts=example["offset_mapping"][:pad_start, 0]
                            .long()
                            .tolist(),
                        )
                        if pred_node_type != "None" and pred_edges:
                            pred_node_name = get_unused_node_name(
                                node_type=pred_node_type, G=G_pred_beam_step
                            )
                            G_pred_beam_step.add_node(
                                pred_node_name,
                                label=pred_node_type,
                                known=True,
                                span2=(-1, -1),
                                step=i
                            )  # span is needed for find_overlaps but will be updated only after a new call to linearization
                            for edge_span, edge_type in pred_edges.items():
                                dst_nodes = find_overlaps(
                                    edge_span=edge_span, G=G_pred_beam_step
                                )
                                is_text_span = edge_span[0] == 1

                                if (
                                    is_text_span and not dst_nodes
                                ):  # node has to be created first
                                    anchor_text = text[edge_span[1] : edge_span[2]]
                                    anchor_name = get_unused_node_name(
                                        "anchor", G_pred_beam_step
                                    )
                                    G_pred_beam_step.add_node(
                                        anchor_name,
                                        text=anchor_text,
                                        label="anchor",
                                        span=edge_span[1:],
                                        step=i
                                    )
                                    dst_nodes.append(anchor_name)

                                for dst_node in dst_nodes:
                                    G_pred_beam_step.add_edge(
                                        pred_node_name,
                                        dst_node,
                                        label=edge_type,
                                        known=True,
                                    )
                        G_beams_new.append((G_pred_beam_step, prob, pred_node_type))

                G_beams = sorted(G_beams_new, reverse=True, key=itemgetter(1))[
                    :beam_size
                ]
                pass

        return G_beams[0][0]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

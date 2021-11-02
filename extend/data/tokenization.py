import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Tuple

from segtok.segmenter import split_multi

@dataclass
class Sentence:
    text: str
    start_position: int

class SentenceSplitter(ABC):
    r"""An abstract class representing a :class:`SentenceSplitter`.

    Sentence splitters are used to represent algorithms and models to split plain text into
    sentences and individual tokens / words. All subclasses should overwrite :meth:`splits`,
    which splits the given plain text into a sequence of sentences (:class:`Sentence`). The
    individual sentences are in turn subdivided into tokens / words. In most cases, this can
    be controlled by passing custom implementation of :class:`Tokenizer`.

    Moreover, subclasses may overwrite :meth:`name`, returning a unique identifier representing
    the sentence splitter's configuration.
    """

    @abstractmethod
    def split(self, text: str) -> List[Sentence]:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SegtokSentenceSplitter(SentenceSplitter):
    """
    Implementation of :class:`SentenceSplitter` using the SegTok library.

    For further details see: https://github.com/fnl/segtok
    """

    def __init__(self):
        super(SegtokSentenceSplitter, self).__init__()

    def split(self, text: str) -> List[Sentence]:
        sentences = []
        offset = 0

        plain_sentences = split_multi(text)
        for sentence in plain_sentences:
            sentence_offset = text.find(sentence, offset)

            if sentence_offset == -1:
                raise AssertionError(
                    f"Can't find offset for sentences {plain_sentences} "
                    f"starting from {offset}"
                )

            sentences += [
                Sentence(
                    text=sentence,
                    start_position=sentence_offset,
                )
            ]

            offset += len(sentence)

        return sentences

    @property
    def name(self) -> str:
        return self.__class__.__name__



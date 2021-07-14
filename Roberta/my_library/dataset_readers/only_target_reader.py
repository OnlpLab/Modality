
from typing import Dict, List, Union
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import sys
logger = logging.getLogger(__name__)

DEFAULT_WORD_TAG_DELIMITER = "###"


@DatasetReader.register("only_target_reader")
class TextClassificationJsonReader(DatasetReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "text" field and a "label" field in JSON format.
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    Registered as a `DatasetReader` with name "text_classification_json".
    [0]: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional
        optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences : `bool`, optional (default = `False`)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences, like [the Hierarchical
        Attention Network][0].
    max_sequence_length : `int`, optional (default = `None`)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing : `bool`, optional (default = `False`)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    """

    def __init__(
        self,
        word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        segment_sentences: bool = False,
        max_sequence_length: int = 510,
        skip_label_indexing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__( **kwargs)
        self._word_tag_delimiter = word_tag_delimiter
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line_numb, line in enumerate(data_file.readlines()):
                if not line:
                    continue
                line = line.split()
                text = []
                labels = []
                targets = []
                for token in line:
                    tok, tag, head = token.split(self._word_tag_delimiter)
                    text.append(tok)
                    labels.append(tag)
                    if head != 'O':
                        targets.append(tok)
                # mask_indices, label
                out = []
                for ind, label in enumerate(labels):
                    if label != 'O':
                        tag, sense = label.split('-')
                        if tag == 'S':
                            out.append([[ind], sense])
                        elif tag == 'B':
                            out.append([[ind], sense])
                        else:
                            out[-1][0].append(ind)
                for out_instance in out:
                    indices = out_instance[0]
                    newtext = ' '.join(text)
                    newtext = newtext.split()
                    masked_words = ' '.join(newtext[indices[0]:indices[-1]+1])
                    newtext = newtext[:indices[0]]+['<mask>']+newtext[indices[-1]+1:]
                    out_text = ' '.join(newtext)
                    out_heads = ' '.join(targets)
                    label = out_instance[1]
                    indices = [str(ind) for ind in indices]
                    if len(label) > 0 and len(text)>7:
                        yield self.text_to_instance(text=masked_words+' '+out_heads, label=label, sent_id=line_numb, indices=' '.join(indices), masked_words = masked_words)

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(
        self, text: str, sent_id, indices, masked_words, label: Union[str, int] = None
    ) -> Instance:  # type: ignore
        """
        # Parameters
        text : `str`, required.
            The text to classify
        label : `str`, optional, (default = `None`).
            The label for this text.
        # Returns
        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`LabelField`) :
              The label label of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self._skip_label_indexing)
        fields["metadata"] = MetadataField({"sent_id": sent_id, "indices": indices, "masked_words": masked_words, "gold_label": label})
        return Instance(fields)
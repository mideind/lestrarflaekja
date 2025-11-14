# pylint: disable=unused-import,unused-argument,W0611,logging-fstring-interpolation,not-callable
from typing import NamedTuple
from dataclasses import dataclass
import sys

import torch
from loguru import logger
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import rich
import pickle


example_texts = [
    """Heit og geislandi uppnumin í sæluvímu. Ást og aðdáun allra þeirra sem nutu hennar mettaði sál hennar dýrðlegri fullnægju, sem sökk niður í djúp vit­undarinnar. Fullnægju sem varð að óslökkvandi, já óseðjandi þrá,sem hvatti hana til dáða.Fékk hana til að skína og skína. Já, hvort hún skyldi skína í allri sinni dýrð!

Skyndilega féll skuggi á þessa dásamlegu stund. Það fór hreinlega um hana. Já, hún skalf við tilhugsunina. Skyndilega varð henni ískalt, þrátt fyrir að henni væri alltaf heitt í hamsi. Auðvitað gat hann ekki látið hana í friði. Þessi andstyggilegi ísmeygilegi andstæðingur. Hann hafði, svo lengi sem hana rak minni til , reynt að gera lítið úr henni. Hún vissi alveg hvað fór svona í taugarnar á honum. Ó,já hann var hreinlega að sturlast úr öfund. Hún vissi nákvæmlega hvað hann þráði heitt þá ást og aðdáun sem hún var aðnjótandi. Guð hafði svo sannarlega haft á réttu að standa þegar hann fullyrti að langdregin þrá geri hjartað sjúkt. Hún hafði alltaf fyrirlitið þennan óvildarmann sinn en kannski var honum bara vorkunn."""
]


@dataclass
class InferConfig:
    """Configuration for the project."""

    # model_id: str = "mideind/byt5-large-spanmask-pretrain"
    model_id: str = "mideind/byt5-large-spanmask-pretrain"
    # legal values are 1-30
    mask_length: int = 25
    allow_span_length_hints: bool = False
    # discard the scores of bytes near the edges of the target span,
    # since they are typically partially revealed words
    margin: int = 0


@dataclass
class ScoredToken:
    """A byte with its score."""

    text: str
    index: int
    start: int
    end: int
    rms_score: float
    peak_score: float
    avg_score: float


@dataclass
class ScoredChunk:
    start: int
    end: int
    scores: torch.Tensor
    byte_ids: torch.Tensor


@dataclass
class AnnotatedExample:
    """Text that has been tokenized and scored by a model. Each byte has a score.
    Scores are exportable to a JSON (for visualization) and pickle (for further processing).
    """

    text: str
    byte_ids: torch.Tensor
    scored_chunks: list["ScoredChunk"]
    scores: torch.Tensor
    scored_tokens: list[ScoredToken]

    def save_to_file(self, filepath: str) -> None:
        """Save the scored example to a file.

        Args:
            filepath: path to the output file
        """
        # torch.save(self, filepath)
        # use pickle ourselves to avoid issues with torch.save
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> "AnnotatedExample":
        """Load a scored example from a file.

        Args:
            filepath: path to the input file
        Returns:
            ScoredExample: the loaded scored example
        """
        # return torch.load(filepath)
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def from_scored_chunks(
        cls,
        text: str,
        byte_ids: torch.Tensor,
        byte_scores: torch.Tensor,
        scored_chunks: list["ScoredChunk"],
    ) -> "AnnotatedExample":
        """Create a ScoredExample from scored chunks.

        Args:
            text: the original text
            byte_scores: scores for each byte in the text
            scored_chunks: list of scored chunks
        Returns:
            ScoredExample: the created scored example
        """
        # the first word does not have leading space
        word_strings = [" " + w if i > 0 else w for i, w in enumerate(text.split())]
        word_lens = [len(w.encode("utf8")) for w in word_strings]
        word_offsets = torch.cumsum(torch.tensor([0] + word_lens), dim=0)
        # word_byte_scores = []
        scored_words = []

        assert all(
            chunk_curr.start < chunk_next.start
            for chunk_curr, chunk_next in zip(scored_chunks, scored_chunks[1:])
        )

        # TODO: make sure we aren't supposed to shift by one to scores to ids
        byte_cursor = 0
        for word_index, (word_start, word_end, word_str) in enumerate(
            zip(word_offsets[:-1], word_offsets[1:], word_strings)
        ):
            # accumulate byte scores for this word
            word_str = word_strings[word_index]

            word_scores = byte_scores[word_start:word_end]
            avg_word_score = word_scores.mean()
            peak_word_score = word_scores.max()
            rms_word_score = (word_scores**2).mean().sqrt()

            byte_cursor += word_end - word_start
            scored_word = ScoredToken(
                text=word_str,
                index=word_index,
                start=int(word_start),
                end=int(word_end),
                avg_score=avg_word_score.item(),
                peak_score=peak_word_score.item(),
                rms_score=rms_word_score.item(),
            )
            scored_words.append(scored_word)

        return cls(
            text=text,
            byte_ids=byte_ids,
            scored_chunks=scored_chunks,
            scores=byte_scores,
            scored_tokens=scored_words,
        )


class SpanInfillingScorerResult(NamedTuple):
    """Return type for span infilling scorer."""

    text: str
    byte_ids: torch.Tensor
    scores: torch.Tensor
    scored_chunks: list[ScoredChunk]


@dataclass
class SpanInfillingScorer:
    cfg: InferConfig
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer

    @classmethod
    def from_config(cls, cfg: InferConfig):
        byte_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        # TODO: device map
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id)
        return cls(cfg=cfg, model=model, tokenizer=byte_tokenizer)

    def score_string(self, text: str) -> SpanInfillingScorerResult:
        """Score the input text string using span infilling.
        We divide the text into infillable chunks, and score each chunk.
        We aggregate the scores, keeping only scores when a byte was masked (and reconstructed).

        TODO: use stride so we don't count partially revealed words.

        Args:
            text: input text string
        """

        # we are assuming no overlap (stride=mask_length) for now
        # the mask sequence without hints is "<MASK>" (in upper case),
        # the mask sequence with length hint is f"<MASK_{length}>"
        logger.info(text)
        byte_ids = torch.tensor(self.tokenizer(text).input_ids)  # type: ignore[operator]
        assert isinstance(byte_ids, torch.Tensor)

        byte_ids_unshifted = torch.tensor(self.tokenizer(text).input_ids)  # type: ignore[operator]
        assert len(byte_ids_unshifted.shape) == 1
        # (T) → (B × T)
        byte_ids_unshifted = byte_ids_unshifted.unsqueeze(0)

        # ByT5/T5 implementation shifts the label sequence internally
        # https://huggingface.co/docs/transformers/en/model_doc/byt5
        labels_unshifted = byte_ids_unshifted.clone()

        mask_str_wo_length_hint = "<MASK>"

        mask_seq = self.tokenizer(mask_str_wo_length_hint).input_ids  # type: ignore[operator]
        mask_seq = torch.tensor(mask_seq[:-1])  # remove the EOS token
        mask_seq = mask_seq.unsqueeze(0)  # remove the EOS token
        logger.debug(f"mask_seq: {mask_seq}")

        idxs = list(range(0, byte_ids_unshifted.numel(), self.cfg.mask_length // 2))
        # add end point of last interval
        if idxs[-1] < len(byte_ids_unshifted) - 5:
            idxs.append(len(byte_ids_unshifted))

        chunk_intervals = list(zip(idxs[:-1], idxs[1:]))
        logger.debug(f"{chunk_intervals=}")

        scores_byte_infilling = torch.zeros_like(byte_ids_unshifted, dtype=torch.float)
        # since our intervals overlap we need to track how often we scored each byte
        scores_denom = torch.zeros_like(byte_ids_unshifted, dtype=torch.float)
        scored_chunks = []

        for _chunk_idx, (loc_span_start, loc_span_end) in enumerate(chunk_intervals):
            # shape: (T)
            prefix = byte_ids_unshifted[:loc_span_start]
            suffix = byte_ids_unshifted[loc_span_end:]
            # middle
            target_ids = byte_ids_unshifted[loc_span_start:loc_span_end]

            logger.debug(_chunk_idx)
            input_ids_w_masking = torch.cat([prefix, mask_seq, suffix], dim=1)

            logger.debug(input_ids_w_masking.shape)
            # (T) → (B × T)
            input_ids_w_masking = input_ids_w_masking.unsqueeze(0)

            out = self.model(input_ids=input_ids_w_masking, labels=labels_unshifted)  # type: ignore[operator]
            # out.logits shape: (B × T × V)
            assert out.logits[:, 0].numel() == 1
            # (B × T × V) → (T × V)
            logits = out.logits.cpu().squeeze(0)

            span_logits = logits[loc_span_start:loc_span_end]
            target_scores = span_logits.gather(
                index=target_ids.unsqueeze(-1), dim=1
            ).squeeze(-1)

            foo = span_logits.gather(index=target_ids, dim=1)
            logger.debug(f"foo shape: {foo.shape}")

            scores_byte_infilling[loc_span_start:loc_span_end] += target_scores
            scores_denom[loc_span_start:loc_span_end] += 1.0

            # chunk_scores = chunk_scores.gather(index=byte_ids_unshifted, dim=2)
            # store chunk info (for possible later analysis)
            scored_chunk = ScoredChunk(
                start=loc_span_start,
                end=loc_span_end,
                scores=target_scores,
                byte_ids=target_ids,
            )
            scored_chunks.append(scored_chunk)

        # make sure we don't divide by zero when calculating average
        scores_denom = torch.clamp(scores_denom, min=1.0)
        # calculate average
        scores_byte_infilling = scores_byte_infilling / scores_denom

        return SpanInfillingScorerResult(
            text=text,
            byte_ids=byte_ids_unshifted,
            scores=scores_byte_infilling.squeeze(0),
            scored_chunks=scored_chunks,
        )


def do_main(cfg: InferConfig):
    scorer = SpanInfillingScorer.from_config(cfg=cfg)

    text = example_texts[0][:100]
    result = scorer.score_string(text=text)
    logger.info(result)

    scored_example = AnnotatedExample.from_scored_chunks(
        text=text,
        byte_ids=result.byte_ids,
        byte_scores=result.scores,
        scored_chunks=result.scored_chunks,
    )
    logger.info(scored_example)

    # test save/load
    torch.serialization.add_safe_globals([AnnotatedExample])
    scored_example.save_to_file("scored_example.pt")
    loaded_example = AnnotatedExample.load_from_file("scored_example.pt")
    logger.info(f"Loaded example successfully: {loaded_example}")

    rich.print(loaded_example)


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(InferConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    try:
        cfg = InferConfig(**cfg)  # type: ignore[arg-type]
    except Exception as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    do_main(cfg)


if __name__ == "__main__":
    main()

import sys

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from icecream import ic

from utils import (
    DataConfig,
    chunk_text_by_word_count,
    transform_example_word_noise,
    transform_example_word_soup,
    transform_vanilla,
    PAT_ALPHANUMERIC,
    remove_non_alphanumeric,
    PAT_MULTISPACE,
    collapse_multispace,
    normalize_and_make_auxiliary,
)

from utils import (
    TrainConfig,
    tokenizer_fn,
)

model_name = "AI-Sweden-Models/gpt-sw3-126m"
# model_name = "AI-Sweden-Models/gpt-sw3-356m"
# model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
model = AutoModelForCausalLM.from_pretrained(model_name)

enc = AutoTokenizer.from_pretrained(model_name)
cfg = DataConfig()

# 2000 samræmt próf  7. bekkur íslensku
example_text = """Er mönnum fjölgaði breyttist land hraðar vegna aukins landbúnaðar, námuvinnslu og stærri búsetusvæða. Nýjar stéttir komu fram og á 18. öld hófst iðnbyltingin í Bretlandi. Vélar komu í stað handanna einna, kol og gufa kom í stað dráttardýra og fólk fluttist í þéttbýliskjarna."""

# 1997 samræmt próf 10. bekkur í íslensku
distractors = """Undir dumbrauðum kvöldhimni drúpir eitt blóm með daggir á hálfvöxnum fræjum. Og senn kemur haustnótt á héluðum skóm og hjúpar það svalköldum blæjum. Því veðrið er annað en var hér í gær og vorið og sumarið liðið. Hinn nafnlausi brunnur mun niða þér fjær, hitt nálgast sem fyrir var kviðið. Þú hræðist ei lengur þinn hlut og þinn dóm, en hjarta þitt glúpnar og viknar: Undir dumbrauðum kvöldhimni drúpir eitt blóm, - það deyr kannski í nótt og bliknar"""

# styttra
example_text = """Er mönnum fjölgaði breyttist land hraðar vegna aukins landbúnaðar, námuvinnslu og stærri búsetusvæða."""


distractors = collapse_multispace(distractors.lower())
distractors = remove_non_alphanumeric(distractors)


# vanilla
inputs_vanilla = transform_vanilla(example_text, cfg=cfg, enc=enc)
ids_vanilla = torch.tensor(inputs_vanilla["input_ids"]).unsqueeze(0)
model_out = model(input_ids=ids_vanilla, labels=ids_vanilla)
target_ids = ids_vanilla.roll(-1)
surprisal = torch.gather(
    model_out.logits.log_softmax(-1), dim=2, index=target_ids.unsqueeze(-1)
)
surprisal = surprisal.squeeze(-1)
surprisal_vanilla = surprisal.squeeze(0)
# ic(surprisal_vanilla)


tokens = enc.convert_ids_to_tokens(ids_vanilla.squeeze().tolist())
token_scores = [round(x, 4) for x in surprisal_vanilla.exp().squeeze().tolist()]
paired_tokens = list(zip(tokens, token_scores))
ic(paired_tokens)
sys.exit(0)


def logits_per_bpe_to_score_per_byte(ids, logits, enc):
    sentinel = "▁"
    tokens = enc.convert_ids_to_tokens(ids)
    text = enc.decode(ids)
    nbytes = len(list(text.encode("utf8")))
    byte_scores = []
    byte_ids = []
    for i, (bpe_id, bpe_score) in enumerate(zip(ids, logits)):
        bpe_str = enc.convert_ids_to_tokens([bpe_id])
        bpe_str = bpe_str[0].replace(sentinel, " ")
        for byte_id in list(bpe_str.encode("utf8")):
            byte_ids.append(byte_id)
            byte_scores.append(bpe_score)
    return np.array(byte_ids), np.array(byte_scores)


try:
    from logit_colorizer.byte_colorizer import ByteLevelTextColorizer

    colorizer = ByteLevelTextColorizer.make_probability_colorizer(low_is_green=True)

    byteseq, bytescores = logits_per_bpe_to_score_per_byte(
        ids_vanilla.squeeze().tolist(), surprisal_vanilla.exp().squeeze().tolist(), enc
    )

    print(f"\n{example_text}\n\n\n", end="")
    colorizer.print_stylized(byteseq, bytescores)
    print("\n\n", end="")

    # # scramble
    # inputs_scramble = transform_example_word_noise(example_text, cfg=cfg, enc=enc, aux=distractors)
    # ids_scramble = torch.tensor(inputs_scramble["input_ids"]).unsqueeze(0)
    # input_mask_scramble = torch.tensor(inputs_soup["weights"]).unsqueeze(0).bool()
    # model_out = model(input_ids=ids_scramble, labels=ids_scramble)
    # target_ids = ids_scramble.roll(-1)
    # surprisal = torch.gather(model_out.logits.log_softmax(-1), dim=2, index=target_ids.unsqueeze(-1))
    # surprisal = surprisal.squeeze(-1)
    # surprisal_scramble = surprisal[input_mask_scramble].squeeze(0)

    # # soup
    # inputs_soup = transform_example_word_soup(example_text, cfg=cfg, enc=enc, aux=distractors)
    # ids_soup = torch.tensor(inputs_soup["input_ids"]).unsqueeze(0)
    # input_mask_soup = torch.tensor(inputs_soup["weights"]).unsqueeze(0).bool()
    # model_out = model(input_ids=ids_soup, labels=ids_soup)
    # target_ids = ids_soup.roll(-1)
    # surprisal = torch.gather(model_out.logits.log_softmax(-1), dim=2, index=target_ids.unsqueeze(-1))
    # surprisal = surprisal.squeeze(-1)
    # surprisal_soup = surprisal[input_mask_soup].squeeze(0)

except Exception as e:
    tokens = enc.convert_ids_to_tokens(ids_vanilla.squeeze().tolist())
    token_scores = [round(x, 4) for x in surprisal_vanilla.exp().squeeze().tolist()]
    paired_tokens = list(zip(tokens, token_scores))
    ic(paired_tokens)

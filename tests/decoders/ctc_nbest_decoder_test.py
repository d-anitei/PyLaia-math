from types import SimpleNamespace
from unittest.mock import patch

import torch

from laia.decoders import CTCNBestDecoder
from laia.utils import SymbolsTable


def test_ctc_nbest_decoder_uses_requested_nbest_and_tokens():
    x = torch.tensor(
        [
            [[1.0, 3.0, -1.0, 0.0]],
            [[-1.0, 2.0, -2.0, 3.0]],
        ]
    )
    syms = SymbolsTable(from_dict={0: "<ctc>", 1: "a", 2: "b", 3: "<space>"})

    def fake_beam_search_decoder(batch_features, batch_sizes):
        assert tuple(batch_features.shape) == (1, 2, 4)
        assert batch_sizes.tolist() == [2]
        return [[SimpleNamespace(tokens=torch.tensor([1, 3]), score=-0.75)]]

    with patch("laia.decoders.ctc_nbest_decoder.ctc_decoder") as ctc_decoder_mock:
        ctc_decoder_mock.return_value = fake_beam_search_decoder
        decoder = CTCNBestDecoder(4, tokens=syms)
        result = decoder(x)

    ctc_decoder_mock.assert_called_once_with(
        lexicon=None,
        tokens=["<ctc>", "a", "b", "<space>"],
        blank_token="<ctc>",
        sil_token="<eos>",
        unk_word="<unk>",
        nbest=4,
        beam_size=4,
    )
    assert result == [[([1, 3], -0.75)]]

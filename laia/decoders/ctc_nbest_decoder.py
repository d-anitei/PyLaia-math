import torch

from laia.losses.ctc_loss import transform_batch
from laia.utils import SymbolsTable

from torchaudio.models.decoder import ctc_decoder


class CTCNBestDecoder:
    """N-best decoder based on CTC output."""

    def __init__(
        self,
        nbest,
        tokens,
        blank_token="<ctc>",
        sil_token="<eos>",
        unk_word="<unk>",
    ):
        assert isinstance(nbest, int) and nbest > 0
        if isinstance(tokens, SymbolsTable):
            max_index = len(tokens) - 1
            tokens = [tokens[i] for i in range(max_index + 1)]
        assert isinstance(tokens, list) and tokens
        self._nbest = nbest
        self._tokens = tokens
        self._blank_token = blank_token
        self._sil_token = sil_token
        self._unk_word = unk_word
        self._output = None

    def __call__(self, x):
        beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=self._tokens,
            blank_token=self._blank_token,
            sil_token=self._sil_token,
            unk_word=self._unk_word,
            nbest=self._nbest,
            beam_size=self._nbest,
        )
        # Get the actual size of each feature in the batch
        batch_features, batch_sizes = transform_batch(x)
        batch_features = batch_features.detach()

        # Reshape from (n_frame, batch_size, n_tokens) to (batch_size, n_frame, n_tokens)
        batch_features = batch_features.permute((1, 0, 2))

        # Apply log softmax
        batch_features = torch.nn.functional.log_softmax(batch_features, dim=-1)

        # No GPU support for torchaudio's ctc_decoder
        device = torch.device("cpu")
        batch_features = batch_features.to(device)
        if isinstance(batch_sizes, list):
            batch_sizes = torch.tensor(batch_sizes)
            batch_sizes.to(device)

        # Decode
        hypotheses = beam_search_decoder(batch_features, batch_sizes)
        # Format the output
        output = []
        for hypothesis in hypotheses:
            out = []
            for hyp in hypothesis:
                out.append((hyp.tokens.tolist(), hyp.score))
            output.append(out)

        return output

    @property
    def output(self):
        return self._output

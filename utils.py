from pathlib import Path
from huggingface_hub import snapshot_download


def download_mistral_model(destination: Path | str | None = None) -> Path:
    """Download the Mistral-7B-Instruct-v0.3 model weights.

    Parameters
    ----------
    destination : Path | str | None, optional
        Directory to store the model. If ``None`` (default), ``~/mistral_models/7B-Instruct-v0.3``
        is used.

    Returns
    -------
    Path
        Path to the downloaded model directory.
    """
    if destination is None:
        destination = Path.home() / "mistral_models" / "7B-Instruct-v0.3"
    else:
        destination = Path(destination)

    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
        local_dir=destination,
    )

    return destination

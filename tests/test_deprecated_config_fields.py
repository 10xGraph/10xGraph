"""Config fields that nothing reads are accepted, and warn when set to a non-default value."""

import warnings

import pytest

from tenxgraph.core.realtime import VADConfig
from tenxgraph.qa.evaluation.config.eval_config import EvalConfig
from tenxgraph.qa.evaluation.config.reporter_config import UserSimulatorConfig
from tenxgraph.storage.media import DocumentHandling, ImageHandling, MultimodalConfig


@pytest.mark.parametrize(
    "build",
    [
        lambda: MultimodalConfig(image_handling=ImageHandling.URL),
        lambda: MultimodalConfig(document_handling=DocumentHandling.SKIP),
        lambda: MultimodalConfig(supported_doc_types={"application/pdf"}),
        lambda: VADConfig(start_sensitivity="HIGH"),
        lambda: VADConfig(end_sensitivity="LOW"),
        lambda: VADConfig(prefix_padding_ms=300),
        lambda: VADConfig(silence_duration_ms=800),
        lambda: EvalConfig(mock_mode=True),
        lambda: UserSimulatorConfig(thinking_enabled=True),
    ],
)
def test_unused_field_warns_when_set(build):
    with pytest.warns(DeprecationWarning, match="no effect and will be removed in 2.0"):
        build()


def test_defaults_and_used_fields_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MultimodalConfig(max_image_size_mb=5)
        VADConfig(enabled=False)
        EvalConfig(parallel=True)
        UserSimulatorConfig(temperature=0.1)

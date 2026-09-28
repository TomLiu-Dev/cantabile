"""分声部 (streams) 和找旋律 (melody)。"""
from .streams import (Stream, Separation, contigs, separate, skyline,
                      streaming_cost, evaluate, CONNECTION_FEATURES,
                      WEIGHTS_CW, WEIGHTS_IMS, WEIGHTS_GA1, WEIGHTS_DEFAULT,
                      METHODS)
from .melody import (MELODY_FEATURES, FEATURE_DOCS, DEFAULT_WEIGHTS, Feature,
                     stream_features, identify, melody_line, evaluate_against)

__all__ = [
    "Stream", "Separation", "contigs", "separate", "skyline", "streaming_cost",
    "evaluate", "CONNECTION_FEATURES", "WEIGHTS_CW", "WEIGHTS_IMS",
    "WEIGHTS_GA1", "WEIGHTS_DEFAULT", "METHODS",
    "MELODY_FEATURES", "FEATURE_DOCS", "DEFAULT_WEIGHTS", "Feature",
    "stream_features", "identify", "melody_line", "evaluate_against",
]

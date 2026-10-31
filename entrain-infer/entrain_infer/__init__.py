"""entrain-infer -- what a park can and cannot know about its own producers.

M1 of the Entrain pivot: replace a declared threshold (0.5) with a declared risk
(``alpha``), and answer quantitatively what a lie has to be to survive.

    >>> from entrain_infer import detector, bench
    >>> criteria = bench.load_criteria()
    >>> criteria["alpha"]
    0.001

Modules
-------
``model``      units, the declared constants, one observation of a park
``estimator``  turning producer series into frequency offsets (the observable)
``detector``   the GLRT, its null table, its power, its minimum detectable b
``bench``      real background, declared injections, measured noise
``report``     the artefacts: report and ROC plot

There is no hardware here, by design: a board is a fixture, a series is a record.
"""
from __future__ import annotations

from . import bench, detector, estimator, model

__all__ = ["bench", "detector", "estimator", "model"]
__version__ = "0.1.0"

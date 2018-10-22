import numpy as np
from openpnm.phases import GenericPhase
from numpy import ndarray
from typing import List


def weighted_average(target: GenericPhase, prop: str, phases: List[GenericPhase],
                     weights: str) -> ndarray:
    r"""
    Return a weighted average on a given property `prop` from a list of phases.

    Parameters
    ----------
    target : OpenPNM Object
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.

    prop : string
        The dictionary key to the array containing the pore/throat property to
        be used in the calculation.

    phases : list
        List of OpenPNM phase objects over which the given `prop` is to be
        averaged out.

    weights : string
        The dictionary key to the array containing the weights associated
        with each of the given ``phases`` for averaging.

    Returns
    -------
    weighted_average : ndarray
        Weighted average of the given `prop` averaged over `phases`.

    """
    wmean = np.zeros_like(phases[0][prop])

    for i, phase in enumerate(phases):
        wmean += phase[weights] * phase[prop]

    return wmean

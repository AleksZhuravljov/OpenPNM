from openpnm.phases import GenericPhase
import openpnm.models.phases as models


class Multiphase(GenericPhase):
    r"""
    Creates Phase object that represents a multiphase system consisting of
    a given list of OpenPNM Phase objects.

    Parameters
    ----------
    network : OpenPNM Network object
        The network to which this phase object will be attached.

    project : OpenPNM Project object, optional
        The Project with which this phase should be associted.  If a
        ``network`` is given then this is ignored and the Network's project is
        used.  If a ``network`` is not given then this is mandatory.

    name : string, optional
        The name of the phase.  This is useful to keep track of the objects
        throughout the simulation.  The name must be unique to the project.  If
        no name is given, one is generated.

    Examples
    --------
    >>> import openpnm as op
    >>> pn = op.network.Cubic(shape=[5, 5, 5])
    >>> air = op.phases.Air(network=pn)
    >>> water = op.phases.Water(network=pn)
    >>> multiphase = op.phases.Multiphase(network=pn, phases=[air, water])

    Notes
    -----
    The table below shows all of the pore-scale models that are included with
    this class to calculate the physical properties of this fluid as functions
    of the relevant state variables.

    This object is initialized at standard conditions of 298 K and 101325 Pa.
    If these conditions are changed the dependent properties can be
    recalculated by calling ``regenerate_models``.

    All of these parameters can be adjusted manually by editing the entries in
    the **ModelsDict** stored in the ``models`` attribute of the object.

    For a full listing of models and their parameters use ``print(obj.models)``
    where ``obj`` is the handle to the object.

    In addition to these models, this class also has a number of constant
    values assigned to it which can be found by running
    ``props(mode='constants')``.

    +---+----------------------+------------------+--------------------------+
    | # | Property Name        | Parameter        | Value                    |
    +===+======================+==================+==========================+
    | 1 | pore.molar_density   | model:           | ideal_gas                |
    +---+----------------------+------------------+--------------------------+
    |   |                      | regen_mode       | normal                   |
    +---+----------------------+------------------+--------------------------+
    |   |                      | pressure         | pore.pressure            |
    +---+----------------------+------------------+--------------------------+
    |   |                      | temperature      | pore.temperature         |
    +---+----------------------+------------------+--------------------------+
    | 2 | pore.diffusivity     | model:           | fuller                   |
    +---+----------------------+------------------+--------------------------+
    |   |                      | MA               | 0.032                    |
    +---+----------------------+------------------+--------------------------+
    |   |                      | MB               | 0.028                    |
    +---+----------------------+------------------+--------------------------+
    |   |                      | vA               | 16.6                     |
    +---+----------------------+------------------+--------------------------+
    |   |                      | vB               | 17.9                     |
    +---+----------------------+------------------+--------------------------+
    |   |                      | regen_mode       | normal                   |
    +---+----------------------+------------------+--------------------------+
    |   |                      | temperature      | pore.temperature         |
    +---+----------------------+------------------+--------------------------+
    |   |                      | pressure         | pore.pressure            |
    +---+----------------------+------------------+--------------------------+
    | 3 | pore.thermal_cond... | model:           | polynomial               |
    +---+----------------------+------------------+--------------------------+
    |   |                      | prop             | pore.temperature         |
    +---+----------------------+------------------+--------------------------+
    |   |                      | a                | [0.00422791, 7.89606e... |
    +---+----------------------+------------------+--------------------------+
    |   |                      | regen_mode       | normal                   |
    +---+----------------------+------------------+--------------------------+
    | 4 | pore.viscosity       | model:           | polynomial               |
    +---+----------------------+------------------+--------------------------+
    |   |                      | prop             | pore.temperature         |
    +---+----------------------+------------------+--------------------------+
    |   |                      | a                | [1.82082e-06, 6.51815... |
    +---+----------------------+------------------+--------------------------+
    |   |                      | regen_mode       | normal                   |
    +---+----------------------+------------------+--------------------------+

    References
    ----------
    The pore scale models for this class are taken from its constituent phases.

    """
    def __init__(self, phases, **kwargs):
        super().__init__(**kwargs)
        self.phases = phases

        props = ['pore.molecular_weight',
                 'pore.critical_pressure',
                 'pore.contact_angle',
                 'pore.electrical_conductivity',
                 'pore.thermal_conductivity',
                 'pore.diffusivity',
                 'pore.molar_density',
                 'pore.surface_tension',
                 'pore.viscosity']

        # By default, the mixing rule is based on volume occupancy of each
        # phase. Feel free to define your own mixing rule!
        for prop in props:
            self.add_model(propname=prop,
                           model=models.misc.weighted_average,
                           prop=prop,
                           weights='pore.occupancy',
                           phases=self.phases)

    # Override regenerate_models to update constituent phases before updating
    # the Multiphase models
    def regenerate_models(self, **kwargs):
        # Regenerate all phases within Mixture
        for phase in self.phases:
            phase.regenerate_models(*kwargs)
        # Regenerate Mixture phase
        super().regenerate_models(self, *kwargs)

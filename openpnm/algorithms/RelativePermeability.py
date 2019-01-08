from openpnm.algorithms import GenericAlgorithm, StokesFlow
from openpnm.utils import logging
from openpnm import models
import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


default_settings = {'pore_inv_seq': 'pore.invasion_sequence',
                    'throat_inv_seq': 'throat.invasion_sequence',
                    'points': 10,
                    'gh': 'throat.hydraulic_conductance',
                    'mode': 'strict',
                    }
class RelativePermeability(GenericAlgorithm):
    r"""
    Parameters
    ----------


    Notes
    -----


    """
    def __init__(self, settings={}, **kwargs):
        # Apply default settings
        self.settings.update(default_settings)
        # Apply any settings received during initialization
        self.settings.update(settings)
        super().__init__(**kwargs)
        
    def setup(self, inv_phase=None, def_phase=None,points=None,
              pore_inv_seq=None,
              throat_inv_seq=None):
        r"""
        """
        if inv_phase:
            self.settings['inv_phase'] = inv_phase.name
        if def_phase:
            self.settings['def_phase'] = def_phase.name
        if points:
            self.settings['points'] = points
        if pore_inv_seq:
            self.settings['pore_inv_seq'] = pore_inv_seq
        else:
            self.IP(self)
        if throat_inv_seq:
            self.settings['thorat_inv_seq'] = throat_inv_seq
        else:
            self.IP(self)
    def IP(self):
        network = self.project.network
        phase = self.project.phases(self.settings['inv_phase'])
        inv=op.algorithms.InvasionPercolation(phase=phase,network=network,project=self.project)
        inv.setup(phase=phase,
                  entry_pressure='throat.entry_pressure',pore_volume='pore.volume', throat_volume='throat.volume')
        inlets = network.pores(['top'])
        used_inlets = [inlets[x] for x in range(0, len(inlets), 2)]
        inv.set_inlets(pores=used_inlets)
        inv.run()
        Snwparr =  []
        Pcarr =  []
        Sarr=np.linspace(0,1,num=60)
        for Snw in Sarr:
            res1=inv.results(Snwp=Snw)
            occ_ts=res1['throat.occupancy']
            if np.any(occ_ts):
                max_pthroat=np.max(phase['throat.entry_pressure'][occ_ts])
                Pcarr.append(max_pthroat)
                Snwparr.append(Snw)
        self.settings['pore_inv_seq'] = inv['pore.invasion_sequence']
        self.settings['thorat_inv_seq'] = inv['throat.invasion_sequence']
        plt.figure(1)
        y=np.array(Pcarr[:])
        x=1.0-np.array(Snwparr[:])
        plt.xticks(np.arange(x.min(), x.max(), 0.05))
        plt.yticks(np.arange(y.min(), y.max(),0.1))
        plt.plot(x, y)
        plt.xlabel('Invading Phase Saturation')
        plt.ylabel('Capillary Pressure')
        plt.grid(True)
        sat=np.array(Snwparr[:])

    def set_inlets(self, pores):
        r"""
        """
        self['pore.inlets'] = False
        self['pore.inlets'][pores] = True

    def set_outlets(self, pores):
        r"""
        """
        self['pore.outlets'] = False
        self['pore.outlets'][pores] = True
        
    def update_phase_and_phys(self,results):
        self.settings['inv_phase']['pore.occupancy'] = results['pore.occupancy']
        self.settings['def_phase']['pore.occupancy'] = 1-results['pore.occupancy']
        self.settings['inv_phase']['throat.occupancy'] = results['throat.occupancy']
        self.settings['def_phase']['throat.occupancy'] = 1-results['throat.occupancy']
        #adding multiphase conductances
        mode=self.settings['mode']
        self.settings['def_phase'].add_model(model=op.models.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance',
                       mode=mode)
        self.settings['inv_phase'].add_model(model=op.models.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_hydraulic_conductance',
                         throat_conductance='throat.hydraulic_conductance',
                         mode=mode)
    def top_b(lx,ly,lz):
        da = lx*ly
        dl = lz
        res_2=[da,dl]
        return res_2

    def run(self, inlets=None, outlets=None):
        r"""
        """
        # Retrieve phase and network
        network = self.project.network
        
        bounds = [ ['top', 'bottom']]
        if inlets is not None:
            self.set_inlets(pores=inlets)
        else:
            self.set_inlets(pores=network.pores(labels=bounds[0][0]))
        if outlets is not None:
            self.set_outlets(pores=outlets)
        else:
            self.set_outlets(pores=network.pores(labels=bounds[0][1]))
        #first calc single phase absolute permeability (assumming in 1 direction only)
        [amax, bmax, cmax] = np.max(network['pore.coords'], axis=0)
        [amin, bmin, cmin] = np.min(network['pore.coords'], axis=0)
        lx = amax-amin
        ly = bmax-bmin
        lz = cmax-cmin
        options = {0 : self.top_b(lx,ly,lz)}
        K_def=1
        K_inv=1
        ##apply single phase flow
        for bound_increment in range(len(bounds)):
        # Run Single phase algs effective properties
        #bound_increment=0
            [da,dl]=options[bound_increment]
            #Kw
            St_def = op.algorithms.StokesFlow(network=network, phase=self.project.phases(self.settings['def_phase']))
            St_def.setup(conductance='throat.hydraulic_conductance')
            St_def._set_BC(pores=self['pore.inlets'], bctype='value', bcvalues=1)
            St_def._set_BC(pores=self['pore.outlets'], bctype='value', bcvalues=0)
            St_def.run()
            K_def[bound_increment] = St_def.calc_effective_permeability(domain_area=da, domain_length=dl,inlets=self['pore.inlets'], outlets=self['pore.outlets'])
            #proj.purge_object(obj=St_def)
            #Ko
            St_inv = op.algorithms.StokesFlow(network=network, phase=self.project.phases(self.settings['inv_phase']))
            St_inv.setup(conductance='throat.hydraulic_conductance')
            St_inv._set_BC(pores=self['pore.inlets'], bctype='value', bcvalues=1)
            St_inv._set_BC(pores=self['pore.outlets'], bctype='value', bcvalues=0)
            St_inv.run()
            K_inv[bound_increment] = St_inv.calc_effective_permeability(domain_area=da, domain_length=dl,inlets=self['pore.inlets'], outlets=self['pore.outlets'])
            #proj.purge_object(obj=St_inv)
        #apply two phase effective perm calculation        
        for Sp in sat:
            self.update_phase_and_phys(self.IP.results(Snwp=Sp))
            print('sat is equal to', Sp)
            for bound_increment in range(len(bounds)):
                 #water
                 St_def_tp = op.algorithms.StokesFlow(network=network, phase=self.project.phases(self.settings['def_phase']))
                 St_def_tp.setup(conductance='throat.conduit_hydraulic_conductance')
                 St_def_tp.set_value_BC(pores=self['pore.inlets'], bctype='value', bcvalues=1)
                 St_def_tp.set_value_BC(pores=self['pore.outlets'], bctype='value', bcvalues=0)
                 #oil
                 St_inv_tp = op.algorithms.StokesFlow(network=network, phase=self.project.phases(self.settings['inv_phase']))
                 St_inv_tp.setup(conductance='throat.conduit_hydraulic_conductance')
                 St_inv_tp.set_value_BC(pores=self['pore.inlets'], bctype='value', bcvalues=1)
                 St_inv_tp.set_value_BC(pores=self['pore.outlets'], bctype='value', bcvalues=0)
                 # Run Multiphase algs
                 St_def_tp.run()
                 St_inv_tp.run()
                 K_def_tp = St_def_tp.calc_effective_permeability(domain_area=da, domain_length=dl)
                 K_inv_tp = St_inv_tp.calc_effective_permeability(domain_area=da, domain_length=dl)
                 krel_def =K_def_tp/K_def[bound_increment]
                 krel_inv= K_inv_tp /K_inv[bound_increment]
                 K_rel_def[str(bound_increment)].append(krel_def)
                 K_rel_inv[str(bound_increment)].append(krel_inv)
                 proj.purge_object(obj=St_def_tp)
                 proj.purge_object(obj=St_inv_tp)
#        sf = StokesFlow(network=network)
#        sf.setup(phase=phase,
#                 quantity='pore.pressure',
#                 conductance=self.settings['gh'])
#        sf.set_value_BC(pores=self['pore.inlets'], values=1)
#        sf.set_value_BC(pores=self['pore.outlets'], values=0)
#        phase['pore.occupancy'] = 0
#        phase['throat.occupancy'] = 0
#        phase.add_model(propname='throat.multiphase_hydraulic_conductance',
#                        model=models.physics.multiphase.conduit_conductance,
#                        throat_conductance=self.settings['gh'],
#                        throat_occupancy='throat.occupancy',
#                        pore_occupancy='pore.occupancy',
#                        mode=self.settings['mode'],
#                        )
#        max_inv = np.amax(phase['pore.invasion_sequence'])
#        invs = np.linspace(0, max_inv, self.settings['points'])
#        results = []
#        for s in invs:
#            phase['pore.occupancy'] = 1.0*(phase[self.settings['pore_inv_seq']] <= s)
#            phase['throat.occupancy'] = 1.0*(phase[self.settings['throat_inv_seq']] <= s)
#            phase.regenerate_models(deep=True)
#            sf.run()
#            results.append([[s, sf.rate(pores=self['pore.inlets'])]])
        return results

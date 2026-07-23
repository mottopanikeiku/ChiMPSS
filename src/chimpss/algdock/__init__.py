"""
chimpss.algdock
===============
Standalone BPMF (Binding Potential of Mean Force) engine — the AlGDock codebase
vendored directly into ChiMPSS.  No external AlGDock package is required.

Public API
----------
BindingPMF : the main BPMF calculation class
SystemConverter : converts OpenMM XML + PDB → AMBER prmtop/inpcrd inputs
"""

from chimpss.algdock.BindingPMF import BPMF as BindingPMF
from chimpss.algdock.converter import SystemConverter

__all__ = ['BindingPMF', 'SystemConverter']

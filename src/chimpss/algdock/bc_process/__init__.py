"""BC thermodynamic path module for the vendored BPMF engine

Handles the BC replica exchange ladder: warming the unbound ligand from target
temperature to high temperature (e.g., 300K to 600K).

The BC ladder uses alpha as a progress variable (1.0 to 0.0) that controls:
  - Temperature: T=T_TARGET to T=T_HIGH (e.g., 300K to 600K)
  - Solvation (depends on mode):
    - Desolvated/Reduced/Fractional: OBC scales from 1.0 to 0.0 (desolvates as it warms)
    - Full: OBC=1.0 (no desolvation)

State labels:
  - alpha=1.0: T=T_TARGET, OBC=1.0 or mode-dependent (e.g., 300K, state B)
  - alpha=0.0: T=T_HIGH, OBC=0.0 or 1.0 (e.g., 600K, state C)

Purpose: Enhanced sampling at elevated temperature to overcome energy barriers.
"""

from .bc_calculator import BCCalculator

__all__ = ['BCCalculator']

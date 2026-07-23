"""CD thermodynamic path module for the vendored BPMF engine

Handles the CD replica exchange ladder: cooling while turning on receptor-ligand
interactions (e.g., from 600K with grids OFF to 300K with grids ON).

The CD ladder uses alpha as a progress variable (0.0 to 1.0) that controls:
  - Temperature: T=T_HIGH to T=T_TARGET (e.g., 600K to 300K)
  - Grid coupling: OFF to ON via alpha_g sigmoid
  - Solvation (mode-dependent):
    - Desolvated/Reduced: OBC=0 (no solvation)
    - Fractional: OBC=alpha_g (scales with grids)
    - Full: OBC=1.0 (full solvation)

State labels:
  - alpha=0.0: T=T_HIGH, grids OFF, OBC mode-dependent (e.g., 600K, state C)
  - alpha=1.0: T=T_TARGET, grids ON, OBC mode-dependent (e.g., 300K, state D)

Grid scaling: alpha_g = 4*(alpha-0.5)^2 / (1 + exp(-100*(alpha-0.5)))
Purpose: Gradually introduce receptor interactions while cooling to drive binding.
"""

from .state_manager import CDStateManager
from .cd_calculator import CDCalculator

__all__ = ['CDStateManager', 'CDCalculator']

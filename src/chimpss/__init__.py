__version__ = "0.1.2"

# Stage classes will be re-exported here as they migrate into src/chimpss/:
#   Phase 2: from chimpss.bridgeport import Bridgeport
#   Phase 3: from chimpss.motorrow import MotorRow
#   Phase 4: from chimpss.fultonmarket import FultonMarket

__all__ = ["BindingPMF"]


def __getattr__(name):
    """Load optional high-level APIs only when they are requested."""
    if name == "BindingPMF":
        from chimpss.algdock import BindingPMF

        return BindingPMF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

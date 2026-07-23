"""Verify that the vendored BPMF engine is accessible through chimpss.algdock."""

import pytest


def test_import_chimpss_algdock():
    import chimpss.algdock  # noqa: F401


def test_bindingpmf_is_vendored_bpmf():
    from chimpss.algdock import BindingPMF
    from chimpss.algdock.BindingPMF import BPMF
    assert BindingPMF is BPMF


def test_chimpss_top_level_export():
    from chimpss import BindingPMF  # noqa: F401


def test_algdock_submodules_importable():
    """Core vendored BPMF submodules must import without error."""
    import chimpss.algdock.IO  # noqa: F401
    import chimpss.algdock.system  # noqa: F401
    import chimpss.algdock.replica_exchange  # noqa: F401
    import chimpss.algdock.free_energy  # noqa: F401
    import chimpss.algdock.bc_process  # noqa: F401
    import chimpss.algdock.cd_process  # noqa: F401

"""
Targeted diagnostic logger for comparing MMTK and OpenMM implementations

Only logs at critical checkpoints to keep output manageable.

Usage:
    from chimpss.algdock.diagnostic_logger import DiagnosticLogger

    # Initialize with implementation name
    diag = DiagnosticLogger('diagnostics', implementation='mmtk')

    # Log at critical checkpoints only
    diag.log_bc_state(state_idx, params, E, confs_sample)
"""

import json
import numpy as np
import os
import hashlib

class DiagnosticLogger(object):
    """Logs values at critical checkpoints for comparison"""

    def __init__(self, base_dir, implementation='mmtk'):
        """
        Parameters
        ----------
        base_dir : str
            Directory for storing diagnostic files
        implementation : str
            'mmtk' or 'openmm'
        """
        self.base_dir = base_dir
        self.implementation = implementation

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.filename = os.path.join(base_dir, '%s_diagnostics.json' % implementation)

        # Load existing data if file exists, otherwise start fresh
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
            self.checkpoint_count = len(self.data['checkpoints'])
        else:
            self.data = {
                'implementation': implementation,
                'checkpoints': []
            }
            self.checkpoint_count = 0

    def _serialize_value(self, val):
        """Convert value to JSON-serializable format with summary stats"""
        if isinstance(val, np.ndarray):
            # Python 2 / older numpy compatibility
            try:
                data_bytes = val.tobytes()
            except AttributeError:
                data_bytes = val.tostring()

            return {
                'type': 'array',
                'shape': list(val.shape),
                'mean': float(np.mean(val)),
                'std': float(np.std(val)),
                'min': float(np.min(val)),
                'max': float(np.max(val)),
                'first_3': val.flatten()[:3].tolist() if val.size >= 3 else val.flatten().tolist(),
                'hash': hashlib.md5(data_bytes).hexdigest()  # For exact comparison
            }
        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (int, float)):
            arr = np.array(val)
            return self._serialize_value(arr)
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        else:
            return val

    def log_bc_state(self, state_idx, params, E=None, conf_sample=None):
        """
        Log BC protocol state

        Parameters
        ----------
        state_idx : int
            Index in protocol
        params : dict
            Parameter dictionary
        E : dict, optional
            Energy terms dictionary
        conf_sample : array, optional
            First 5 atoms of a sample configuration
        """
        checkpoint = {
            'type': 'bc_state',
            'index': state_idx,
            'alpha': params.get('alpha'),
            'T': params.get('T'),
            'OBC': params.get('OBC'),
            'crossed': params.get('crossed', False)
        }

        if E is not None:
            checkpoint['energies'] = {
                key: self._serialize_value(val) for key, val in E.items()
            }

        if conf_sample is not None:
            checkpoint['coords_sample'] = self._serialize_value(conf_sample)

        self._add_checkpoint(checkpoint)

    def log_cd_state(self, state_idx, params, E=None, conf_sample=None):
        """
        Log CD protocol state

        Parameters
        ----------
        state_idx : int
            Index in protocol
        params : dict
            Parameter dictionary
        E : dict, optional
            Energy terms dictionary
        conf_sample : array, optional
            First 5 atoms of a sample configuration
        """
        checkpoint = {
            'type': 'cd_state',
            'index': state_idx,
            'alpha': params.get('alpha'),
            'T': params.get('T'),
            'OBC': params.get('OBC'),
            'sLJr': params.get('sLJr'),
            'LJr': params.get('LJr'),
            'LJa': params.get('LJa'),
            'ELE': params.get('ELE'),
            'crossed': params.get('crossed', False)
        }

        if E is not None:
            checkpoint['energies'] = {
                key: self._serialize_value(val) for key, val in E.items()
            }

        if conf_sample is not None:
            checkpoint['coords_sample'] = self._serialize_value(conf_sample)

        self._add_checkpoint(checkpoint)

    def log_repx_cycle(self, process, cycle, state_inds, mean_energies_per_state,
                       context=None, protocol=None, energy_threshold=10000.0,
                       confs=None, inv_state_inds_now=None, system=None,
                       replicas_that_swapped=None):
        """
        Log replica exchange cycle summary with enhanced diagnostics

        If any state has extreme energy (> threshold), logs detailed force breakdown
        and positions for debugging.

        Parameters
        ----------
        process : str
            'BC' or 'CD'
        cycle : int
            Cycle number
        state_inds : list
            State assignments for each replica
        mean_energies_per_state : array
            Mean energy for each state
        context : OpenMM Context, optional
            OpenMM context for detailed force analysis
        protocol : list, optional
            Protocol information for each state
        energy_threshold : float, optional
            Energy threshold (kJ/mol) above which to log detailed diagnostics
            Default: 10000.0 kJ/mol
        confs : list of arrays, optional
            Configurations for each replica (needed to load correct replica)
        inv_state_inds_now : array, optional
            Mapping from state index to replica index
        system : chimpss.algdock.system.System, optional
            System object to set positions and parameters
        replicas_that_swapped : list, optional
            List of replica indices that had accepted swaps this sweep
        """
        checkpoint = {
            'type': '%s_repx_cycle' % process.lower(),
            'cycle': cycle,
            'state_inds': list(state_inds),
            'mean_energies': self._serialize_value(mean_energies_per_state)
        }

        # Track which replicas had accepted swaps
        if replicas_that_swapped is not None:
            checkpoint['replicas_that_swapped'] = list(replicas_that_swapped)

        # Check for extreme energies and log detailed diagnostics
        if context is not None:
            max_energy = float(np.max(mean_energies_per_state))
            if max_energy > energy_threshold:
                checkpoint['extreme_energy_detected'] = True
                checkpoint['max_energy'] = max_energy

                # Find which state(s) have extreme energies
                extreme_states = np.where(mean_energies_per_state > energy_threshold)[0]
                checkpoint['extreme_states'] = extreme_states.tolist()

                # Log detailed force breakdown for first extreme state
                if len(extreme_states) > 0:
                    try:
                        state_idx = extreme_states[0]

                        # If we have access to replicas, load the CORRECT configuration
                        if (confs is not None and inv_state_inds_now is not None and
                            system is not None and protocol is not None):

                            # Find which replica has this state
                            replica_idx = inv_state_inds_now[state_idx]
                            checkpoint['extreme_replica_idx'] = int(replica_idx)

                            # Check if this replica just had a swap accepted
                            if replicas_that_swapped is not None:
                                swap_just_accepted = replica_idx in replicas_that_swapped
                                checkpoint['swap_just_accepted'] = swap_just_accepted
                                if swap_just_accepted:
                                    checkpoint['analysis'] = 'EXTREME ENERGY IN NEWLY ACCEPTED SWAP - bad swap was accepted!'
                                    print("\n*** WARNING: EXTREME ENERGY DETECTED ***")
                                    print("    Cycle %d, State %d (Replica %d)" % (cycle, state_idx, replica_idx))
                                    print("    Energy: %.2f kJ/mol" % max_energy)
                                    print("    STATUS: SWAP WAS JUST ACCEPTED - Bad configuration accepted into ensemble!")
                                    print("    This suggests acceptance criterion or energy evaluation has a problem.\n")
                                else:
                                    checkpoint['analysis'] = 'EXTREME ENERGY IN PRE-EXISTING STATE - swap was rejected or no swap attempted'
                                    print("\n*** WARNING: EXTREME ENERGY DETECTED ***")
                                    print("    Cycle %d, State %d (Replica %d)" % (cycle, state_idx, replica_idx))
                                    print("    Energy: %.2f kJ/mol" % max_energy)
                                    print("    STATUS: PRE-EXISTING STATE - No recent swap accepted")
                                    print("    This suggests the bad configuration persisted from earlier or was generated during sampling.\n")

                            # Load the configuration from this replica
                            extreme_conf = confs[replica_idx]

                            # Set the system to the extreme state parameters
                            system.setParams(protocol[state_idx])

                            # Set positions in context
                            try:
                                import openmm.unit as unit
                            except ImportError:
                                import openmm.unit as unit

                            # Convert numpy array to positions with units
                            positions_with_units = extreme_conf * unit.nanometers
                            context.setPositions(positions_with_units)

                            # Now get force breakdown with the CORRECT configuration and state
                            force_breakdown = self._get_force_breakdown(context)
                            checkpoint['force_breakdown'] = force_breakdown

                            # Get positions from the extreme replica
                            positions = extreme_conf  # Already in nm as numpy array
                            checkpoint['positions_com'] = list(np.mean(positions, axis=0))
                            checkpoint['positions_min'] = list(np.min(positions, axis=0))
                            checkpoint['positions_max'] = list(np.max(positions, axis=0))
                            checkpoint['positions_sample'] = self._serialize_value(positions[:5])

                        else:
                            # Fallback to old behavior (will be wrong replica but better than nothing)
                            checkpoint['warning'] = 'Using context state (may be wrong replica - need confs/system)'

                            force_breakdown = self._get_force_breakdown(context)
                            checkpoint['force_breakdown'] = force_breakdown

                            try:
                                import openmm.unit as unit
                            except ImportError:
                                import openmm.unit as unit

                            positions_quantity = context.getState(getPositions=True).getPositions(asNumpy=True)
                            positions = positions_quantity.value_in_unit(unit.nanometers)

                            checkpoint['positions_com'] = list(np.mean(positions, axis=0))
                            checkpoint['positions_min'] = list(np.min(positions, axis=0))
                            checkpoint['positions_max'] = list(np.max(positions, axis=0))
                            checkpoint['positions_sample'] = self._serialize_value(positions[:5])

                        # Log protocol info if available
                        if protocol is not None and state_idx < len(protocol):
                            checkpoint['state_params'] = {
                                'alpha': protocol[state_idx].get('alpha'),
                                'T': protocol[state_idx].get('T'),
                                'MM': protocol[state_idx].get('MM'),
                                'site': protocol[state_idx].get('site'),
                                'LJr': protocol[state_idx].get('LJr'),
                                'LJa': protocol[state_idx].get('LJa'),
                                'ELE': protocol[state_idx].get('ELE'),
                                'sLJr': protocol[state_idx].get('sLJr'),
                                'sELE': protocol[state_idx].get('sELE')
                            }
                    except Exception as e:
                        import traceback
                        checkpoint['diagnostic_error'] = str(e)
                        checkpoint['diagnostic_traceback'] = traceback.format_exc()

        self._add_checkpoint(checkpoint)

    def _get_force_breakdown(self, context):
        """
        Get energy contribution from each force group

        Parameters
        ----------
        context : OpenMM Context
            OpenMM context

        Returns
        -------
        dict
            Force name -> energy (kJ/mol) mapping
        """
        import openmm.unit as unit

        system = context.getSystem()
        breakdown = {}

        # Get total energy first
        state_total = context.getState(getEnergy=True)
        breakdown['total'] = state_total.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Get energy for each force group
        for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)
            force_group = force.getForceGroup()

            # Get force name - try getName() first, fall back to class name
            if hasattr(force, 'getName'):
                force_name = force.getName()
            else:
                force_name = force.__class__.__name__

            # Get energy for this force group
            state_force = context.getState(getEnergy=True, groups={force_group})
            energy_kj = state_force.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            breakdown[force_name] = {
                'energy_kj': float(energy_kj),
                'force_group': force_group,
                'force_type': force.__class__.__name__
            }

            # For GridForce objects, also extract scaling factors
            # GridForce appears as generic "Force" class but has custom methods
            if hasattr(force, 'getGridParameters'):
                try:
                    import gridforceplugin as gfp
                    # Get grid parameters including scaling factors
                    counts, spacing, vals, scaling_factors = force.getGridParameters()
                    sf_array = np.array(scaling_factors)
                    breakdown[force_name]['scaling_factors_stats'] = {
                        'min': float(np.min(sf_array)),
                        'max': float(np.max(sf_array)),
                        'mean': float(np.mean(sf_array)),
                        'num_zero': int(np.sum(sf_array == 0)),
                        'num_nonzero': int(np.sum(sf_array != 0)),
                        'total_atoms': len(scaling_factors)
                    }
                except Exception as e:
                    breakdown[force_name]['scaling_factor_error'] = str(e)

        return breakdown

    def log_energy_terms(self, label, E, conf_sample=None):
        """
        Log detailed energy breakdown at a specific point

        Parameters
        ----------
        label : str
            Description of this checkpoint (e.g., 'bc_init', 'cd_first_eval')
        E : dict
            Energy terms dictionary
        conf_sample : array, optional
            First 5 atoms of configuration
        """
        checkpoint = {
            'type': 'energy_breakdown',
            'label': label,
            'energies': {key: self._serialize_value(val) for key, val in E.items()}
        }

        if conf_sample is not None:
            checkpoint['coords_sample'] = self._serialize_value(conf_sample)

        self._add_checkpoint(checkpoint)

    def _add_checkpoint(self, checkpoint):
        """Add checkpoint and save"""
        checkpoint['checkpoint_num'] = self.checkpoint_count
        self.checkpoint_count += 1
        self.data['checkpoints'].append(checkpoint)

        # Save after each checkpoint
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def summarize(self, verbose=False):
        """Print summary of logged checkpoints

        Parameters
        ----------
        verbose : bool
            If True, print summary to stdout. Default False to reduce test output noise.
        """
        if not verbose:
            return

        print("\n=== Diagnostic Summary for %s ===" % self.implementation)
        print("Total checkpoints: %d" % len(self.data['checkpoints']))

        types = {}
        for cp in self.data['checkpoints']:
            t = cp['type']
            types[t] = types.get(t, 0) + 1

        for t, count in sorted(types.items()):
            print("  %s: %d" % (t, count))

        print("Saved to: %s\n" % self.filename)


def compare_diagnostics(mmtk_file, openmm_file, output_file=None):
    """
    Compare MMTK and OpenMM diagnostic files

    Parameters
    ----------
    mmtk_file : str
        Path to MMTK diagnostics JSON
    openmm_file : str
        Path to OpenMM diagnostics JSON
    output_file : str, optional
        Path for output report (default: comparison_report.txt)

    Returns
    -------
    str
        Comparison report
    """
    if output_file is None:
        base_dir = os.path.dirname(mmtk_file)
        output_file = os.path.join(base_dir, 'comparison_report.txt')

    with open(mmtk_file, 'r') as f:
        mmtk_data = json.load(f)

    with open(openmm_file, 'r') as f:
        openmm_data = json.load(f)

    mmtk_cps = mmtk_data['checkpoints']
    openmm_cps = openmm_data['checkpoints']

    report = []
    report.append("=" * 80)
    report.append("MMTK vs OpenMM Diagnostic Comparison")
    report.append("=" * 80)
    report.append("MMTK checkpoints: %d" % len(mmtk_cps))
    report.append("OpenMM checkpoints: %d" % len(openmm_cps))
    report.append("")

    # Compare checkpoint by checkpoint
    max_len = max(len(mmtk_cps), len(openmm_cps))

    first_divergence = None

    for i in range(max_len):
        if i >= len(mmtk_cps):
            report.append("Checkpoint %d: MISSING in MMTK" % i)
            continue
        if i >= len(openmm_cps):
            report.append("Checkpoint %d: MISSING in OpenMM" % i)
            continue

        mmtk_cp = mmtk_cps[i]
        openmm_cp = openmm_cps[i]

        report.append("-" * 80)
        report.append("Checkpoint %d: %s" % (i, mmtk_cp['type']))
        report.append("-" * 80)

        # Compare type
        if mmtk_cp['type'] != openmm_cp['type']:
            report.append("  TYPE MISMATCH: %s vs %s" % (mmtk_cp['type'], openmm_cp['type']))
            if first_divergence is None:
                first_divergence = i
            continue

        # Compare scalar parameters
        has_diff = False
        for key in ['alpha', 'T', 'OBC', 'sLJr', 'LJr', 'LJa', 'ELE', 'index', 'cycle']:
            if key in mmtk_cp and key in openmm_cp:
                m_val = mmtk_cp[key]
                o_val = openmm_cp[key]

                if isinstance(m_val, (int, float)) and isinstance(o_val, (int, float)):
                    diff = abs(m_val - o_val)
                    if diff > 1e-10:
                        report.append("  %s: MMTK=%.10e, OpenMM=%.10e, diff=%.3e" %
                                    (key, m_val, o_val, diff))
                        has_diff = True
                elif m_val != o_val:
                    report.append("  %s: MMTK=%s, OpenMM=%s" % (key, m_val, o_val))
                    has_diff = True

        # Compare energies
        if 'energies' in mmtk_cp and 'energies' in openmm_cp:
            for ekey in mmtk_cp['energies'].keys():
                if ekey not in openmm_cp['energies']:
                    report.append("  Energy %s: MISSING in OpenMM" % ekey)
                    has_diff = True
                    continue

                m_e = mmtk_cp['energies'][ekey]
                o_e = openmm_cp['energies'][ekey]

                if isinstance(m_e, dict) and 'mean' in m_e:
                    diff_mean = abs(m_e['mean'] - o_e['mean'])
                    diff_std = abs(m_e['std'] - o_e['std'])

                    if diff_mean > 1e-6 or diff_std > 1e-6:
                        report.append("  Energy %s:" % ekey)
                        report.append("    MMTK:   mean=%.6e, std=%.6e" % (m_e['mean'], m_e['std']))
                        report.append("    OpenMM: mean=%.6e, std=%.6e" % (o_e['mean'], o_e['std']))
                        report.append("    Diff:   mean=%.3e, std=%.3e" % (diff_mean, diff_std))
                        has_diff = True

                        # Check hash for exact comparison
                        if m_e.get('hash') != o_e.get('hash'):
                            report.append("    *** Arrays are DIFFERENT (hash mismatch) ***")

        # Compare coordinates
        if 'coords_sample' in mmtk_cp and 'coords_sample' in openmm_cp:
            m_c = mmtk_cp['coords_sample']
            o_c = openmm_cp['coords_sample']

            if m_c.get('hash') != o_c.get('hash'):
                diff_mean = abs(m_c['mean'] - o_c['mean'])
                report.append("  Coordinates:")
                report.append("    MMTK:   mean=%.6e" % m_c['mean'])
                report.append("    OpenMM: mean=%.6e" % o_c['mean'])
                report.append("    Diff:   %.3e" % diff_mean)
                has_diff = True

        if has_diff and first_divergence is None:
            first_divergence = i

        if not has_diff:
            report.append("  OK: All values match")

        report.append("")

    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    if first_divergence is None:
        report.append("OK: All checkpoints match!")
    else:
        report.append("DIVERGENCE: First divergence at checkpoint %d" % first_divergence)
    report.append("")

    # Write to file
    report_text = '\n'.join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print("Comparison report written to:", output_file)
    return report_text

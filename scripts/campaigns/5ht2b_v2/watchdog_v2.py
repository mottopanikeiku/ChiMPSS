"""Keep every v2 5-HT2B system moving until 60/60, without ever double-booking one.

Each production window resubmits itself 15 min before its wall clock, but that
sbatch can be refused (GPU-reservation pressure has refused submissions at
8-17 queued jobs before), and a refused system just stops. v1 lost 8 and then
41 days that way. This watchdog runs every few hours and, per system:

  DONE        60/60 saved           -> make sure a convergence report exists
  ACTIVE      a job for it is queued or running             -> nothing
  RESUBMIT    no job, < 60, equilibrated inputs present     -> submit_fm_v2.sh
  FLAG        blocked dependency / missing inputs / last window gave up on
              setup errors / no progress after 3 resubmits  -> leave it, alert

Safety: a system must never have two FultonMarket jobs (it corrupts
output.ncdf). If ANY queued job's system cannot be identified, the watchdog
takes no action at all this round. Exit 1 when anything is flagged, so the
SLURM FAIL email reaches a human.

Usage: python watchdog_v2.py [--dry-run] [--selftest]
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

V2 = '/expanse/lustre/projects/iit127/fcetin/5ht2b/v2'
SYSTEMS = ['LSD', 'lisuride', 'methylergonovine', 'methysergide', 'LY266097',
           'lisuride_L362F_mutseq', 'methylergonovine_T140A_mutseq',
           'methysergide_A225G_mutseq', 'LSD_L362F_mutseq']
TARGET = 60
MAX_NO_PROGRESS = 3
STATE = f'{V2}/logs/watchdog_state.json'
OUT_RE = re.compile(r'/v2/fultonmarket/([A-Za-z0-9_]+)(?=\s|$)')


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def queued_jobs(user='fcetin'):
    """[(jobid, name, state, reason)] for this user's queued/running jobs."""
    rows = [l.split('|') for l in sh(['squeue', '-u', user, '-h', '-o', '%i|%j|%T|%r']).splitlines() if l]
    return [tuple(r) for r in rows if len(r) == 4]


def fm_job_systems(jobids):
    """Map FM_V2 job ids -> system via the full sbatch SubmitLine."""
    if not jobids:
        return {}
    out = sh(['sacct', '-j', ','.join(jobids), '-X', '-n', '-P', '--format=JobID,SubmitLine'])
    found = {}
    for line in out.splitlines():
        jid, _, submit = line.partition('|')
        m = OUT_RE.search(submit)
        if jid in jobids and m:
            found[jid] = m.group(1)
    return found


def last_fm_log(system):
    """Newest FM_V2 log whose header names this system's output dir."""
    logs = sorted(glob.glob(f'{V2}/logs/fultonmarket/FM_V2.*.out'), key=os.path.getmtime, reverse=True)
    for p in logs:
        with open(p, errors='replace') as f:
            head = f.readline()
        if re.search(rf'/v2/fultonmarket/{re.escape(system)}(?=\s|$)', head):
            return p
    return None


def decide(system, n_saved, jobs_for_system, has_inputs, last_log_text, state, conv_done):
    """Pure decision function (see --selftest). Returns (action, reason)."""
    if n_saved >= TARGET:
        conv_job = any(name.startswith('CONV_') for _, name, _, _ in jobs_for_system)
        return ('DONE', 'convergence report present') if conv_done or conv_job else ('CONVERGE', 'no report yet')
    live = [j for j in jobs_for_system if j[3] != 'DependencyNeverSatisfied']
    if live:
        return 'ACTIVE', ', '.join(f'{n}:{s}' for _, n, s, _ in live)
    if jobs_for_system:
        return 'FLAG', 'only blocked jobs (DependencyNeverSatisfied): ' + ', '.join(n for _, n, _, _ in jobs_for_system)
    if not has_inputs:
        return 'FLAG', 'no equilibrated inputs (equil_hmr/<name>/step_5.*) and no job: equilibration needs attention'
    if last_log_text and 'giving up' in last_log_text and 'fast failure' in last_log_text:
        return 'FLAG', 'last window gave up on repeated fast (setup) failures; read its log'
    st = state.get(system, {})
    if st.get('n') == n_saved and st.get('count', 0) >= MAX_NO_PROGRESS:
        return 'FLAG', f'no new sub-sims after {st["count"]} watchdog resubmits at {n_saved}/{TARGET}'
    return 'RESUBMIT', f'{n_saved}/{TARGET}, no job queued'


def selftest():
    j = lambda name, reason='Priority', state='PENDING': ('1', name, state, reason)
    S = {}
    cases = [
        (('x', 60, [], True, '', S, True), 'DONE'),
        (('x', 60, [], True, '', S, False), 'CONVERGE'),
        (('x', 60, [j('CONV_x')], True, '', S, False), 'DONE'),
        (('x', 12, [j('FM_V2')], True, '', S, False), 'ACTIVE'),
        (('x', 0, [j('LAUNCH_x', 'DependencyNeverSatisfied')], True, '', S, False), 'FLAG'),
        (('x', 0, [], False, '', S, False), 'FLAG'),
        (('x', 5, [], True, '=== giving up: 3 fast failures ===', S, False), 'FLAG'),
        (('x', 5, [], True, '', {'x': {'n': 5, 'count': 3}}, False), 'FLAG'),
        (('x', 5, [], True, '', {'x': {'n': 4, 'count': 3}}, False), 'RESUBMIT'),
        (('x', 5, [], True, '=== exit 1 after 9000s (expected NaN) -- resuming ===', S, False), 'RESUBMIT'),
    ]
    bad = [(args[:2], want, decide(*args)[0]) for args, want in cases if decide(*args)[0] != want]
    m = OUT_RE.search('sbatch x.job /a/v2/equil_hmr/LSD/step_5.pdb /b /p/v2/fultonmarket/LSD_L362F_mutseq --x 1')
    if not m or m.group(1) != 'LSD_L362F_mutseq':
        bad.append(('OUT_RE', 'LSD_L362F_mutseq', m and m.group(1)))
    print('selftest:', 'PASS' if not bad else f'FAIL {bad}')
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    jobs = queued_jobs()
    fm_ids = [jid for jid, name, _, _ in jobs if name == 'FM_V2']
    fm_map = fm_job_systems(fm_ids)
    unmapped = [jid for jid in fm_ids if jid not in fm_map]
    if unmapped:
        print(f'[{now}] SAFETY STOP: cannot identify the system of FM_V2 job(s) {unmapped}; taking no action.')
        print('ALL_DONE=0')
        sys.exit(1)

    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    flags, actions, all_done = [], [], True
    for s in SYSTEMS:
        out = f'{V2}/fultonmarket/{s}'
        n = len([d for d in os.listdir(f'{out}/saved_variables') if d.isdigit()]) if os.path.isdir(f'{out}/saved_variables') else 0
        mine = [jb for jb in jobs if (jb[1] == 'FM_V2' and fm_map.get(jb[0]) == s) or
                jb[1] in (f'EQUIL_V2_{s}', f'LAUNCH_{s}', f'CONV_{s}')]
        has_inputs = all(os.path.isfile(f'{V2}/equil_hmr/{s}/step_5.{e}') for e in ('pdb', 'xml'))
        log = last_fm_log(s)
        tail = open(log, errors='replace').read()[-4000:] if log else ''
        action, why = decide(s, n, mine, has_inputs, tail, state, os.path.isfile(f'{out}/convergence.json'))
        all_done &= action == 'DONE'
        print(f'[{now}] {s:30s} {n:>2}/{TARGET}  {action:9s} {why}')

        if action == 'FLAG':
            flags.append(s)
        elif action == 'RESUBMIT' and not a.dry_run:
            r = subprocess.run(['bash', f'{V2}/scripts/submit_fm_v2.sh', s], capture_output=True, text=True)
            ok = 'link 0:' in r.stdout
            print(f'           -> submit_fm_v2.sh {"OK: " + r.stdout.split("link 0:")[1].split()[0] if ok else "REFUSED (retry next round): " + (r.stderr or r.stdout).strip()[-300:]}')
            if ok:
                st = state.get(s, {})
                state[s] = {'n': n, 'count': st.get('count', 0) + 1 if st.get('n') == n else 1}
                actions.append(s)
        elif action == 'CONVERGE' and not a.dry_run:
            r = subprocess.run(['sbatch', '--parsable', '-J', f'CONV_{s}', f'{V2}/scripts/CONVERGENCE.job',
                                out, f'{V2}/equil_hmr/{s}/step_5.pdb'], capture_output=True, text=True)
            print(f'           -> convergence job {r.stdout.strip() or r.stderr.strip()}')

    if not a.dry_run:
        json.dump(state, open(STATE, 'w'), indent=1)
    print(f'[{now}] summary: resubmitted {actions or "none"}; flagged {flags or "none"}')
    print(f'ALL_DONE={int(all_done)}')
    sys.exit(1 if flags else 0)


if __name__ == '__main__':
    main()

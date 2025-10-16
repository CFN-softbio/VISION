import os
import argparse
from copy import deepcopy
from execution_cache import ExecutionCache
import Levenshtein
import json
import pexpect
import difflib
from epics import caput_many, camonitor
from datetime import datetime
import sys
import re, time
from pprint import pprint
from scipy.stats import linregress
import numpy as np
from codebleu import calc_codebleu
import tomllib

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from src.hal_beam_com.model_manager import ModelManager
from utils import strip_comments, calculate_mape, print_log_comparison
from base_test_framework import BaseTestFramework
from src.hal_beam_com.cogs.op_cog import invoke
from src.hal_beam_com.utils import get_data_field, CogType
from src.hal_beam_com.pv_config import get_pv_config


class ExecutionCancellationError(Exception):
    """Raised when snippet execution must be canceled due to timeout or hang."""

class ExecutionFailedError(Exception):
    """Raised when an error was detected in the execution of a code snippet."""
    
    def __init__(self, message="Execution failed"):
        self.message = message
        super().__init__(self.message)


def is_temp_match(gt_temp_logs, pred_temp_logs,
                   mae_threshold=5.0, final_temp_threshold=5.0):
    """
    Compare temperature progressions using MAE and final temperature accuracy.

    Args:
        gt_temp_logs: Ground truth temperature logs [(timestamp, temp), ...]
        pred_temp_logs: Predicted temperature logs
        mae_threshold: Maximum acceptable MAE in °C for score=0 (default: 5.0°C)
        final_temp_threshold: Maximum final temperature difference in °C for score=0 (default: 5.0°C)

    Returns:
        (binary_match, continuous_score, details)
    """
    if len(gt_temp_logs) == 0 and len(pred_temp_logs) == 0:
        # Both empty - perfect match
        return True, 1.0, {"reason": "both empty", "score": 1.0}

    if len(gt_temp_logs) == 0 or len(pred_temp_logs) == 0:
        # One empty, one not - no match
        return False, 0.0, {"reason": "one empty", "score": 0.0}

    # Extract just the temperature values (ignoring timestamps)
    gt_temps = np.array([temp for _, temp in gt_temp_logs])
    pred_temps = np.array([temp for _, temp in pred_temp_logs])

    # Pad the shorter sequence with its last value
    max_len = max(len(gt_temps), len(pred_temps))
    if len(gt_temps) < max_len:
        gt_temps = np.pad(gt_temps, (0, max_len - len(gt_temps)), 'edge')
    elif len(pred_temps) < max_len:
        pred_temps = np.pad(pred_temps, (0, max_len - len(pred_temps)), 'edge')

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(gt_temps - pred_temps))

    # Calculate final temperature difference
    final_temp_diff = abs(float(gt_temps[-1]) - float(pred_temps[-1]))

    # Calculate continuous scores using exponential decay
    # MAE score: exponential decay with characteristic scale of 15°C
    mae_score = np.exp(-mae / 15.0)
    
    # Final temp score: exponential decay with characteristic scale of 15°C
    final_temp_score = np.exp(-final_temp_diff / 15.0)

    # Weighted average for overall score (70% MAE, 30% final temp)
    continuous_score = 0.7 * mae_score + 0.3 * final_temp_score

    # Binary match decision (both conditions must be satisfied)
    binary_match = (mae <= mae_threshold and
                   final_temp_diff <= final_temp_threshold)

    # Print diagnostic information if mismatch
    if not binary_match:
        print(f"Temperature progression mismatch:")
        print(f"  MAE: {mae:.2f}°C (score: {mae_score:.3f})")
        print(f"  Final temp diff: {final_temp_diff:.2f}°C (score: {final_temp_score:.3f})")
        print(f"  Overall score: {continuous_score:.3f}")
        if len(gt_temps) <= 10:  # Only print if not too many values
            print(f"  GT temps: {gt_temps}")
            print(f"  Pred temps: {pred_temps}")

    details = {
        "mae": mae,
        "final_temp_diff": final_temp_diff,
        "mae_score": mae_score,
        "final_temp_score": final_temp_score,
        "score": continuous_score
    }

    return binary_match, continuous_score, details


class OpAgentTestFramework(BaseTestFramework):
    # Define IPython prompt pattern as a class constant
    IPYTHON_PROMPT = r'In\s*\[[0-9]+\]:\s*'
    # ANSI-tolerant error detection regexes and helper to build expect patterns
    ERROR_TRACEBACK_RE = re.compile(r'(?m)^\s*(?:\x1b\[[0-9;]*m)*Traceback \(most recent call last\):')
    ERROR_HEADER_RE = re.compile(r'(?m)^\s*(?:\x1b\[[0-9;]*m)*An exception has occurred, use .+ to see the full traceback\.')
    ERROR_LINE_RE = re.compile(r'(?m)^\s*(?:\x1b\[[0-9;]*m)*(?!except\b)[A-Za-z_]\w*(?:Error|Exception|Interrupt|Exit)(?::|$)')

    def _error_expect_patterns(self):
        return [self.IPYTHON_PROMPT, self.ERROR_TRACEBACK_RE, self.ERROR_HEADER_RE, self.ERROR_LINE_RE]

    def __init__(self, dataset_path, results_path, *,
                 scientist_file=None,
                 base_model='mistral',
                 system_prompt_path=None,
                 num_runs=1, debug=False,
                 use_gt_cache=True, use_cache=True,
                 complex_only=False, simple_only=False,
                 beamline='11BM'):

        super().__init__(dataset_path, results_path,
                         system_prompt_path=system_prompt_path,
                         base_model=base_model, num_runs=num_runs)

        self.log_file = None
        self.monitor_logs = []
        self._current_test_start_time: float | None = None   # wall-clock (sec) when we begin logging
        self.interactive_shell = None
        self.debug = debug
        self.use_gt_cache = use_gt_cache
        self.use_cache = use_cache
        self.complex_only = complex_only
        self.simple_only = simple_only

        self._idx = 0       # running counter for entries

        self.beamline = beamline

        # ── load scientists' answers if provided ────────────────────────────
        self.scientist_map = {}
        if scientist_file:
            with open(scientist_file, 'rb') as f:
                self.scientist_map = tomllib.load(f)    # {command: {'code': …}}
            print(f"Loaded {len(self.scientist_map)} beamline scientist answers "
                  f"from {scientist_file}")
        
        # CATEGORY 1: Functions to mock (log but remove from execution)
        # Dictionary mapping function names to regex patterns
        self.functions_to_mock = {
            # capture the call **plus** any trailing whitespace so no stray
            # space/indent is left behind after removal
            "sam.align": r"sam\.align\s*\([^)]*\)\s*;?\s*"
            # Add more functions here that should be logged but not executed
        }
        
        # CATEGORY 2: Functions to log with parameters (log but keep in code)
        # Dictionary mapping function names to regex patterns and parameter extraction config
        self.functions_to_log = {
            "sam.setOrigin": {
                "pattern": r"sam\.setOrigin\(\s*(\[[^\]]*\])\s*(?:,\s*(\[[^\]]*\]))?\s*\)",
                "param_format": lambda matches: f"{matches[0]}, {matches[1] if len(matches) > 1 and matches[1] else 'None'}",
                "compare_params": False
            },
            "wsam": {                                    # NEW ─ log but keep in code
                "pattern": r"wsam\s*\(\s*\)",            # match `wsam()` with any spacing
                "param_format": lambda matches: "",      # no parameters to record
                "compare_params": False                  # value is ignored during comparison
            },
            "sam.pos": {
                "pattern": r"sam.pos\s*\(\s*\)",
                "param_format": lambda matches: "",
                "compare_params": False
            }
        }
        
        # Validate that complex_only and simple_only are not both True
        if self.complex_only and self.simple_only:
            raise ValueError("Cannot specify both complex_only and simple_only")
        
        # Set up logging to file
        self.console_log_path = os.path.join(os.path.dirname(self.results_path), 'console_output.log')
        self.console_log_file = open(self.console_log_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
        # Create a custom stdout that writes to both console and file
        self.original_stdout = sys.stdout
        sys.stdout = self
        
        # Initialize the execution cache
        if self.use_gt_cache or self.use_cache or self.debug:
            cache_file_path = os.path.join(os.path.dirname(os.path.sep.join(results_path.split(os.path.sep)[:-3])), 'execution_cache.json')
            self.execution_cache = ExecutionCache(cache_file_path)
        else:
            self.execution_cache = None
            
        # Log the start of the test run
        print(f"Test run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {self.results_path}")
        print(f"Console output is being logged to: {self.console_log_path}")

        _, self.pvs_defaults = get_pv_config(self.beamline)

        # --- CodeBLEU-related column names ---------------------------------
        _cb_keys = list(calc_codebleu(["print('x')"], ["print('x')"], lang="python").keys())
        _cb_keys = ["average_CodeBLEU" if k == "codebleu" else k for k in _cb_keys]
        _cb_keys += ["comb_7_CodeBLEU"]
        self.fieldnames = ([
                            'command', 'expected_codes', 'generated_code', 'exact_match', 'inference_time',
                            'pv_match_rate', 'pv_mismatch_rate', 'exact_pv_match',
                            'timing_match', 'temp_match', 'full_match',
                            'timing_score', 'temp_score', 'full_score',
                            'temperature_involved',
                            'pv_value_matches', 'pv_total_pairs', 'num_gt_pv_events', 'num_pred_pv_events',
                            'timing_r2', 'timing_slope', 'timing_mape', 'timing_duration_ratio',
                            'timing_details', 'temp_details',
                            'codebleu_scores', 'levenshtein_distances', 'normalized_levenshtein_distances',
                            'best_normalized_levenshtein_distance', 'is_complex',
                            'gt_pv_logs', 'pred_pv_logs', 'ignore_pvs', 'aligned_gt_logs', 'aligned_pred_logs',
                            'ground_truth_snippet', 'gt_temp_logs', 'pred_temp_logs'
                           ]
                           + _cb_keys
                           + ['best_codebleu_score', 'best_levenshtein_distance', 'average_levenshtein_distance'])

        # Single dictionary to track all metrics by complexity
        self.metrics = {
                     'simple': {'results': [], 'execution_times': [], 'count': 0, 'full_matches': 0,
                               'exact_code_match_count': 0,
                               'pv_exact_match_count': 0, 'pv_exact_mismatch_count': 0, 'pv_match_rates': [],
                               'pv_mismatch_rates': [],
                               'timing_match_count': 0, 'timing_mismatch_count': 0,
                               'timing_scores': [], 'temp_scores': [], 'full_scores': [],
                               'temp_match_count': 0, 'temp_mismatch_count': 0,
                               'full_match_count': 0, 'full_mismatch_count': 0},
                     'complex': {'results': [], 'execution_times': [], 'count': 0, 'full_matches': 0,
                                'exact_code_match_count': 0,
                                'pv_exact_match_count': 0, 'pv_exact_mismatch_count': 0, 'pv_match_rates': [],
                                'pv_mismatch_rates': [],
                                'timing_match_count': 0, 'timing_mismatch_count': 0,
                                'timing_scores': [], 'temp_scores': [], 'full_scores': [],
                                'temp_match_count': 0, 'temp_mismatch_count': 0,
                                'full_match_count': 0, 'full_mismatch_count': 0}
                     }

        if not self.debug and not scientist_file:
            # Preload the model and do a warm-up request
            model = ModelManager.get_model(self.base_model)

            # We should not use model cache for testing, because we're not using an actual model
            if not self.use_cache:
                print("Performing warm-up request...")
                warmup_data = [{
                    'beamline': self.beamline,
                    'text_input': 'print hello',
                    'include_context_functions': True,
                    'only_text_input': 1,
                    'operator_cog_history': "",
                    'operator_cog_db_history': "",
                    'user_id': "test",
                }]
                invoke(warmup_data, base_model=self.base_model, finetuned=False, system_prompt_path=self.system_prompt_path)
                print("Warm-up complete")

        self.start_ipython_env()
        self.setup_monitoring()

    def _log_entries_match(self, gt, pred):
        gt_pv, _, gt_val = gt
        pr_pv, _, pr_val = pred
        if gt_pv != pr_pv:
            return False
        if gt_pv in self.functions_to_log and \
           self.functions_to_log[gt_pv].get("compare_params") is False:
            return True
        try:
            return abs(float(gt_val) - float(pr_val)) < 1e-3
        except (ValueError, TypeError):
            return str(gt_val) == str(pr_val)

    # ------------------------------------------------------------------
    # PV‑name comparison that is used for sequence alignment _only_
    # (value equality is checked later, after the alignment is fixed)
    # ------------------------------------------------------------------
    def _canonical_pv(self, pv: str) -> str:
        """Returns a canonical identifier for the PV (aliases can be added here if desired)."""
        return pv

    def _pv_equal(self, pv_a: str, pv_b: str) -> bool:
        return self._canonical_pv(pv_a) == self._canonical_pv(pv_b)

    # ------------------------------------------------------------------
    # Linear-regression helper: R² + slope + duration + MAPE
    # ------------------------------------------------------------------
    def _timing_match(
        self,
        gt_ts, pr_ts,
        r2_thresh=0.90,
        slope_lo=0.8, slope_hi=1.2,
        dur_tol=0.25,          # 25% tolerance on total duration
        mape_tol=1          # 100% tolerance on intervals
    ):
        """
        Compare two sequences of timestamps.
        Returns (match: bool, score: float, details: dict)
        """
        n = len(gt_ts)

        # Different number of timestamps -> immediate failure
        if n != len(pr_ts):
            return False, 0.0, {"reason": "length mismatch", "score": 0.0}

        # One (or zero) timestamp: no timing regression can be computed,
        # therefore consider the timing a match. This avoids false negatives
        # when only a single intercepted call is logged (e.g., sam.align).
        if n < 2:
            return True, 1.0, {"score": 1.0}

        gt = np.asarray(gt_ts) - gt_ts[0]  # shift timestamps so first is zero
        pr = np.asarray(pr_ts) - pr_ts[0]

        res = linregress(gt, pr)
        r2 = res.rvalue ** 2
        slope = res.slope

        # Calculate continuous score components
        r2_score = r2

        # Slope score: perfect=1.0 at slope=1, decreases as slope deviates
        slope_score = 1.0 - min(abs(slope - 1.0), 1.0)

        # Duration score
        dur_gt, dur_pr = gt[-1], pr[-1]
        if dur_gt == 0:
            duration_score = 0.0
        else:
            duration_diff = abs(dur_pr - dur_gt) / dur_gt
            duration_score = max(0, 1.0 - duration_diff / dur_tol)

        # MAPE score on intervals
        gt_intervals = np.diff(gt)
        pr_intervals = np.diff(pr)
        mape = calculate_mape(gt_intervals, pr_intervals)
        mape_score = max(0, 1.0 - mape / mape_tol)

        # Combined timing score (weighted average)
        timing_score = (0.4 * r2_score + 0.2 * slope_score +
                        0.2 * duration_score + 0.2 * mape_score)

        # Binary match decision (keep existing logic)
        binary_match = (r2 >= r2_thresh and
                        slope_lo <= slope <= slope_hi and
                        (dur_gt == 0 or abs(dur_pr - dur_gt) / dur_gt <= dur_tol) and
                        mape <= mape_tol)

        details = {
            "r2": r2,
            "slope": slope,
            "mape": mape,
            "duration_ratio": dur_pr / (dur_gt or 1),
            "score": timing_score,
            "r2_score": r2_score,
            "slope_score": slope_score,
            "duration_score": duration_score,
            "mape_score": mape_score
        }

        return binary_match, timing_score, details

    def _compute_pv_match_metrics(self, gt_logs, pr_logs):
        n = len(gt_logs)
        if n == 0:
            # When no PV changes (or caught functions) have been observed, then exact match should've triggered,
            # otherwise the codes can't be correct/equivalent if no PVs changes were tracked.
            # All the examples should either be exactly matchable, or produce PV changes that are tracked.
            return 0.0, 1.0, False, 0, [], []

        # Lists with only the canonical PV names
        gt_tokens = [self._canonical_pv(pv) for pv, _, _ in gt_logs]
        pr_tokens = [self._canonical_pv(pv) for pv, _, _ in pr_logs]

        # Alignment using difflib.SequenceMatcher (favors earlier matches)
        sm = difflib.SequenceMatcher(None, gt_tokens, pr_tokens, autojunk=False)
        aligned_gt, aligned_pr = [], []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":               # Identical PVs
                for k in range(i2 - i1):
                    aligned_gt.append(gt_logs[i1 + k])
                    aligned_pr.append(pr_logs[j1 + k])
            elif tag == "delete":            # Only in GT
                for k in range(i1, i2):
                    aligned_gt.append(gt_logs[k])
                    aligned_pr.append(None)
            elif tag == "insert":            # Only in Pred
                for k in range(j1, j2):
                    aligned_gt.append(None)
                    aligned_pr.append(pr_logs[k])
            else:                            # 'replace' → differences
                span = max(i2 - i1, j2 - j1)
                for off in range(span):
                    aligned_gt.append(gt_logs[i1 + off] if i1 + off < i2 else None)
                    aligned_pr.append(pr_logs[j1 + off] if j1 + off < j2 else None)

        # Comparing values for aligned pairs
        value_matches = sum(
            1
            for g, p in zip(aligned_gt, aligned_pr)
            if g is not None and p is not None and self._log_entries_match(g, p)
        )

        total_pairs = len(aligned_gt)         # this is equal to the number of rows in the comparison table
        mismatches  = total_pairs - value_matches
        mismatch_rate = mismatches / total_pairs if total_pairs else 0.0

        rate  = value_matches / total_pairs if total_pairs else 1.0
        exact = value_matches == n and value_matches == len(pr_logs)
        return rate, mismatch_rate, exact, value_matches, aligned_gt, aligned_pr

    def start_ipython_env(self):
        print("Setting up command...")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        example_services = os.path.join(repo_root, "external", "example-services")
        profile_collection = os.path.join(repo_root, "external", "profile_collection")

        cmd = (
            f'source $(conda info --base)/etc/profile.d/conda.sh && '
            f'conda activate 2024-3.0-py311-tiled && '
            f'source {example_services}/environment.sh && '
            f'cd {profile_collection} && '
            './.ci/apply-autosettings.sh &&'
            'ipython --profile-dir=. --no-term-title --simple-prompt'
        )

        print("Starting bash...")
        self.interactive_shell = pexpect.spawn('/bin/bash', encoding='utf-8', timeout=30)

        # Create log file in the same directory as results
        log_path = os.path.join(os.path.dirname(self.results_path), 'pexpect.log')
        self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)  # Line buffered
        self.interactive_shell.logfile = self.log_file #sys.stdout

        print("Running IPython...")
        self.interactive_shell.sendline(cmd)

        # Wait for IPython prompt
        self.interactive_shell.expect('In \\[[0-9]+\\]: ', timeout=300)

        print("Sending drop-in command...")
        self.interactive_shell.sendline('%run -i ./.ci/linkam-drop-in.py')

        return self.interactive_shell

    def _monitor_callback(self, pvname=None, value=None,
                          char_value=None, timestamp=None, **kwargs):
        """
        EPICS monitor callback.  Ignore PV changes that occurred before the
        current snippet’s start-time, so previous test activity does not leak
        into the present logs.
        """
        # Use the EPICS timestamp if provided, otherwise the local clock
        ts = float(timestamp) if timestamp is not None else time.time()

        # Accept only events that happen during/after the active test window
        if (self._current_test_start_time is None) or (ts >= self._current_test_start_time):
            self.monitor_logs.append((pvname, ts, value))

    def setup_monitoring(self):
        self.monitor_logs = []
        for pv in self.pvs_defaults:
            camonitor(pv, callback=self._monitor_callback)
        # Give monitors time to establish fully
        time.sleep(1)

    def reset_pvs(self):
        pvs    = list(self.pvs_defaults.keys())
        values = list(self.pvs_defaults.values())

        # 1st try – blocking put (wait until all PVs report success)
        results = caput_many(
            pvs, values,
            wait="all",
            connection_timeout=5,
            put_timeout=60,
        )

        # Collect PVs that did NOT complete (caput_many returns -1)
        failed_pvs  = [pv   for pv, r in zip(pvs, results) if r == -1]

        if failed_pvs:
            print(f"Warning: {len(failed_pvs)} PV resets timed-out – "
                  "retrying with instant motor-teleport.")

            # ---------- classify failures ---------------------------------
            motor_pvs   = [pv for pv in failed_pvs if pv.endswith("Mtr")]
            other_pvs   = [pv for pv in failed_pvs if pv not in motor_pvs]

            # ---------- instant reset for motors --------------------------
            if motor_pvs:
                motor_vals = [self.pvs_defaults[pv] for pv in motor_pvs]
                set_fields = [pv + ".SET" for pv in motor_pvs]
                val_fields = [pv + ".VAL" for pv in motor_pvs]

                # 1) enter SET-mode
                caput_many(set_fields, [1]*len(set_fields),
                           wait="all", connection_timeout=5, put_timeout=60)
                # 2) override VAL (teleport)
                caput_many(val_fields, motor_vals, wait=False)
                # 3) leave SET-mode
                caput_many(set_fields, [0]*len(set_fields), wait=False)

            # ---------- non-motor PVs: simple fire-and-forget --------------
            if other_pvs:
                other_vals = [self.pvs_defaults[pv] for pv in other_pvs]
                caput_many(other_pvs, other_vals, wait=False)

        # short pause to allow network traffic to settle
        time.sleep(2)

    def initialize_environment(self):
        """Initialize the testing environment with required objects and settings."""
        print("Waiting for initial prompt...")

        patterns = self._error_expect_patterns()
        try:
            index = self.interactive_shell.expect(patterns, timeout=60)
            print(f"Got initial prompt with index {index}")
            # print("Buffer after prompt:", repr(self.interactive_shell.before))
            # print("After buffer:", repr(self.interactive_shell.after))
        except pexpect.TIMEOUT:
            print("Timed out waiting for initial prompt")
            # print("Final buffer state:", repr(self.interactive_shell.before))
            # print("Final after buffer:", repr(self.interactive_shell.after))
            raise

        print("Initializing sample...")
        self.interactive_shell.sendline("sam = Sample('test')")
        print("Waiting for prompt after sample init...")
        
        try:
            index = self.interactive_shell.expect(patterns, timeout=30)
            print(f"Got sample init prompt with index {index}")
            print("Buffer after sample init:", repr(self.interactive_shell.before))
            print("After buffer:", repr(self.interactive_shell.after))
            print("Sample initialized")
        except pexpect.TIMEOUT:
            print("Timed out waiting for sample init prompt")
            print("Final buffer state:", repr(self.interactive_shell.before))
            print("Final after buffer:", repr(self.interactive_shell.after))
            raise

    def pre_process_code(self, code_string):
        """Process code to handle function calls that need special treatment."""
        
        # Process functions that should be mocked out
        for func, pattern in self.functions_to_mock.items():
            matches = re.findall(pattern, code_string)
            if matches:
                for _ in matches:
                    # Log a dummy PV event for the mocked function call
                    self.monitor_logs.append((func, time.time(), 0))
                code_string = re.sub(pattern, "", code_string)
        
        # Process functions that should be logged with parameters but still executed
        for func, config in self.functions_to_log.items():
            pattern = config["pattern"]
            param_formatter = config["param_format"]
            
            matches = re.findall(pattern, code_string)
            for match in matches:
                # Use the custom formatting function to format parameters for logging
                param_value = param_formatter(match if isinstance(match, tuple) else (match,))
                
                # Log the function call with its parameters
                self.monitor_logs.append((func, time.time(), param_value))
        
        return code_string

    def execute_snippet_in_mock_env(self, code_string, is_gt=False, is_complex=False):
        # Check if we have cached results for this code snippet
        # Previously we had that debug mode would re-run GT here, but I don't think that makes sense.
        if self.use_cache or (is_gt and self.use_gt_cache) and self.execution_cache:
            cached_logs = self.execution_cache.get(code_string)

            # Previously our error regex was too inclusive, leading try except code throw a false positive.
            # Therefore, if the code contains 'error' (case-insensitive), bypass any cached result
            # and force re-execution to refresh potentially corrupted cache entries.
            if cached_logs is not None and re.search(r'error', code_string, re.IGNORECASE):
                print("[CACHE] Bypassing cache because code contains 'error' (case-insensitive); will re-execute and refresh cache.")
                cached_logs = None
            
            # Validate cache quality - don't use suspiciously small logs for complex code
            if cached_logs is not None and is_gt:
                # For ground truth execution, check if we have enough logs
                # Complex code with loops should generate multiple PV changes
                if len(cached_logs) < 3:
                    print(f"Warning: Cached ground truth logs only have {len(cached_logs)} entries")
                    if is_complex:
                        print("This is a complex example but has few PV changes in cache. Executing instead.")
                        cached_logs = None
            
            if cached_logs is not None:
                print(f"Using cached execution logs ({len(cached_logs)} entries)")
                return cached_logs
        
        # If not in cache or cache disabled, execute the code
        print("Executing code snippet (not in cache)")
        self.reset_pvs()
        self.monitor_logs = []  # Reset logs before execution
        # Mark the beginning of the logging window – anything earlier will be ignored
        self._current_test_start_time = time.time()

        try:
            # Pre-process the code to remove (and log) mocked function calls
            code_string = self.pre_process_code(code_string)
            self.initialize_environment()

            # Send the command
            print(f"Sending command: {code_string}")
            # Use %cpaste magic instead of sendline
            self.interactive_shell.sendline("%cpaste -q")  # -q for quiet mode
            self.interactive_shell.sendline(code_string)
            self.interactive_shell.sendline("--")  # End of paste marker

            # self.interactive_shell.sendline('')  # End current block

            # TODO: Perhaps only for complex control flows ending on for loops?
            self.interactive_shell.sendline('')  # Additional newline to confirm termination

            # Look for either the next prompt or an error
            patterns = self._error_expect_patterns()

            while True:
                try:
                    index = self.interactive_shell.expect(patterns, timeout=300)
                    
                    if index == 0:  # Got prompt
                        break
                    elif index in [1, 2, 3]:  # Got error
                        # Start with what we have now (header/traceback/type)
                        error_message = self.interactive_shell.before + self.interactive_shell.after

                        # Try to collect the rest of the error block (exception line and "See ..." line) until the next prompt
                        try:
                            post_error_patterns = [
                                self.ERROR_LINE_RE,
                                re.compile(r'(?m)^\s*(?:\x1b\[[0-9;]*m)*See .+ for the full traceback\.'),
                                re.compile(self.IPYTHON_PROMPT),
                            ]
                            while True:
                                try:
                                    idx2 = self.interactive_shell.expect(post_error_patterns, timeout=2)
                                    error_message += self.interactive_shell.before + self.interactive_shell.after
                                    if idx2 == 2:  # Prompt reached
                                        break
                                except pexpect.TIMEOUT:
                                    # No more output to collect
                                    break
                        except Exception:
                            # If anything goes wrong while collecting, fall back to what we already captured
                            pass

                        # Format the error message for better readability
                        formatted_error = "\n" + "*"*80 + "\n"
                        formatted_error += "** ERROR DETECTED:\n"
                        formatted_error += "** " + "-"*76 + "\n"
                        for line in error_message.strip().split('\n'):
                            formatted_error += f"** {line}\n"
                        formatted_error += "*"*80
                        print(formatted_error)
                        
                        # Capture logs before raising the exception
                        logs_snapshot = deepcopy(self.monitor_logs)
                        
                        # Cache policy: do not cache an empty GT log, but tell the user.
                        if is_gt and len(logs_snapshot) == 0:
                            print("[INFO] Ground-truth snippet produced no PV logs – nothing cached.")
                        elif (self.use_cache or self.debug) and self.execution_cache:
                            self.execution_cache.put(code_string, logs_snapshot)
                        
                        # Raise exception but include the logs we captured
                        error = ExecutionFailedError(error_message.strip())
                        error.logs = logs_snapshot
                        raise error
                except pexpect.TIMEOUT:
                    print("Timeout in pattern match loop")
                    print("Current buffer:", repr(self.interactive_shell.before))
                    print("Current after buffer:", repr(self.interactive_shell.after))
                    
                    # Capture logs before raising the exception
                    logs_snapshot = deepcopy(self.monitor_logs)
                    
                    # Cache policy: do not cache an empty GT log, but tell the user.
                    if is_gt and len(logs_snapshot) == 0:
                        print("[INFO] Ground-truth snippet produced no PV logs – nothing cached.")
                    elif (self.use_cache or self.debug) and self.execution_cache:
                        self.execution_cache.put(code_string, logs_snapshot)
                    
                    raise pexpect.TIMEOUT("Timeout in pattern match loop")

            logs_snapshot = deepcopy(self.monitor_logs)
            
            # Cache policy: do not cache an empty GT log, but tell the user.
            if is_gt and len(logs_snapshot) == 0:
                print("[INFO] Ground-truth snippet produced no PV logs – nothing cached.")
            elif (self.use_cache or self.debug) and self.execution_cache:
                self.execution_cache.put(code_string, logs_snapshot)
                
            return logs_snapshot

        except pexpect.TIMEOUT:
            print("Execution timed out, sending Ctrl-C (twice) to shell.")
            self.interactive_shell.sendintr()  # Ctrl+C
            time.sleep(1)
            self.interactive_shell.sendintr()
            time.sleep(1)

            print("Sending RE.abort() to clean up.")
            self.interactive_shell.sendline("RE.abort()")
            time.sleep(1)

            # Capture logs before raising the exception
            logs_snapshot = deepcopy(self.monitor_logs)
            
            # Cache the execution logs even though there was a timeout
            if self.use_cache and self.execution_cache:
                self.execution_cache.put(code_string, logs_snapshot)
            
            # Create a cancellation error but include the logs we captured
            error = ExecutionCancellationError("Snippet timed out, canceled by operator")
            error.logs = logs_snapshot
            raise error
        except Exception as e:
            # Create a nicely formatted error box
            error_box = "\n" + "*"*80 + "\n"
            error_box += "** EXECUTION ERROR\n"
            error_box += "** " + "-"*76 + "\n"
            
            # Add exception type and message
            error_box += f"** Type: {type(e).__name__}\n"
            
            # Only show detailed message if it's not already shown above
            if isinstance(e, ExecutionFailedError):
                error_box += f"** Details: See error output above\n"
            else:
                error_box += f"** Details: {str(e)}\n"
                
                # Only show buffer contents for non-ExecutionFailedError exceptions
                # since ExecutionFailedError already shows the full output
                if self.interactive_shell.before.strip():
                    error_box += "** " + "-"*76 + "\n"
                    error_box += "** Last buffer content:\n"
                    error_box += "** " + "-"*76 + "\n"
                    for line in self.interactive_shell.before.strip().split('\n'):
                        error_box += f"** {line}\n"
                
                if self.interactive_shell.after.strip():
                    error_box += "** " + "-"*76 + "\n"
                    error_box += "** After buffer content:\n"
                    error_box += "** " + "-"*76 + "\n"
                    for line in self.interactive_shell.after.strip().split('\n'):
                        error_box += f"** {line}\n"
            
            error_box += "*"*80
            print(error_box)
            
            # For non-timeout, non-execution errors, we don't have logs to return
            if not hasattr(e, 'logs'):
                # Capture logs before re-raising the exception
                logs_snapshot = deepcopy(self.monitor_logs)
                
                # Cache policy: do not cache an empty GT log, but tell the user.
                if is_gt and len(logs_snapshot) == 0:
                    print("[INFO] Ground-truth snippet produced no PV logs – nothing cached.")
                elif self.use_cache and self.execution_cache:
                    self.execution_cache.put(code_string, logs_snapshot)
                
                # Add logs to the exception
                e.logs = logs_snapshot
            
            raise


    def test_entry(self, entry):
        self._idx += 1      # will be 1-based
        # Determine complexity type
        is_complex = entry.get('is_complex', False)
        complexity_type = 'complex' if is_complex else 'simple'

        # Skip based on complexity filter settings
        if hasattr(self, 'complex_only') and self.complex_only and not is_complex:
            print(f"Skipping simple example (complex_only={self.complex_only}).")
            return False
        
        if hasattr(self, 'simple_only') and self.simple_only and is_complex:
            print(f"Skipping complex example (simple_only={self.simple_only}).")
            return False

        command = entry['command']
        expected_codes = entry['expected_code']
        # Get list of PVs to ignore during comparison
        ignore_pvs = set(entry.get('ignore_pvs', []))

        data = [{
            'beamline': self.beamline,
            'text_input': command,
            'include_context_functions': True,
            'only_text_input': 1,
            'operator_cog_history': "",
            'operator_cog_db_history': "",
            'user_id': "test",
        }]

        # Execute both ground truth and predicted code
        ground_truth_snippet = expected_codes[0]  # Use first expected code

        start_time = time.time()

        # --- 1)  scientist-provided answer? ------------------------------------
        using_scientist_answer = command in self.scientist_map
        if using_scientist_answer:
            generated_code = self.scientist_map[command]['code']
            print(f"Using scientist answer for command: {command}")

        elif self.scientist_map:
            raise ValueError("Human code does not cover all code cases it is assigned to:", command)

        # --- 2)  debug mode  -----------------------------------------------------
        elif self.debug:
            self.system_prompt = "debug"
            generated_code = ground_truth_snippet

        # --- 3)  normal LLM invocation with retry logic for empty responses ----
        else:
            max_retries = 5
            retry_count = 0
            generated_code = ""
            
            while retry_count < max_retries:
                cog_result = invoke(data, base_model=self.base_model,
                                    finetuned=False,
                                    system_prompt_path=self.system_prompt_path)
                generated_code = cog_result[0][f'{CogType.OP.value}_cog_output']
                
                # Check if the response is empty or contains only whitespace
                if generated_code and generated_code.strip():
                    break  # Got a valid response, exit retry loop

                # Exponential backoff
                time.sleep(min(int(5 ** retry_count), 60))

                retry_count += 1
                if retry_count < max_retries:
                    print(f"Warning: Model returned empty response. Retrying... (attempt {retry_count + 1}/{max_retries})")
                else:
                    print(f"Error: Model returned empty response after {max_retries} attempts.")
                    # Keep the empty string if all retries failed

        end_time = time.time()
        inference_time = end_time - start_time
        self.execution_times.append(inference_time)
        print(f"{inference_time=}")

        # --------------------------------------------------------------
        # Pre-compute code-string similarity metrics before simulation
        # --------------------------------------------------------------
        generated_code_clean  = strip_comments(generated_code)
        expected_codes_clean  = [strip_comments(code) for code in expected_codes]

        exact_match = any(
            generated_code_clean.strip() == exp.strip()
            for exp in expected_codes_clean
        )
        if self.debug:
            exact_match = False

        codebleu_scores = []
        for exp in expected_codes_clean:
            # 2.1  old (= average) weighting -----------------------------
            _avg = calc_codebleu([exp], [generated_code_clean],
                                 lang="python",
                                 weights=(0.25, 0.25, 0.25, 0.25),
                                 tokenizer=None)

            # 2.2  new (= comb-7) weighting ------------------------------
            _c7  = calc_codebleu([exp], [generated_code_clean],
                                 lang="python",
                                 weights=(0.10, 0.10, 0.40, 0.40),
                                 tokenizer=None)

            # 2.3  merge into a single dict ------------------------------
            comb = _avg.copy()                          # keeps n-gram, syntax, …
            comb["average_CodeBLEU"] = comb.pop("codebleu")
            comb["comb_7_CodeBLEU"]  = _c7["codebleu"]

            codebleu_scores.append(comb)
        levenshtein_distances = [
            Levenshtein.distance(generated_code_clean, exp)
            for exp in expected_codes_clean
        ]
        normalized_levenshtein_distances = [
            d / max(len(generated_code_clean), len(exp))
            for d, exp in zip(levenshtein_distances, expected_codes_clean)
        ]

        best_codebleu_output                = max(codebleu_scores,
                                                  key=lambda x: x["comb_7_CodeBLEU"])
        best_levenshtein_distance           = min(levenshtein_distances)
        best_normalized_levenshtein_distance = min(normalized_levenshtein_distances)

        # ------------------------------------------------------------------
        # Default values for execution-comparison metrics; may be overwritten
        # in the non-exact-match branch.
        # ------------------------------------------------------------------
        pv_match_rate    = 1.0
        pv_mismatch_rate = 0.0
        exact_pv_match   = True

        # --------------------------------------------------------------
        #  If the generated code is an exact match, skip the simulator
        # --------------------------------------------------------------
        if exact_match and not self.debug:
            # One-line summary in case of exact match
            print(f"[{self._idx}] EXACT MATCH │ Command: {command} │ Predicted: {generated_code.strip()}")
            pv_match_rate        = 1.0
            pv_mismatch_rate     = 0.0
            exact_pv_match       = True
            timing_match         = True
            timing_score         = 1.0
            temp_match           = True      # no temp PVs involved
            temp_score           = 1.0
            full_match           = True
            full_score           = 1.0
            
            # For exact matches, we still need to store empty PV logs
            gt_pv_logs = []
            pred_pv_logs = []
            aligned_gt_logs = []
            aligned_pred_logs = []
            # New: dummy temp logs for CSV
            gt_temp_logs   = []
            pred_temp_logs = []

            # Additional metrics for CSV
            temperature_involved = False
            pv_value_matches = 0
            pv_total_pairs = 0
            num_gt_pv_events = 0
            num_pred_pv_events = 0
            timing_r2 = None
            timing_slope = None
            timing_mape = None
            timing_duration_ratio = None
            timing_details = {}
            temp_details = {}

            # ----------------------------------------------------------
            # UPDATE METRICS for this exact-match entry
            # ----------------------------------------------------------
            self.metrics[complexity_type]['exact_code_match_count'] += 1
            self.metrics[complexity_type]['pv_exact_match_count']  += 1
            self.metrics[complexity_type]['pv_match_rates'].append(1.0)
            self.metrics[complexity_type]['pv_mismatch_rates'].append(0.0)
            self.metrics[complexity_type]['timing_match_count']    += 1
            self.metrics[complexity_type]['timing_scores'].append(1.0)
            self.metrics[complexity_type]['temp_scores'].append(1.0)
            self.metrics[complexity_type]['full_scores'].append(1.0)
            self.metrics[complexity_type]['full_match_count']      += 1
            # (temperature counters stay unchanged – no temp PVs)

            if self.system_prompt is None and not self.debug and not using_scientist_answer:
                self.system_prompt = get_data_field(cog_result, CogType.OP, "system_prompt")
            self._record_result(
                complexity_type       = complexity_type,
                command               = command,
                expected_codes        = expected_codes,
                generated_code        = generated_code,
                exact_match           = exact_match,
                inference_time        = inference_time,
                pv_match_rate         = pv_match_rate,
                pv_mismatch_rate      = pv_mismatch_rate,
                exact_pv_match        = exact_pv_match,
                timing_match          = timing_match,
                temp_match            = temp_match,
                timing_score          = timing_score,
                temp_score            = temp_score,
                full_score            = full_score,
                temperature_involved  = temperature_involved,
                pv_value_matches      = pv_value_matches,
                pv_total_pairs        = pv_total_pairs,
                num_gt_pv_events      = num_gt_pv_events,
                num_pred_pv_events    = num_pred_pv_events,
                timing_r2             = timing_r2,
                timing_slope          = timing_slope,
                timing_mape           = timing_mape,
                timing_duration_ratio = timing_duration_ratio,
                timing_details        = timing_details,
                temp_details          = temp_details,
                codebleu_scores       = codebleu_scores,
                levenshtein_distances = levenshtein_distances,
                normalized_levenshtein_distances =
                    normalized_levenshtein_distances,
                best_codebleu_output  = best_codebleu_output,
                best_levenshtein_distance           =
                    best_levenshtein_distance,
                best_normalized_levenshtein_distance =
                    best_normalized_levenshtein_distance,
                is_complex            = is_complex,
                full_match            = full_match,
                gt_pv_logs            = gt_pv_logs,
                pred_pv_logs          = pred_pv_logs,
                ignore_pvs            = list(ignore_pvs),
                aligned_gt_logs       = aligned_gt_logs,
                aligned_pred_logs     = aligned_pred_logs,
                ground_truth_snippet  = ground_truth_snippet,
                gt_temp_logs          = gt_temp_logs,
                pred_temp_logs        = pred_temp_logs,
            )
            return full_match
        else:
            # In debug mode, even exact matches go through execution comparison
            if self.debug and exact_match:
                print(f"[{self._idx}] EXACT MATCH (DEBUG MODE - Testing execution) │ Command: {command}")
                # Still count it as an exact code match
                self.metrics[complexity_type]['exact_code_match_count'] += 1
            # ------------------------------------------------------------------
            # 0)  First execute *all* ground-truth snippets.  If none of them
            #     yields PV changes (after applying ignore_pvs + temperature filter),
            #     we can skip simulating the predicted code – a correct answer
            #     would then have been an exact code match.
            # ------------------------------------------------------------------
            temp_pv = "XF:11BM-ES:{LINKAM}:TEMP"
            any_pv_changes = False
            gt_logs_cache = {}
            gt_execution_failed = False
            for gt_snippet in expected_codes:
                try:
                    logs = self.execute_snippet_in_mock_env(
                        gt_snippet, is_gt=True, is_complex=is_complex
                    )
                except (ExecutionCancellationError, ExecutionFailedError) as exc:
                    logs = self._handle_execution_exception(
                        exc, is_ground_truth=True,
                        snippet_code=gt_snippet, command=command)
                    gt_execution_failed = True

                # Check if ground truth produced zero PV changes (concerning case)
                # TODO: Verify how many times this happens and if execute_ground_truth_with_retries was actually necessary.
                if is_complex and len(logs) == 0 and not gt_execution_failed:
                    print("\n" + "!"*80)
                    print("!! WARNING: Ground truth produced ZERO PV changes!")
                    print(f"!! Command: {command}")
                    print(f"!! Ground truth snippet:")
                    print("!! " + "-"*76)
                    for line in gt_snippet.split('\n'):
                        print(f"!! {line}")
                    print("!"*80 + "\n")

                # Check whether any logs remain after filtering.
                if [l for l in logs if l and l[0] not in ignore_pvs and l[0] != temp_pv]:
                    any_pv_changes = True

                gt_logs_cache[gt_snippet] = logs

            if not any_pv_changes:
                print(
                    "No PV changes detected in any ground-truth variant – "
                    "skipping simulation of the predicted code."
                )
                # basic metrics for this mismatch case
                pv_match_rate        = 0.0
                pv_mismatch_rate     = 1.0
                exact_pv_match       = False
                timing_match         = False
                timing_score         = 0.0
                temp_match           = False
                temp_score           = 0.0
                full_match           = False
                full_score           = 0.0

                gt_temp_logs   = []
                pred_temp_logs = []

                # update aggregate counters.
                self.metrics[complexity_type]['pv_exact_mismatch_count'] += 1
                self.metrics[complexity_type]['pv_match_rates'].append(0.0)
                self.metrics[complexity_type]['pv_mismatch_rates'].append(1.0)
                self.metrics[complexity_type]['timing_mismatch_count'] += 1
                self.metrics[complexity_type]['timing_scores'].append(0.0)
                self.metrics[complexity_type]['temp_scores'].append(0.0)
                self.metrics[complexity_type]['full_scores'].append(0.0)
                self.metrics[complexity_type]['full_mismatch_count'] += 1

                # log result and return early
                temperature_involved = False
                pv_value_matches = 0
                pv_total_pairs = 0
                num_gt_pv_events = 0
                num_pred_pv_events = 0
                timing_r2 = None
                timing_slope = None
                timing_mape = None
                timing_duration_ratio = None
                timing_details = {}
                temp_details = {}
                self._record_result(
                    complexity_type       = complexity_type,
                    command               = command,
                    expected_codes        = expected_codes,
                    generated_code        = generated_code,
                    exact_match           = exact_match,
                    inference_time        = inference_time,
                    pv_match_rate         = pv_match_rate,
                    pv_mismatch_rate      = pv_mismatch_rate,
                    exact_pv_match        = exact_pv_match,
                    timing_match          = timing_match,
                    temp_match            = temp_match,
                    timing_score          = timing_score,
                    temp_score            = temp_score,
                    full_score            = full_score,
                    temperature_involved  = temperature_involved,
                    pv_value_matches      = pv_value_matches,
                    pv_total_pairs        = pv_total_pairs,
                    num_gt_pv_events      = num_gt_pv_events,
                    num_pred_pv_events    = num_pred_pv_events,
                    timing_r2             = timing_r2,
                    timing_slope          = timing_slope,
                    timing_mape           = timing_mape,
                    timing_duration_ratio = timing_duration_ratio,
                    timing_details        = timing_details,
                    temp_details          = temp_details,
                    codebleu_scores       = codebleu_scores,
                    levenshtein_distances = levenshtein_distances,
                    normalized_levenshtein_distances =
                        normalized_levenshtein_distances,
                    best_codebleu_output  = best_codebleu_output,
                    best_levenshtein_distance           =
                        best_levenshtein_distance,
                    best_normalized_levenshtein_distance =
                        best_normalized_levenshtein_distance,
                    is_complex            = is_complex,
                    full_match            = full_match,
                    gt_pv_logs            = [],  # Empty for no PV changes case
                    pred_pv_logs          = [],
                    ignore_pvs            = list(ignore_pvs),
                    aligned_gt_logs       = [],
                    aligned_pred_logs     = [],
                    ground_truth_snippet  = ground_truth_snippet,
                    gt_temp_logs          = gt_temp_logs,
                    pred_temp_logs        = pred_temp_logs,
                )
                return full_match
            # ------------------------------------------------------------------
            # 1)  Execute the *predicted* code once – we will later compare all
            #     ground-truth variants against its PV log.
            # ------------------------------------------------------------------
            pred_logs = []
            pred_execution_failed = False
            try:
                print("Executing predicted code...")
                pred_logs = self.execute_snippet_in_mock_env(
                    generated_code, is_complex=is_complex
                )
            except (ExecutionCancellationError, ExecutionFailedError) as e:
                pred_execution_failed = True
                pred_logs = self._handle_execution_exception(
                    e, is_ground_truth=False,
                    snippet_code=generated_code, command=command
                )

            # ------------------------------------------------------------------
            # 2)  Evaluate *every* candidate ground-truth snippet
            # ------------------------------------------------------------------
            best_candidate = None
            best_score     = -1.0     # Use full_score for comparison

            for gt_snippet in expected_codes:
                candidate = self._evaluate_ground_truth_variant(
                    gt_snippet,
                    pred_logs,
                    ignore_pvs,
                    is_complex,
                    precomputed_logs=gt_logs_cache.get(gt_snippet),
                    command=command,
                )
                if best_candidate is None or candidate["full_score"] > best_score:
                    best_candidate = candidate
                    best_score = candidate["full_score"]
                    
                    # If we found a perfect match, no need to check other candidates
                    if best_score >= 1.0:
                        break

            # Unpack metrics of the elected best ground truth
            ground_truth_snippet      = best_candidate["snippet"]
            gt_logs                   = best_candidate["gt_logs"]
            pv_match_rate             = best_candidate["pv_match_rate"]
            pv_mismatch_rate          = best_candidate["pv_mismatch_rate"]
            exact_pv_match            = best_candidate["exact_pv_match"]
            timing_match              = best_candidate["timing_match"]
            timing_score              = best_candidate["timing_score"]
            temp_match                = best_candidate["temp_match"]
            temp_score                = best_candidate["temp_score"]
            full_match                = best_candidate["full_match"]
            full_score                = best_candidate["full_score"]
            aligned_gt                = best_candidate["aligned_gt"]
            aligned_pred              = best_candidate["aligned_pred"]
            pv_match                  = exact_pv_match   # keeps legacy name

            # Store the raw PV logs for later analysis
            gt_pv_logs = gt_logs
            pred_pv_logs = pred_logs
            aligned_gt_logs = aligned_gt
            aligned_pred_logs = aligned_pred

            timing_details = best_candidate["timing_details"]
            temp_details   = best_candidate["temp_details"]

            gt_temp_logs        = best_candidate["gt_temp_logs"]
            pred_temp_logs      = best_candidate["pred_temp_logs"]
            temperature_involved = best_candidate["temperature_involved"]

            # Derived CSV metrics
            pv_value_matches    = best_candidate.get("pv_value_matches", 0)
            pv_total_pairs      = best_candidate.get("pv_total_pairs", len(aligned_gt))
            num_gt_pv_events    = best_candidate.get("num_gt_pv_events", len(gt_logs))
            num_pred_pv_events  = best_candidate.get("num_pred_pv_events", len(pred_logs))
            timing_r2           = timing_details.get("r2") if isinstance(timing_details, dict) else None
            timing_slope        = timing_details.get("slope") if isinstance(timing_details, dict) else None
            timing_mape         = timing_details.get("mape") if isinstance(timing_details, dict) else None
            timing_duration_ratio = timing_details.get("duration_ratio") if isinstance(timing_details, dict) else None

            self._print_full_score_breakdown(
                command               = command,
                ground_truth_snippet  = ground_truth_snippet,
                predicted_snippet     = generated_code,
                pv_rate               = pv_match_rate,
                timing_score          = timing_score,
                temp_score            = temp_score,
                full_score            = full_score,
                timing_details        = timing_details,
                temp_details          = temp_details,
                temperature_involved  = temperature_involved,
                gt_temp_logs          = gt_temp_logs,
                pred_temp_logs        = pred_temp_logs,
            )

            print("Ground Truth logs:", gt_logs)
            print("Predicted logs:",    pred_logs)
            print("Ignoring PVs:",      ignore_pvs)

            # Construct display log lists from aligned sequences
            gt_logs_display, pred_logs_display = \
                self._construct_logs_tracked_functions(aligned_gt, aligned_pred)

            print_log_comparison(
                gt_logs_display,
                pred_logs_display,
                command,
                match_func=self._log_entries_match
            )

            # All remaining metric-updates, result recording and restart logic stay unchanged.

            # Full match may involve temperature – logic unchanged.

            print(f"Exact PV match (non-temp): {exact_pv_match}")
            print(f"PV match rate (non-temp): {pv_match_rate*100:.2f}%")
            print(f"PV mismatch rate (non-temp): {pv_mismatch_rate*100:.2f}%")
            print(f"Timing match: {timing_match} (score: {timing_score:.3f})")
            if temperature_involved:
                print(f"Temperature match: {temp_match} (score: {temp_score:.3f})")
            print(f"Full match: {full_match} (score: {full_score:.3f})")

            # Update metrics
            if exact_pv_match:
                self.metrics[complexity_type]['pv_exact_match_count'] += 1
            else:
                self.metrics[complexity_type]['pv_exact_mismatch_count'] += 1
            self.metrics[complexity_type]['pv_match_rates'].append(pv_match_rate)
            self.metrics[complexity_type]['pv_mismatch_rates'].append(pv_mismatch_rate)

            if timing_match:
                self.metrics[complexity_type]['timing_match_count'] += 1
            else:
                self.metrics[complexity_type]['timing_mismatch_count'] += 1
            self.metrics[complexity_type]['timing_scores'].append(timing_score)

            if temperature_involved:
                if temp_match:
                    self.metrics[complexity_type]['temp_match_count'] += 1
                else:
                    self.metrics[complexity_type]['temp_mismatch_count'] += 1
                self.metrics[complexity_type]['temp_scores'].append(temp_score)
            else:
                # No temperature involved, perfect temp score
                self.metrics[complexity_type]['temp_scores'].append(1.0)

            # Calculate full score
            if temperature_involved:
                full_score = 0.6 * pv_match_rate + 0.2 * timing_score + 0.2 * temp_score
            else:
                full_score = 0.8 * pv_match_rate + 0.2 * timing_score
            
            self.metrics[complexity_type]['full_scores'].append(full_score)
            
            if full_match:
                self.metrics[complexity_type]['full_match_count'] += 1
            else:
                self.metrics[complexity_type]['full_mismatch_count'] += 1

            if not full_match:
                mismatch_reasons = []
                if not exact_pv_match:
                    mismatch_reasons.append("Non-temperature PV logs do not match")
                if not timing_match:
                    mismatch_reasons.append("Timing intervals do not match")
                if not temp_match:
                    mismatch_reasons.append("Temperature progression does not match")

                self._print_command_and_snippets(
                    command=command,
                    ground_truth_snippet=ground_truth_snippet,
                    predicted_snippet=generated_code,
                    header="EXECUTION RESULTS MISMATCH",
                    mismatch_reasons=mismatch_reasons,
                )

            if not pv_match:
                print("Trying to restart the shell due to failed command")
                self.start_ipython_env()

        # Also display snippets for perfect PV matches (simple or complex)
        if full_match and not exact_match:
            self._print_command_and_snippets(
                command              = command,
                ground_truth_snippet = ground_truth_snippet,
                predicted_snippet    = generated_code,
                header               = "EXECUTION RESULTS (FULL MATCH)",
            )

        self._record_result(
            complexity_type       = complexity_type,
            command               = command,
            expected_codes        = expected_codes,
            generated_code        = generated_code,
            exact_match           = exact_match,
            inference_time        = inference_time,
            pv_match_rate         = pv_match_rate,
            pv_mismatch_rate      = pv_mismatch_rate,
            exact_pv_match        = exact_pv_match,
            timing_match          = timing_match,
            temp_match            = temp_match,
            timing_score=timing_score,
            temp_score=temp_score,
            full_score=full_score,
            temperature_involved  = temperature_involved,
            pv_value_matches      = pv_value_matches,
            pv_total_pairs        = pv_total_pairs,
            num_gt_pv_events      = num_gt_pv_events,
            num_pred_pv_events    = num_pred_pv_events,
            timing_r2             = timing_r2,
            timing_slope          = timing_slope,
            timing_mape           = timing_mape,
            timing_duration_ratio = timing_duration_ratio,
            timing_details        = timing_details,
            temp_details          = temp_details,
            codebleu_scores       = codebleu_scores,
            levenshtein_distances = levenshtein_distances,
            normalized_levenshtein_distances = normalized_levenshtein_distances,
            best_codebleu_output  = best_codebleu_output,
            best_levenshtein_distance           = best_levenshtein_distance,
            best_normalized_levenshtein_distance = best_normalized_levenshtein_distance,
            is_complex            = is_complex,
            full_match            = full_match,
            gt_pv_logs            = gt_pv_logs if not exact_match or self.debug else [],
            pred_pv_logs          = pred_pv_logs if not exact_match or self.debug else [],
            ignore_pvs            = list(ignore_pvs),
            aligned_gt_logs       = aligned_gt_logs if not exact_match or self.debug else [],
            aligned_pred_logs     = aligned_pred_logs if not exact_match or self.debug else [],
            ground_truth_snippet  = ground_truth_snippet,
            gt_temp_logs          = gt_temp_logs,
            pred_temp_logs        = pred_temp_logs,
        )

        return full_match

    # ------------------------------------------------------------------
    # Centralised bookkeeping + CSV writing for one test entry
    # ------------------------------------------------------------------
    def _record_result(
        self,
        *,
        complexity_type: str,
        command: str,
        expected_codes: list[str],
        generated_code: str,
        exact_match: bool,
        inference_time: float,
        pv_match_rate: float,
        pv_mismatch_rate: float,
        exact_pv_match: bool,
        timing_match: bool | None,
        temp_match: bool | None,
        timing_score: float,
        temp_score: float,
        full_score: float,
        temperature_involved: bool | None,
        pv_value_matches: int | None,
        pv_total_pairs: int | None,
        num_gt_pv_events: int | None,
        num_pred_pv_events: int | None,
        timing_r2: float | None,
        timing_slope: float | None,
        timing_mape: float | None,
        timing_duration_ratio: float | None,
        timing_details: dict | None,
        temp_details: dict | None,
        codebleu_scores: list[dict],
        levenshtein_distances: list[int],
        normalized_levenshtein_distances: list[float],
        best_codebleu_output: dict,
        best_levenshtein_distance: int,
        best_normalized_levenshtein_distance: float,
        is_complex: bool,
        full_match: bool,
        gt_pv_logs: list,
        pred_pv_logs: list,
        ignore_pvs: list,
        aligned_gt_logs: list,
        aligned_pred_logs: list,
        ground_truth_snippet: str | None = None,
        gt_temp_logs: list | None = None,
        pred_temp_logs: list | None = None,
    ) -> None:
        gt_temp_logs   = gt_temp_logs   or []
        pred_temp_logs = pred_temp_logs or []

        # --- update aggregated metrics ---------------------------------
        self.metrics[complexity_type]['results'].append({
            'codebleu':               best_codebleu_output,
            'levenshtein':            best_levenshtein_distance,
            'normalized_levenshtein': best_normalized_levenshtein_distance,
        })
        self.metrics[complexity_type]['execution_times'].append(inference_time)
        self.metrics[complexity_type]['count'] += 1
        if full_match:
            self.metrics[complexity_type]['full_matches'] += 1

        # --- write CSV row ---------------------------------------------
        result_data = {
            "best_levenshtein_distance":            best_levenshtein_distance,
            "best_normalized_levenshtein_distance": best_normalized_levenshtein_distance,
            "command":          command,
            "expected_codes":   json.dumps(expected_codes),
            "generated_code":   generated_code,
            "exact_match":      exact_match,
            "inference_time":   f"{inference_time:.5f} seconds",
            "pv_match_rate":    pv_match_rate,
            "pv_mismatch_rate": pv_mismatch_rate,
            "exact_pv_match":   exact_pv_match,
            "timing_match":     timing_match,
            "temp_match":       temp_match,
            "full_match":       full_match,
            "timing_score":     timing_score,
            "temp_score":       temp_score,
            "full_score":       full_score,
            "temperature_involved": temperature_involved,
            "pv_value_matches": pv_value_matches,
            "pv_total_pairs":   pv_total_pairs,
            "num_gt_pv_events": num_gt_pv_events,
            "num_pred_pv_events": num_pred_pv_events,
            "timing_r2":        timing_r2,
            "timing_slope":     timing_slope,
            "timing_mape":      timing_mape,
            "timing_duration_ratio": timing_duration_ratio,
            "timing_details":   json.dumps(timing_details or {}),
            "temp_details":     json.dumps(temp_details or {}),
            "codebleu_scores":  json.dumps(codebleu_scores),
            "levenshtein_distances":            json.dumps(levenshtein_distances),
            "normalized_levenshtein_distances": json.dumps(normalized_levenshtein_distances),
            "best_codebleu_score": best_codebleu_output,
            "is_complex":       is_complex,
            "gt_pv_logs":       json.dumps(gt_pv_logs),
            "pred_pv_logs":     json.dumps(pred_pv_logs),
            "ignore_pvs":       json.dumps(ignore_pvs),
            "aligned_gt_logs":  json.dumps(aligned_gt_logs),
            "aligned_pred_logs": json.dumps(aligned_pred_logs),
            "ground_truth_snippet": ground_truth_snippet or "",
            "gt_temp_logs": json.dumps(gt_temp_logs),
            "pred_temp_logs": json.dumps(pred_temp_logs),
        }
        self.write_result(result_data)

    def _evaluate_ground_truth_variant(
        self,
        gt_snippet: str,
        pred_logs: list,
        ignore_pvs: set,
        is_complex: bool,
        precomputed_logs: list | None = None,
        command: str = "",
    ) -> dict:
        """
        Execute one ground-truth snippet and compute all comparison metrics
        against *pred_logs*. Returns a dict with every piece of data we need
        to rank this variant; precomputed_logs if provided will be used to avoid re-execution.
        """
        # Use pre-executed logs if supplied; otherwise execute now
        if precomputed_logs is not None:
            gt_logs = precomputed_logs
        else:
            try:
                gt_logs = self.execute_snippet_in_mock_env(
                    gt_snippet, is_gt=True, is_complex=is_complex
                )
            except (ExecutionCancellationError, ExecutionFailedError) as exc:
                gt_logs = self._handle_execution_exception(
                    exc,
                    is_ground_truth=True,
                    snippet_code=gt_snippet,
                    command=command,
                )

        temp_pv = "XF:11BM-ES:{LINKAM}:TEMP"
        gt_logs_f   = [l for l in gt_logs  if l[0] not in ignore_pvs and l[0] != temp_pv]
        pred_logs_f = [l for l in pred_logs if l[0] not in ignore_pvs and l[0] != temp_pv]

        pv_match_rate, pv_mismatch_rate, exact_pv_match, pv_value_matches, aligned_gt, aligned_pred = \
            self._compute_pv_match_metrics(gt_logs_f, pred_logs_f)

        num_gt_pv_events = len(gt_logs_f)
        num_pred_pv_events = len(pred_logs_f)
        pv_total_pairs = len(aligned_gt)

        # ------------------------------------------------------------------
        # TIMING  —  now computed even when the PV/value match is not exact.
        # Only those aligned PV-changes that *do* match (name+value) are
        # considered for the regression.
        # ------------------------------------------------------------------
        timing_match   = False
        timing_score   = 0.0
        timing_details = {}

        # Collect timestamps of aligned pairs that are real matches
        matched_gt_ts, matched_pr_ts = [], []
        for g, p in zip(aligned_gt, aligned_pred):
            if g is None or p is None:
                continue
            if not self._log_entries_match(g, p):
                continue       # keep timing independent from the mismatching PVs
            matched_gt_ts.append(g[1])   # timestamp position = index 1
            matched_pr_ts.append(p[1])

        if matched_gt_ts:      # at least one common PV change
            timing_match, timing_score, timing_details = self._timing_match(
                matched_gt_ts, matched_pr_ts
            )
        else:
            timing_details = {
                "reason": "no matching PV/value pairs for timing evaluation",
                "score": timing_score,
            }

        # temperature
        gt_temp   = [(ts, v) for pv, ts, v in gt_logs  if pv == temp_pv]
        pred_temp = [(ts, v) for pv, ts, v in pred_logs if pv == temp_pv]
        temperature_involved = bool(gt_temp) or bool(pred_temp)
        if temperature_involved:
            temp_match, temp_score, temp_details = is_temp_match(
                gt_temp, pred_temp,
                mae_threshold=5.0, final_temp_threshold=5.0
            )
        else:
            temp_match   = True
            temp_score   = 1.0
            temp_details = {"reason": "no temperature PV", "score": 1.0}

        full_match = (exact_pv_match and timing_match and temp_match) \
                     if temperature_involved else (exact_pv_match and timing_match)
        
        # Calculate full continuous score
        if temperature_involved:
            full_score = 0.6 * pv_match_rate + 0.2 * timing_score + 0.2 * temp_score
        else:
            full_score = 0.8 * pv_match_rate + 0.2 * timing_score

        return dict(
            snippet            = gt_snippet,
            gt_logs            = gt_logs,
            pv_match_rate      = pv_match_rate,
            pv_mismatch_rate   = pv_mismatch_rate,
            exact_pv_match     = exact_pv_match,
            pv_value_matches   = pv_value_matches,
            pv_total_pairs     = pv_total_pairs,
            num_gt_pv_events   = num_gt_pv_events,
            num_pred_pv_events = num_pred_pv_events,
            timing_match       = timing_match,
            timing_score       = timing_score,
            temp_match         = temp_match,
            temp_score         = temp_score,
            full_match         = full_match,
            full_score         = full_score,
            aligned_gt         = aligned_gt,
            aligned_pred       = aligned_pred,
            timing_details     = timing_details,
            temp_details       = temp_details,
            gt_temp_logs       = gt_temp,
            pred_temp_logs     = pred_temp,
            temperature_involved = temperature_involved,
        )

    def _construct_logs_tracked_functions(self, aligned_gt, aligned_pred):
        gt_logs_display, pred_logs_display = [], []
        for g, p in zip(aligned_gt, aligned_pred):
            # Ground truth side
            if g is None:
                gt_logs_display.append(None)
            else:
                pv, ts, val = g
                if pv in self.functions_to_log and \
                        not self.functions_to_log[pv].get("compare_params", True):
                    val = "CALLED"
                gt_logs_display.append((pv, ts, val))
            # Predicted side
            if p is None:
                pred_logs_display.append(None)
            else:
                pv, ts, val = p
                if pv in self.functions_to_log and \
                        not self.functions_to_log[pv].get("compare_params", True):
                    val = "CALLED"
                pred_logs_display.append((pv, ts, val))
        return gt_logs_display, pred_logs_display

    def _print_full_score_breakdown(
        self,
        *,
        command: str,
        ground_truth_snippet: str,
        predicted_snippet: str,
        pv_rate: float,
        timing_score: float,
        temp_score: float,
        full_score: float,
        timing_details: dict,
        temp_details: dict,
        temperature_involved: bool,
        gt_temp_logs: list,
        pred_temp_logs: list,
    ) -> None:
        print("\n" + "=" * 120)
        print("COMMAND:", command)
        print("-" * 120)
        print("GROUND-TRUTH SNIPPET:")
        print(ground_truth_snippet)
        print("-" * 120)
        print("PREDICTED SNIPPET:")
        print(predicted_snippet)
        print("-" * 120)
        print("TIMING DETAILS:");  pprint(timing_details)
        print("TEMPERATURE DETAILS:");  pprint(temp_details)
        print("GT temperature logs :", gt_temp_logs)
        print("Pred temperature logs:", pred_temp_logs)
        print("-" * 120)

        # ----- timing-score decomposition ---------------------------------
        if timing_details and not "reason" in timing_details:
            print("\nTIMING SCORE BREAKDOWN:")
            timing_components = [
                ("R² score",        timing_details.get("r2_score"),        0.4),
                ("Slope score",     timing_details.get("slope_score"),     0.2),
                ("Duration score",  timing_details.get("duration_score"),  0.2),
                ("MAPE score",      timing_details.get("mape_score"),      0.2),
            ]
            timing_subtotal = 0.0
            for name, val, weight in timing_components:
                part = (val or 0.0) * weight
                timing_subtotal += part
                if val:
                    print(f"  {name:15s}: {val:6.3f} × {weight:3.1f} = {part:6.3f}")
            print("  " + "-"*38)
            print(f"  Calculated timing_score : {timing_subtotal:6.3f}")
            print(f"  Stored timing_score     : {timing_score:6.3f}")
            print("-"*120)
        else:
            print("No timing calculations:", timing_details)

        # ----- temperature-score decomposition ---------------------------
        if temperature_involved and temp_details and not "reason" in temp_details:
            print("\nTEMPERATURE SCORE BREAKDOWN:")
            temp_components = [
                ("MAE score",          temp_details.get("mae_score"),          0.7),
                ("Final-temp score",   temp_details.get("final_temp_score"),   0.3),
            ]
            temp_subtotal = 0.0
            for name, val, weight in temp_components:
                part = (val or 0.0) * weight
                temp_subtotal += part
                print(f"  {name:15s}: {val:6.3f} × {weight:3.1f} = {part:6.3f}")
            print("  " + "-"*38)
            print(f"  Calculated temp_score   : {temp_subtotal:6.3f}")
            print(f"  Stored temp_score       : {temp_score:6.3f}")
            print("-"*120)
        else:
            print("No temperature calculations:", temp_details)

        print("FULL-SCORE BREAKDOWN:")
        contribs = (
            [("PV match rate", pv_rate, 0.6),
             ("Timing score",  timing_score, 0.2),
             ("Temp score",    temp_score,   0.2)]
            if temperature_involved else
            [("PV match rate", pv_rate, 0.8),
             ("Timing score",  timing_score, 0.2)]
        )
        subtotal = 0.0
        for name, val, w in contribs:
            part = val * w
            subtotal += part
            print(f"  {name:15s}: {val:6.3f} × {w:3.1f} = {part:6.3f}")
        print("  " + "-" * 38)
        print(f"  Calculated total : {subtotal:6.3f}")
        print(f"  Stored full score: {full_score:6.3f}")
        print("=" * 120 + "\n")

    # ------------------------------------------------------------------
    # Pretty‑printer for command + (optionally) both code snippets
    # ------------------------------------------------------------------
    def _print_command_and_snippets(
        self,
        *,
        command: str,
        ground_truth_snippet: str | None = None,
        predicted_snippet: str | None = None,
        header: str | None = None,
        mismatch_reasons: list[str] | None = None,
    ) -> None:
        print("\n" + "*" * 80)
        if header:
            print(f"** {header}")
            print("** " + "-" * 76)

        if mismatch_reasons:
            print("** Mismatch reasons:")
            for reason in mismatch_reasons:
                print(f"**  - {reason}")
            print("** " + "-" * 76)

        print(f"** Command: {command}")

        if ground_truth_snippet is not None:
            print("** " + "-" * 76)
            print("** Ground truth snippet:")
            print("** " + "-" * 76)
            for line in ground_truth_snippet.split("\n"):
                print(f"** {line}")

        if predicted_snippet is not None:
            print("** " + "-" * 76)
            print("** Predicted snippet:")
            print("** " + "-" * 76)
            for line in predicted_snippet.split("\n"):
                print(f"** {line}")

        print("*" * 80 + "\n")

    # ------------------------------------------------------------------
    # Shared failure‑handler for ground‑truth & predicted code execution
    # ------------------------------------------------------------------
    def _handle_execution_exception(
        self,
        exc: Exception,
        *,
        is_ground_truth: bool,
        snippet_code: str,
        command: str,
    ) -> list:
        """
        Uniformly process ExecutionCancellationError / ExecutionFailedError.

        Returns the PV‑change logs extracted from the exception (empty list if
        none present) and prints a formatted error report.
        """
        label = "GROUND TRUTH" if is_ground_truth else "PREDICTED"

        # Pull logs off the exception if available
        logs = getattr(exc, "logs", [])
        if logs:
            print(f"{label} code execution failed but captured {len(logs)} PV changes")
        else:
            print(f"{label} code execution failed with no logs captured")

        self._print_command_and_snippets(
            command=command,
            ground_truth_snippet=snippet_code if is_ground_truth else None,
            predicted_snippet=snippet_code if not is_ground_truth else None,
            header=f"{label} CODE EXECUTION FAILED",
        )

        # ------------------------------------------------------------------
        # After any execution failure, restart the IPython simulator to
        # avoid cascading, non-deterministic errors in subsequent runs.
        # ------------------------------------------------------------------
        try:                       # cleanly close the old shell if it is alive
            if self.interactive_shell is not None and self.interactive_shell.isalive():
                self.interactive_shell.close(force=True)
        except Exception:
            pass                   # ignore problems while shutting down

        print("Restarting IPython environment due to execution failure …")
        self.start_ipython_env()   # fresh shell + new logfile

        return logs

    def calculate_statistics(self):
        stats = {}
        for complexity_type in ['simple', 'complex']:
            results = self.metrics[complexity_type]['results']
            if not results:
                continue

            count = self.metrics[complexity_type]['count']
            correct = self.metrics[complexity_type]['full_matches']

            # PV exact match statistics and new pv_match_rate
            pv_exact_matched       = self.metrics[complexity_type]['pv_exact_match_count']
            pv_exact_mismatched    = self.metrics[complexity_type]['pv_exact_mismatch_count']
            pv_exact_match_rate    = (pv_exact_matched / (pv_exact_matched + pv_exact_mismatched) * 100) \
                                     if (pv_exact_matched + pv_exact_mismatched) else 0
            average_pv_match_rate  = (sum(self.metrics[complexity_type]['pv_match_rates']) /
                                      len(self.metrics[complexity_type]['pv_match_rates']) * 100) \
                                      if self.metrics[complexity_type]['pv_match_rates'] else 0
            average_pv_mismatch_rate = (sum(self.metrics[complexity_type]['pv_mismatch_rates']) /
                                        len(self.metrics[complexity_type]['pv_mismatch_rates']) * 100) \
                                        if self.metrics[complexity_type]['pv_mismatch_rates'] else 0

            # Timing match statistics
            timing_matched = self.metrics[complexity_type]['timing_match_count']
            timing_mismatched = self.metrics[complexity_type]['timing_mismatch_count']

            # Temperature match statistics
            temp_matched = self.metrics[complexity_type].get('temp_match_count', 0)
            temp_mismatched = self.metrics[complexity_type].get('temp_mismatch_count', 0)

            # Full match statistics
            full_matched = self.metrics[complexity_type]['full_match_count']
            full_mismatched = self.metrics[complexity_type]['full_mismatch_count']

            # Exact code match statistics
            exact_code_matches = self.metrics[complexity_type]['exact_code_match_count']
            exact_code_match_rate = (exact_code_matches / count * 100) if count > 0 else 0

            stats[complexity_type] = {
                'count': count,
                'full_matches': correct,
                
                # Exact code match statistics (accuracy without simulator)
                'exact_code_matches': exact_code_matches,
                'exact_code_match_rate': exact_code_match_rate,

                'pv_exact_matched': pv_exact_matched,
                'pv_exact_mismatched': pv_exact_mismatched,
                'pv_exact_match_rate': pv_exact_match_rate,
                'average_pv_match_rate': average_pv_match_rate,
                'average_pv_mismatch_rate': average_pv_mismatch_rate,

                # Timing match statistics
                'timing_matched': timing_matched,
                'timing_mismatched': timing_mismatched,
                'timing_match_rate': (timing_matched / (timing_matched + timing_mismatched) * 100) if (timing_matched + timing_mismatched) > 0 else 0,

                # Temperature match statistics
                'temp_matched': temp_matched,
                'temp_mismatched': temp_mismatched,
                'temp_match_rate': (temp_matched / (temp_matched + temp_mismatched) * 100) if (temp_matched + temp_mismatched) > 0 else 0,

                # Full match statistics
                'full_matched': full_matched,
                'full_mismatched': full_mismatched,
                'full_match_rate': (full_matched / (full_matched + full_mismatched) * 100) if (full_matched + full_mismatched) > 0 else 0,

                'accuracy': (correct / count * 100) if count > 0 else 0,
                'average_timing_score': sum(self.metrics[complexity_type]['timing_scores']) / len(self.metrics[complexity_type]['timing_scores']) if self.metrics[complexity_type]['timing_scores'] else 0,
                'average_temp_score': sum(self.metrics[complexity_type]['temp_scores']) / len(self.metrics[complexity_type]['temp_scores']) if self.metrics[complexity_type]['temp_scores'] else 0,
                'average_full_score': sum(self.metrics[complexity_type]['full_scores']) / len(self.metrics[complexity_type]['full_scores']) if self.metrics[complexity_type]['full_scores'] else 0,
                # ---- average each CodeBLEU component over the entries that
                #      actually contain that component --------------------------
                'average_best_codebleu': {
                    k: (
                        sum(r['codebleu'][k] for r in results if k in r['codebleu'])
                        /
                        sum(1               for r in results if k in r['codebleu'])
                    )
                    for k in sorted({key for r in results for key in r['codebleu']})
                },
                'average_best_levenshtein': sum(r['levenshtein'] for r in results) / len(results),
                'average_best_normalized_levenshtein': sum(r['normalized_levenshtein'] for r in results) / len(results),
                'average_inference_time': sum(self.metrics[complexity_type]['execution_times']) / len(results)
            }
        return stats

    # Custom write method to tee output to both console and log file
    def write(self, text):
        self.original_stdout.write(text)
        self.console_log_file.write(text)
        
    def flush(self):
        self.original_stdout.flush()
        self.console_log_file.flush()
        
    def run_tests(self, run_statistics_path=None):
        try:
            super().run_tests(run_statistics_path)
        finally:
            # Close log files if they exist
            if self.log_file:
                self.log_file.close()
            
            # Restore original stdout and close console log file
            sys.stdout = self.original_stdout
            if hasattr(self, 'console_log_file') and self.console_log_file:
                self.console_log_file.close()
                print(f"Console output has been saved to: {self.console_log_path}")
        
        # Calculate and print statistics
        stats = self.calculate_statistics()
        for complexity_type, metrics in stats.items():
            print(f"\n{complexity_type.title()} Commands Statistics:")
            print(f"Exact code match rate: {metrics['exact_code_match_rate']:.2f}% (accuracy without simulator)")
            print(f"Exact PV match rate (non-temp): {metrics['pv_exact_match_rate']:.2f}%")
            print(f"Average PV match rate (non-temp): {metrics['average_pv_match_rate']:.2f}%")
            print(f"Average PV mismatch rate (non-temp): {metrics['average_pv_mismatch_rate']:.2f}%")
            print(f"Timing match rate: {metrics['timing_match_rate']:.2f}%")
            print(f"Average timing score: {metrics['average_timing_score']:.3f}")
            if metrics['temp_matched'] + metrics['temp_mismatched'] > 0:
                print(f"Temperature match rate: {metrics['temp_match_rate']:.2f}%")
                print(f"Average temperature score: {metrics['average_temp_score']:.3f}")
            print(f"Full match rate: {metrics['full_match_rate']:.2f}%")
            print(f"Average full score: {metrics['average_full_score']:.3f}")
            print(f"Count: {metrics['count']}")
            print(f"Average CodeBLEU Scores: {metrics['average_best_codebleu']}")
            print(f"Average Levenshtein Distance: {metrics['average_best_levenshtein']:.4f}")
            print(f"Average Normalized Levenshtein: {metrics['average_best_normalized_levenshtein']:.4f}")
            print(f"Average Inference Time: {metrics['average_inference_time']:.4f}s")

        if run_statistics_path:
            with open(run_statistics_path, 'r') as f:
                run_statistics = json.load(f)
            run_statistics["metrics_by_complexity"] = stats
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)


if __name__ == '__main__':
    # TODO: make - and _ consistent.
    parser = argparse.ArgumentParser(description='Run OpAgent tests.')
    parser.add_argument('--base_model', type=str, default="claude-3.5-sonnet", help="Base model to use")
    parser.add_argument('--system_prompt_path', type=str, help="Path to the prompt file")
    parser.add_argument('--num_runs', type=int, default=1, help="Number of times to run the experiment")
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction, help="Use ground-truth as predicted code to test various functionalities of the testing code")
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', help="Disable execution caching for non-ground truth elements")
    parser.add_argument('--no-gt-cache', dest='use_gt_cache', action='store_false', help="Disable execution caching for ground truth logs")
    parser.add_argument('--clear-cache', action='store_true', help="Clear the execution cache before running")
    parser.add_argument('--complex_only', action='store_true', help="Only run tests on complex examples (is_complex=True)")
    parser.add_argument('--simple_only', action='store_true', help="Only run tests on simple examples (is_complex=False)")
    parser.add_argument('--dataset_path', type=str,
                        help="Path to the test-dataset JSON file")
    parser.add_argument('--scientist_file', type=str,
                        help='TOML file that maps each command to the scientists’ code')
    parser.add_argument(
        '--beamline',
        type=str,
        default='11BM',
        help='Beamline identifier used for the tests (default: 11BM)'
    )
    parser.set_defaults(use_cache=True, use_gt_cache=True, complex_only=False, simple_only=False)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Use user-supplied dataset if given, otherwise default one
    if args.dataset_path:
        dataset_file_path = args.dataset_path
    else:
        dataset_file_path = os.path.join(base_dir, 'datasets', 'op_cog_dataset.json')

    # Debug
    if args.debug:
        args.base_model = "debug"
    
    # Create a parent timestamp directory for all runs
    parent_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(base_dir, 'results', 'op_cog', args.base_model, parent_timestamp)
    os.makedirs(base_results_dir, exist_ok=True)

    # Clear cache if requested
    if args.clear_cache:
        cache_path = os.path.join(
            base_dir,
            "results",
            "op_cog",
            "execution_cache.json"
        )
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Cleared execution cache: {cache_path}")
        else:
            print("No cache file found to clear")
    
    # Run multiple experiments
    for run in range(args.num_runs):
        run_dir = os.path.join(base_results_dir, f'run_{run}')
        results_file_path = os.path.join(run_dir, 'results_op_agent.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        tester = OpAgentTestFramework(
            dataset_file_path,
            results_file_path,
            base_model=args.base_model,
            system_prompt_path=args.system_prompt_path,
            num_runs=args.num_runs,
            debug=args.debug,
            use_gt_cache=args.use_gt_cache,
            use_cache=args.use_cache,
            complex_only=args.complex_only,
            simple_only=args.simple_only,
            scientist_file=args.scientist_file,
            beamline=args.beamline,
        )

        run_statistics_path = os.path.join(run_dir, 'run_statistics.json')
        tester.run_tests(run_statistics_path)

    # Aggregate results after all runs are complete
    tester.aggregate_run_statistics(base_results_dir)

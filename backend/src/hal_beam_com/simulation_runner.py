import os
import time
import pexpect
import re
import epics
from typing import Callable, Optional, Dict
from src.hal_beam_com.pv_config import get_pv_config
from epics import caput_many

class SimulationRunner:
    """
    Headless simulator for running code in an IPython/Bluesky session and capturing PV events.
    """

    # --- helper to strip colours / cursor codes etc. -----------------
    _ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    @staticmethod
    def _clean_output(txt: str) -> str:
        """
        Make shell output frontend-friendly:
          – turn CR into LF (keeps progress-bar lines)
          – drop all ANSI/VT100 escape sequences
        """
        txt = txt.replace('\r', '\n')
        return SimulationRunner._ANSI_ESCAPE_RE.sub('', txt)

    IPYTHON_PROMPT = r'In\s*\[[0-9]+\]:\s*'

    def __init__(self,
                 event_callback: Optional[Callable[[dict], None]] = None,
                 pv_defaults: Optional[Dict[str, float]] = None):
        # One-off IPython kernel per runner instance.
        self._setup_ipython_shell()
        # Load PV list and their baseline defaults from shared config
        self.pvs, self._pvs_defaults = get_pv_config()
        if pv_defaults is not None:
            self._pvs_defaults = pv_defaults
        # Setup monitoring callback logs
        self.monitor_logs = []
        self._recording = False
        self._event_callback = event_callback
        self._abort_requested = False
        self._environment_initialized = False  # Track if environment is initialized
        self._start_time = None  # Track when recording started
        self._last_pv_time = None  # Track time of last PV event for delta calculation
        
        # Set up monitors once at initialization, like test_op_cog.py does
        self._setup_monitors()

    def abort(self) -> None:
        """External request to stop the current run immediately."""
        self._abort_requested = True
        try:
            # send two ^C to IPython, then Bluesky abort
            self.interactive_shell.sendintr()
            time.sleep(0.2)
            self.interactive_shell.sendintr()
            time.sleep(0.2)
            self.interactive_shell.sendline("RE.abort()")
        except Exception as exc:
            print(f"[sim] abort exception: {exc}")
        # notify listeners
        if self._event_callback:
            self._event_callback({"type": "status", "state": "aborting"})

    def set_pv_defaults(self, pv_defaults: Dict[str, float]) -> None:
        """Replace the baseline PV defaults used by _reset_pvs()."""
        self._pvs_defaults = pv_defaults

    def _reset_pvs(self) -> None:
        """
        Quickly bring all tracked PVs back to their baseline values.

        We use a *non-blocking* caput on each PV so that channels that are
        disconnected do not introduce long timeouts.
        """
        pvs = list(self._pvs_defaults.keys())
        values = list(self._pvs_defaults.values())

        # 1st try – blocking put (wait until all PVs report success)
        results = caput_many(
            pvs, values,
            wait="all",
            connection_timeout=5,
            put_timeout=1,
        )

        # Collect PVs that did NOT complete (caput_many returns -1)
        failed_pvs = [pv for pv, r in zip(pvs, results) if r == -1]

        if failed_pvs:
            print(f"Warning: {len(failed_pvs)} PV resets timed-out – "
                  "retrying with instant motor-teleport.")

            # ---------- classify failures ---------------------------------
            motor_pvs = [pv for pv in failed_pvs if pv.endswith("Mtr")]
            other_pvs = [pv for pv in failed_pvs if pv not in motor_pvs]

            # ---------- instant reset for motors --------------------------
            if motor_pvs:
                motor_vals = [self._pvs_defaults[pv] for pv in motor_pvs]
                set_fields = [pv + ".SET" for pv in motor_pvs]
                val_fields = [pv + ".VAL" for pv in motor_pvs]

                # 1) enter SET-mode
                caput_many(set_fields, [1] * len(set_fields),
                           wait="all", connection_timeout=5, put_timeout=60)
                # 2) override VAL (teleport)
                caput_many(val_fields, motor_vals, wait=False)
                # 3) leave SET-mode
                caput_many(set_fields, [0] * len(set_fields), wait=False)

            # ---------- non-motor PVs: simple fire-and-forget --------------
            if other_pvs:
                other_vals = [self._pvs_defaults[pv] for pv in other_pvs]
                caput_many(other_pvs, other_vals, wait=False)

        # short pause to allow network traffic to settle
        time.sleep(2)

    def _setup_ipython_shell(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
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
        self.interactive_shell = pexpect.spawn('/bin/bash', encoding='utf-8', timeout=30)
        self.interactive_shell.sendline(cmd)
        self.interactive_shell.expect('In \\[[0-9]+\\]: ', timeout=300)
        self.interactive_shell.sendline('%run -i ./.ci/linkam-drop-in.py')
        self.interactive_shell.expect('In \\[[0-9]+\\]: ', timeout=30)

    def _initialize_environment(self):
        """Initialize the testing environment with required objects and settings."""
        print("[sim] Initializing sample...")
        patterns = [self.IPYTHON_PROMPT, 'Traceback.*\\n', '.*Error.*\\n']
        
        # Initialize sample object
        self.interactive_shell.sendline("sam = SampleLinkamTensile('test')")
        
        try:
            index = self.interactive_shell.expect(patterns, timeout=30)
            if index != 0:  # Not a prompt, got an error
                error_msg = self.interactive_shell.before + self.interactive_shell.after
                print(f"[sim] Error initializing sample: {error_msg}")
                raise Exception(f"Failed to initialize sample: {error_msg}")
            print("[sim] Sample initialized")
            self._environment_initialized = True
        except pexpect.TIMEOUT:
            print("[sim] Timeout initializing sample")
            raise

    def _monitor_callback(self, pvname=None, value=None, char_value=None, timestamp=None, **kwargs):
        if not self._recording:
            return
        
        current_time = time.time()
        
        # Calculate elapsed time from start
        elapsed_time = current_time - self._start_time if self._start_time else 0
        
        # Calculate delta time from last PV event
        delta_time = (current_time - self._last_pv_time) if self._last_pv_time else 0
        
        # Update last PV time
        self._last_pv_time = current_time
        
        evt = {
            "type": "pv",
            "pvname": pvname,
            "value": value,
            "timestamp": timestamp or current_time,
            "elapsed_time": elapsed_time,  # Time since start of recording
            "delta_time": delta_time,      # Time since last PV event
        }
        self.monitor_logs.append(evt)
        if self._event_callback:
            self._event_callback(evt)

    def _setup_monitors(self) -> None:
        """
        Setup monitors using camonitor like test_op_cog.py does.
        This is called once at initialization and monitors stay active.
        """
        self.monitor_logs = []
        for pv in self.pvs:
            epics.camonitor(pv, callback=self._monitor_callback)
        
        # Give monitors time to establish
        time.sleep(1)

    def prepare_for_reuse(self) -> None:
        """Prepare the simulator for reuse after a successful run."""
        self._abort_requested = False
        self.monitor_logs = []
        self._recording = False
        self._start_time = None
        self._last_pv_time = None
        
        # Don't re-initialize environment here - it will be done in run_code_sync if needed
        # Just ensure the flag is set correctly
        # self._environment_initialized should remain True if we successfully ran before
        
        # Clear any pending monitor callbacks by polling
        epics.ca.poll(evt=0.05, iot=1.0)
        time.sleep(0.05)
        epics.ca.poll(evt=0.05, iot=1.0)

    def is_reusable(self) -> bool:
        """Check if the simulator can be reused (shell is still responsive)."""
        if self.interactive_shell is None:
            return False
        
        try:
            # Send a simple command to check if shell is responsive
            self.interactive_shell.sendline("print('alive')")
            index = self.interactive_shell.expect([self.IPYTHON_PROMPT, pexpect.TIMEOUT], timeout=5)
            return index == 0
        except:
            return False

    def run_code_sync(self, code):
        """
        Run the provided code in the simulation shell, capture PV events and errors, and return a list of event dicts.
        This is a blocking method.
        """
        self._reset_pvs()
        
        # Clear monitor logs for this run
        self.monitor_logs = []
        
        # Poll to get baseline readings and clear them
        epics.ca.poll(evt=0.05, iot=1.0)
        time.sleep(0.1)
        epics.ca.poll(evt=0.05, iot=1.0)
        
        # Clear logs again to discard baseline readings
        self.monitor_logs = []
        events = []
        
        # Only initialize environment if not already done
        if not self._environment_initialized:
            self._initialize_environment()
        else:
            # If already initialized, just make sure we're at a clean prompt
            # by consuming any pending output
            try:
                self.interactive_shell.expect(self.IPYTHON_PROMPT, timeout=0.5)
            except pexpect.TIMEOUT:
                pass
        
        try:
            time.sleep(0.5)  # Let system stabilize
            self._start_time = time.time()  # Record start time
            self._last_pv_time = 0  # Initialize to start time
            self._recording = True
            self.interactive_shell.sendline("%cpaste -q")
            self.interactive_shell.sendline(code)
            self.interactive_shell.sendline("--")
            self.interactive_shell.sendline('')  # Extra newline
            patterns = [self.IPYTHON_PROMPT, 'Traceback.*\\n', '.*Error.*\\n']
            while True:
                if self._abort_requested:
                    err_evt = {"type": "error", "message": "Simulation aborted by user"}
                    events.append(err_evt)
                    if self._event_callback:
                        self._event_callback(err_evt)
                    break
                try:
                    index = self.interactive_shell.expect(patterns, timeout=1)
                    if index == 0:                         # prompt
                        break
                    elif index in (1, 2):                  # error
                        error_message_raw = self.interactive_shell.before + self.interactive_shell.after
                        error_message = self._clean_output(error_message_raw).strip()
                        err_evt = {"type": "error", "message": error_message}
                        events.append(err_evt)
                        if self._event_callback:
                            self._event_callback(err_evt)
                        break
                except pexpect.TIMEOUT:
                    # During execution, allow more polling to capture PV changes
                    epics.ca.poll(evt=0.01, iot=0.1)
                    continue
        except pexpect.TIMEOUT:
            error_message_raw = self.interactive_shell.before + self.interactive_shell.after
            error_message     = self._clean_output(error_message_raw).strip()
            err_evt = {
                "type": "error",
                "message": error_message
            }
            events.append(err_evt)
            if self._event_callback:
                self._event_callback(err_evt)
        
        # Allow pending monitor updates to fire
        epics.ca.poll(evt=0.05, iot=1.0)
        time.sleep(0.20)
        epics.ca.poll(evt=0.05, iot=1.0)
        
        # Stop recording before collecting logs
        self._recording = False
        
        # Collect monitor logs
        events.extend(self.monitor_logs)
        
        return events

    async def run_code(self, code):
        """
        Async generator for streaming simulation output events (pv / error / status).
        """
        import asyncio
        self._reset_pvs()
        
        # Clear monitor logs for this run
        self.monitor_logs = []
        
        # Clear baseline callbacks
        await asyncio.to_thread(epics.ca.poll, 0.05, 1.0)
        await asyncio.sleep(0.1)
        await asyncio.to_thread(epics.ca.poll, 0.05, 1.0)
        
        # Clear logs again to discard baseline readings
        self.monitor_logs = []
        
        self._start_time = time.time()  # Record start time
        self._last_pv_time = 0 # Initialize to start time
        self._recording = True
        
        # Only initialize environment if not already done
        if not self._environment_initialized:
            await asyncio.to_thread(self._initialize_environment)
        else:
            # If already initialized, just make sure we're at a clean prompt
            try:
                await asyncio.to_thread(self.interactive_shell.expect, self.IPYTHON_PROMPT, 0.5)
            except pexpect.TIMEOUT:
                pass
        
        # start run
        try:
            self.interactive_shell.sendline("%cpaste -q")
            self.interactive_shell.sendline(code)
            self.interactive_shell.sendline("--")
            self.interactive_shell.sendline('')
            patterns = [
                self.IPYTHON_PROMPT,
                'Traceback.*\\n',
                '.*Error.*\\n'
            ]
            while True:
                if self._abort_requested:
                    event_dict = {"type": "error", "message": "Simulation aborted by user"}
                    yield event_dict
                    if self._event_callback:
                        self._event_callback(event_dict)
                    break
                try:
                    index = await asyncio.to_thread(self.interactive_shell.expect, patterns, 1)
                    if index == 0:
                        break
                    elif index in [1,2]:
                        error_message_raw = self.interactive_shell.before + self.interactive_shell.after
                        error_message     = self._clean_output(error_message_raw).strip()
                        event_dict = {"type": "error", "message": error_message}
                        yield event_dict
                        if self._event_callback:
                            self._event_callback(event_dict)
                        break
                except pexpect.TIMEOUT:
                    # During execution, allow more polling to capture PV changes
                    await asyncio.to_thread(epics.ca.poll, 0.01, 0.1)
                    continue
        except pexpect.TIMEOUT:
            error_message_raw = self.interactive_shell.before + self.interactive_shell.after
            error_message     = self._clean_output(error_message_raw).strip()
            event_dict = {"type": "error", "message": error_message}
            yield event_dict
            if self._event_callback:
                self._event_callback(event_dict)
        
        # allow pending monitor updates to fire
        await asyncio.to_thread(epics.ca.poll, 0.05, 1.0)
        await asyncio.sleep(0.20)
        await asyncio.to_thread(epics.ca.poll, 0.05, 1.0)
        
        self._recording = False
        
        # Yield all collected monitor logs
        for evt in self.monitor_logs:
            yield evt

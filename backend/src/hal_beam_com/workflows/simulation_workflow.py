import time, uuid, copy, json, pprint
from typing import List
from threading import Thread
from src.hal_beam_com.simulation_runner import SimulationRunner
from src.hal_beam_com.utils import CogType, cog_output_fields
from src.hal_beam_com.pv_config import get_pv_config
from typing import Dict

# Store simulation events by sim_id for later retrieval during evaluation
_simulation_events: Dict[str, List[Dict]] = {}


def add_rbv_suffix(pv_names: List[str]) -> List[str]:
    """
    Add .RBV suffix to PV names for archiver lookup.
    
    Args:
        pv_names: List of base PV names
        
    Returns:
        List of PV names with .RBV suffix appended
    """
    return [f"{pv}.RBV" for pv in pv_names]


def strip_rbv_suffix(pv_defaults: Dict[str, float]) -> Dict[str, float]:
    """
    Strip .RBV suffix from PV names in the defaults dictionary.
    
    Args:
        pv_defaults: Dictionary with PV names (potentially with .RBV suffix) as keys
        
    Returns:
        Dictionary with .RBV suffix removed from keys
    """
    if pv_defaults is None:
        return None
    
    return {
        pv.removesuffix(".RBV"): value
        for pv, value in pv_defaults.items()
    }


def get_simulation_events(sim_id: str) -> List[Dict]:
    """
    Retrieve stored simulation events by sim_id.
    Returns an empty list if sim_id not found.
    """
    return _simulation_events.get(sim_id, [])


def clear_simulation_events(sim_id: str = None) -> None:
    """
    Clear stored simulation events.
    If sim_id is provided, clears only that simulation.
    If sim_id is None, clears all stored events.
    """
    if sim_id:
        _simulation_events.pop(sim_id, None)
    else:
        _simulation_events.clear()


# ----------------------------------------------------------------------
# PV-DEFAULTS HANDSHAKE (backend ↔ frontend)
#
# • If the incoming message sets  use_archiver_defaults=True  but does NOT
#   already contain a  "pv_defaults"  dict,  the backend must *not* try to
#   contact the archiver.  Instead, it publishes a *request* asking the UI
#   to return the current values for the PVs it will track:
#
#        {
#          "type"     : "pv_defaults_request",
#          "beamline" : "<beamline>",
#          "pv_names" : [ "...", ... ]          # list from get_pv_config
#        }
#
#   and then RETURNS immediately (status field is set to
#   "awaiting_pv_defaults").  No simulation is started.
#
# • The frontend queries the archiver and calls the same endpoint again,
#   *this time* passing a   "pv_defaults": {pv: value, …}   dict.
#
# • When  pv_defaults  is present (regardless of use_archiver_defaults),
#   the backend proceeds with the simulation exactly as before, passing
#   those defaults to  SimulationRunner.
# ----------------------------------------------------------------------

def run(data: List, queue, **_):
    """
    Expected incoming `data`:
        data[0]['code_to_simulate']   – python string (required)
        data[0]['beamline']           – e.g. '11BM'      (already provided elsewhere)
    Generates OUTGOING updates (see message to frontend dev at the end).
    """
    # initialise once
    if not hasattr(run, "_active_runner"):
        run._active_runner = None

    msg_type = data[0].get("bl_input_channel")
    if msg_type == "simulate_abort":
        sim_id_abort = data[0].get("sim_id")
        if run._active_runner is None:
            print(f"[SIM] Abort requested for sim_id={sim_id_abort} but no simulation is running")
            return data

        current_id = getattr(run._active_runner, "sim_id", None)
        if sim_id_abort is None or sim_id_abort == current_id:
            print(f"[SIM] Abort requested → stopping active simulation (sim_id={current_id})")
            run._active_runner.abort()
            run._active_runner = None
        else:
            print(f"[SIM] Abort requested for sim_id={sim_id_abort} "
                  f"but active simulation has sim_id={current_id} – ignoring")
        return data

    use_archiver_defaults = data[0].get("use_archiver_defaults", False)
    pv_defaults           = data[0].get("pv_defaults")          # may be None

    # --------------------------------------------------------------
    # Fallback:  when the caller does NOT request archiver defaults
    # we must still have some initial values → use the built-in
    # defaults from pv_config.
    # --------------------------------------------------------------
    if (not use_archiver_defaults) and (pv_defaults is None):
        _, pv_defaults = get_pv_config(data[0]["beamline"])

    # ------------------------------------------------------------------
    # If the UI asked for archiver defaults but has not supplied them
    # yet, send it the PV list we need and exit.
    # ------------------------------------------------------------------
    if use_archiver_defaults and pv_defaults is None:
        # --------------------------------------------------------------
        # Create (or recycle) an **idle** SimulationRunner so that
        # the expensive IPython start-up happens BEFORE the UI comes
        # back with the pv_defaults.  No code is executed yet.
        # --------------------------------------------------------------
        if run._active_runner is None or not run._active_runner.is_reusable():
            print("[SIM] Pre-initialising simulator while waiting for pv_defaults")
            # dummy callback – will be replaced on the 2nd call
            run._active_runner = SimulationRunner(event_callback=lambda *_: None,
                                                  pv_defaults=None)
        else:
            # Shell already running – just reset it for reuse
            run._active_runner.prepare_for_reuse()

        tracked_pvs, _ = get_pv_config(data[0]["beamline"])
        request_msg = {
            "type":     "pv_defaults_request",
            "beamline": data[0]["beamline"],
            "pv_names": add_rbv_suffix(tracked_pvs),  # Add .RBV for archiver lookup
        }
        queue.publish([request_msg])
        print("[SIM-STATUS]", request_msg)
        data[0]["status"] = "awaiting_pv_defaults"
        return data

    sim_id = str(uuid.uuid4())           # unique run identifier
    code   = data[0]['code_to_simulate']

    # Strip .RBV suffix from pv_defaults if present (frontend sends RBV names from archiver)
    pv_defaults = strip_rbv_suffix(pv_defaults)

    # callback that streams every event as soon as it happens
    def _forward(evt, _id=sim_id):
        evt["sim_id"] = _id
        # Store the event for later retrieval
        if _id not in _simulation_events:
            _simulation_events[_id] = []
        _simulation_events[_id].append(evt)
        queue.publish([evt])
        print("[SIM-EVENT]", evt)

    # Check if we can reuse the existing runner
    if run._active_runner:
        reusable = run._active_runner.is_reusable()
        print(f"[SIM] Current SimulationRunner is_reusable(): {reusable} for sim_id={sim_id}")
    if run._active_runner and run._active_runner.is_reusable():
        print(f"[SIM] Reusing existing simulator for sim_id={sim_id}")
        runner = run._active_runner
        runner.sim_id = sim_id
        runner._event_callback = _forward
        if pv_defaults is not None:
            print(f"[SIM-TIMING] Timing set_pv_defaults()...")
            t0 = time.perf_counter()
            runner.set_pv_defaults(pv_defaults)
            t1 = time.perf_counter()
            print(f"[SIM-TIMING] set_pv_defaults() took {t1 - t0:.2f} seconds.")
        runner.prepare_for_reuse()
    else:
        # Create new runner if needed
        print(f"[SIM] Creating new simulator for sim_id={sim_id}")
        runner = SimulationRunner(event_callback=_forward, pv_defaults=pv_defaults)
        runner.sim_id = sim_id
        run._active_runner = runner

    start_time = time.time()
    status_running = {"sim_id": sim_id, "type": "status", "state": "running"}
    queue.publish([status_running])
    print("[SIM-STATUS]", status_running)

    def _run():
        error_occurred = False
        try:
            events = runner.run_code_sync(code)          # ← blocking call
            
            # Check if any error events occurred
            error_occurred = any(evt.get("type") == "error" for evt in events)
            
        except Exception as e:
            print(f"[SIM] Exception during execution: {e}")
            error_occurred = True
            
        finally:
            final_state = "aborted" if runner._abort_requested else "completed"
            status_done = {
                "sim_id":   sim_id,
                "type":     "status",
                "state":    final_state,
                "duration": time.time() - start_time,
                "terminate": 1,                 # tell UI the run is over
            }
            queue.publish([status_done])
            print("[SIM-STATUS]", status_done)
            
            # Only clear the runner if there was an error or abort
            if error_occurred or runner._abort_requested:
                print(f"[SIM] Clearing runner due to {'abort' if runner._abort_requested else 'error'}")
                run._active_runner = None
            else:
                print(f"[SIM] Keeping runner alive for reuse")

    Thread(target=_run, daemon=True).start()    # fire-and-forget

    data[0]['status'] = "running"               # no 'terminate' here
    return data                                 # ← return immediately

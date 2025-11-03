"""
A simulated Linkam temperature controller IOC.
"""

import asyncio
import sys
import time
import threading
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run

class LinkamIOC(PVGroup):
    """
    Simulated Lakeshore IOC with put completion on the setpoint.
    """

    """A simulated Linkam temperature controller"""
    
    # Temperature control
    temperature_setpoint = pvproperty(value=25.0, name="SETPOINT:SET", doc="Temperature setpoint")
    temperature_current = pvproperty(value=25.0, name="TEMP", doc="Current temperature")

    temperature_rate = pvproperty(value=10.0, name="RAMPRATE:SET", doc="Temperature ramp rate")
    
    # Status
    status = pvproperty(value=0, name="STATUS", doc="Device status", read_only=True)
    heater = pvproperty(value=0, name="STARTHEAT", doc="Heater on/off")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_temp = 25.0
        self._target_temp = 25.0
        self._ramp_rate = 10.0
        self._heating = False
        self._poller_thread = None
        self._stop_poller = threading.Event()
        
        # Create aliases for both PV name formats (with and without braces)
        formatted_pvs = {}
        for k, v in list(self.pvdb.items()):
            # Add version with braces
            formatted_key = k.replace('LINKAM:', '{LINKAM}:')
            formatted_pvs[formatted_key] = v
            # Keep original version without braces
            formatted_pvs[k] = v
        self.pvdb = formatted_pvs
        
        print(f"[LinkamIOC] Initialized with prefix: {self.prefix}")
        print(f"[LinkamIOC] PVs registered: {list(self.pvdb.keys())}")
        
    def _poller_loop(self):
        """Update temperature based on setpoint and ramp rate"""
        print("[LinkamIOC] device_poller started.")
        iteration = 0
        
        while not self._stop_poller.is_set():
            try:
                iteration += 1
                print(f"[LinkamIOC] Poller iteration {iteration} -> _heating: {self._heating}, "
                      f"_current_temp: {self._current_temp}, _target_temp: {self._target_temp}")
                
                if self._heating:
                    diff = self._target_temp - self._current_temp
                    if abs(diff) > 0.1:
                        # Calculate the step and update temperature
                        step = min(abs(diff), self._ramp_rate * 0.1) * (1 if diff > 0 else -1)
                        self._current_temp += step
                        print(f"[LinkamIOC] Heating: _current_temp updated to {self._current_temp:.2f} "
                              f"(target: {self._target_temp}, diff: {diff:.2f})")
                        
                        # Update PVs through their write methods
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.temperature_current.write(self._current_temp))
                        
                        # Set status: 2 when heating, 1 when at setpoint
                        status_val = 1 if abs(diff) < 0.5 else 2
                        loop.run_until_complete(self.status.write(status_val))
                        loop.close()
                        print(f"[LinkamIOC] Status set to {status_val}")
                    else:
                        print(f"[LinkamIOC] At target temperature with heater on (diff: {diff:.2f}).")
                        # Update status to 1 when stable at target
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.status.write(1))
                        loop.close()
                        print("[LinkamIOC] Status set to 1 (at setpoint)")
                else:
                    print("[LinkamIOC] Heater is off; skipping temperature update.")
                
                time.sleep(1.0)
            except Exception as e:
                print(f"[LinkamIOC] Error in poller: {e}")
                time.sleep(1.0)  # Avoid tight error loop

    @heater.putter
    async def heater(self, instance, value):
        """Handle heater on/off"""
        self._heating = bool(value)
        print(f"[LinkamIOC] Heater putter called: setting _heating to {self._heating}")
        return value

    @temperature_setpoint.putter
    async def temperature_setpoint(self, instance, value):
        """Handle new temperature setpoint"""
        self._target_temp = float(value)
        print(f"[LinkamIOC] Temperature setpoint putter called: new target set to {self._target_temp}")
        return value

    @temperature_rate.putter
    async def temperature_rate(self, instance, value):
        """Handle new ramp rate"""
        self._ramp_rate = float(value)
        print(f"[LinkamIOC] Temperature rate putter called: new ramp rate set to {self._ramp_rate}")
        return value

    @temperature_current.putter
    async def temperature_current(self, instance, value):
        """Handle direct temperature setting. 
        WARNING: This putter should only be used for debugging/testing purposes.
        In normal operation, the temperature should be controlled via setpoint and 
        heater settings, allowing the simulator to adjust temperature according to 
        ramp rate."""
        self._current_temp = float(value)
        print(f"[LinkamIOC] Temperature directly set to {self._current_temp} (DEBUG/TEST MODE)")
        return value

    async def shutdown(self):
        """Shutdown the poller thread"""
        print("[LinkamIOC] shutdown called; stopping poller thread.")
        if self._poller_thread is not None:
            self._stop_poller.set()
            self._poller_thread.join(timeout=2.0)
            self._poller_thread = None

    async def startup(self):
        print("[LinkamIOC] startup called: scheduling device poller.")
        try:
            # Create but don't start the poller yet
            self._device_poller_task = None
            # Wait for first iteration to complete
            await asyncio.sleep(0.1)
            print("[LinkamIOC] Device poller ready")
        except Exception as e:
            print(f"[LinkamIOC] Error in startup: {e}")

    def start_poller(self):
        """Start the device poller thread"""
        if self._poller_thread is None:
            self._stop_poller.clear()
            self._poller_thread = threading.Thread(target=self._poller_loop, daemon=True)
            self._poller_thread.start()
            print("[LinkamIOC] Device poller thread started")


if __name__ == "__main__":
    ioc_options, run_options = ioc_arg_parser(
        default_prefix="XF:11BM-ES:{LINKAM}:", desc="Linkam Temperature Controller Simulator"
    )
    ioc = LinkamIOC(**ioc_options)
    try:
        run(ioc.pvdb, **run_options)
    finally:
        # On shutdown, cancel the pending device_poller task
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ioc.shutdown())

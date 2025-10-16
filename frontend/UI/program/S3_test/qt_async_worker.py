#!/usr/bin/python3
"""
Qt-compatible async worker for non-blocking HAL communication.
This module provides a QThread-based solution for running async operations
without blocking the UI thread.
"""

import asyncio
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication
import traceback

class QtAsyncWorker(QObject):
    """
    Qt-compatible worker that runs async operations in a separate thread.
    Uses signals to communicate results back to the main thread.
    """
    
    # Signals for communicating with the main thread
    result_ready = pyqtSignal(object)  # Emitted when operation completes successfully
    error_occurred = pyqtSignal(str)   # Emitted when an error occurs
    finished = pyqtSignal()            # Emitted when the worker is done
    
    def __init__(self):
        super().__init__()
        self._thread = None
        self._loop = None
        self._running = False
        self._loop_ready = threading.Event()  # Event to signal when loop is ready
        self._current_task = None
    
    def start(self):
        """Start the worker thread with its own event loop."""
        if self._running:
            return
        
        self._running = True
        self._loop_ready.clear()  # Reset the ready event
        self._thread = QThread()

        # Move the worker to the thread AFTER connecting signals
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run_loop)
        self._thread.start()
        
        # Wait for the event loop to be ready (with timeout)
        if not self._loop_ready.wait(timeout=2.0):
            raise RuntimeError("Failed to start worker thread event loop within timeout")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.quit()
            self._thread.wait(1000)  # Wait up to 1 second
            if self._thread.isRunning():
                self._thread.terminate()
                self._thread.wait()
    
    def _run_loop(self):
        """Run the event loop in the worker thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Signal that the loop is ready
        self._loop_ready.set()
        
        try:
            self._loop.run_forever()
        except Exception as e:
            self.error_occurred.emit(f"Event loop error: {str(e)}")
        finally:
            self._loop.close()
            self.finished.emit()
    
    def run_coroutine(self, coro):
        """
        Run a coroutine in the worker thread.
        
        Args:
            coro: The coroutine to run
        """
        try:
            # Ensure the worker is started and loop is ready
            if not self._running or not self._loop:
                self.start()
            
            # Double-check that loop is ready
            if not self._loop:
                raise RuntimeError("Event loop not initialized")
            
            async def _run_with_signals():
                try:
                    print(f"[QT_WORKER] Starting async operation")
                    result = await coro
                    print(f"[QT_WORKER] Async operation completed with result: {result}")
                    # Emit result signal (will be handled in main thread)
                    print(f"[QT_WORKER] Emitting result signal")
                    # Use moveToThread to ensure signal is emitted from the correct thread
                    self.result_ready.emit(result)
                    print(f"[QT_WORKER] Result signal emitted")
                except Exception as e:
                    print(f"[QT_WORKER] Async operation failed with error: {e}")
                    # Emit error signal (will be handled in main thread)
                    error_msg = f"Async operation error: {str(e)}\n{traceback.format_exc()}"
                    print(f"[QT_WORKER] Emitting error signal")
                    self.error_occurred.emit(error_msg)
                    print(f"[QT_WORKER] Error signal emitted")

            # Schedule the coroutine in the worker thread's event loop
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_run_with_signals())
            )
        except Exception as e:
            # If we can't even schedule the coroutine, emit an error
            error_msg = f"Failed to schedule coroutine: {str(e)}"
            self.error_occurred.emit(error_msg)

class QtAsyncHelper:
    """
    Helper class for managing Qt async operations.
    Provides a simple interface for running async operations with callbacks.
    """
    
    def __init__(self):
        self._worker = QtAsyncWorker()
        self._current_success_callback = None
        self._current_error_callback = None
        self._operation_in_progress = False
        self._operation_queue = []
        self._current_operation_id = 0

        # Connect signals once during initialization
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)

    def run_async(self, coro, success_callback=None, error_callback=None):
        """
        Run an async operation with callbacks.
        
        Args:
            coro: The coroutine to run
            success_callback: Function to call when operation succeeds
            error_callback: Function to call when operation fails
        """
        print(f"[QT_HELPER] run_async called")

        # Generate unique operation ID
        operation_id = self._current_operation_id + 1
        self._current_operation_id = operation_id

        # If an operation is in progress, queue this one
        if self._operation_in_progress:
            print(f"[QT_HELPER] Operation in progress, queuing this operation (ID: {operation_id})")
            self._operation_queue.append((operation_id, coro, success_callback, error_callback))
            return

        # Start the operation
        self._start_operation(operation_id, coro, success_callback, error_callback)

    def _start_operation(self, operation_id, coro, success_callback=None, error_callback=None):
        """Start a new async operation."""
        print(f"[QT_HELPER] Starting operation ID: {operation_id}")

        # Store current callbacks with operation ID
        print(f"[QT_HELPER] Storing callbacks for operation {operation_id} - success: {success_callback is not None}, error: {error_callback is not None}")
        self._current_success_callback = success_callback
        self._current_error_callback = error_callback
        self._operation_in_progress = True
        print(f"[QT_HELPER] Callbacks stored for operation {operation_id} - success: {self._current_success_callback is not None}, error: {self._current_error_callback is not None}")
        
        # Run the coroutine
        print(f"[QT_HELPER] Running coroutine for operation {operation_id}")
        self._worker.run_coroutine(coro)
        print(f"[QT_HELPER] run_async completed for operation {operation_id}")

    def _process_next_operation(self):
        """Process the next operation in the queue."""
        if self._operation_queue:
            print(f"[QT_HELPER] Processing next operation from queue")
            operation_id, coro, success_callback, error_callback = self._operation_queue.pop(0)
            self._start_operation(operation_id, coro, success_callback, error_callback)

    def _on_result(self, result):
        """Handle result from worker."""
        print(f"[QT_HELPER] _on_result called with result: {result}")
        print(f"[QT_HELPER] Current success callback: {self._current_success_callback is not None}")
        if self._current_success_callback:
            print(f"[QT_HELPER] Calling success callback")
            try:
                self._current_success_callback(result)
                print(f"[QT_HELPER] Success callback completed")
            except Exception as e:
                print(f"[QT_HELPER] Error in success callback: {e}")
        else:
            print(f"[QT_HELPER] No success callback available")
        # Clear callbacks after use (signals will be reconnected for next operation)
        print(f"[QT_HELPER] Clearing callbacks after successful operation")
        self._current_success_callback = None
        self._current_error_callback = None
        self._operation_in_progress = False
        print(f"[QT_HELPER] Callbacks cleared - operation_in_progress: {self._operation_in_progress}")

        # Process next operation if any
        self._process_next_operation()

    def _on_error(self, error_msg):
        """Handle error from worker."""
        print(f"[QT_HELPER] _on_error called with error: {error_msg}")
        if self._current_error_callback:
            print(f"[QT_HELPER] Calling error callback")
            self._current_error_callback(error_msg)
        else:
            print(f"[QT_HELPER] No error callback available")
        # Clear callbacks after use (signals will be reconnected for next operation)
        print(f"[QT_HELPER] Clearing callbacks after error")
        self._current_success_callback = None
        self._current_error_callback = None
        self._operation_in_progress = False
        print(f"[QT_HELPER] Callbacks cleared - operation_in_progress: {self._operation_in_progress}")

        # Process next operation if any
        self._process_next_operation()

    def cleanup(self):
        """Clean up the worker."""
        if self._worker:
            self._worker.stop()

# Global helper instance
_qt_helper = QtAsyncHelper()

def run_qt_async(coro, success_callback=None, error_callback=None):
    """
    Convenience function to run async operations with Qt.
    
    Args:
        coro: The coroutine to run
        success_callback: Function to call when operation succeeds
        error_callback: Function to call when operation fails
    """
    print(f"[QT_ASYNC] run_qt_async called with success_callback: {success_callback is not None}, error_callback: {error_callback is not None}")
    _qt_helper.run_async(coro, success_callback, error_callback)
    print(f"[QT_ASYNC] run_qt_async completed")

def cleanup_qt_async():
    """Clean up the Qt async helper."""
    _qt_helper.cleanup() 
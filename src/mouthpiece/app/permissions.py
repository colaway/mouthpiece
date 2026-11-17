"""macOS privacy permissions the menu-bar app needs: Input Monitoring for its global hotkey,
Accessibility for typing (posting key events) and for swallowing the picker's keys.

Permissions belong to the process that runs Python: Terminal, iTerm, VS Code, … — so that is
the app to enable in System Settings, and it must be restarted after a change.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys

_AX_LIB = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"

INPUT_MONITORING_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
)
ACCESSIBILITY_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)


def input_monitoring_granted() -> bool | None:
    """Whether this process may listen to global key events (None: not macOS)."""
    if sys.platform != "darwin":
        return None
    try:
        import Quartz
    except ImportError:
        return None
    return bool(Quartz.CGPreflightListenEventAccess())


def accessibility_granted() -> bool | None:
    """Whether this process is a trusted Accessibility client (None: not macOS)."""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.cdll.LoadLibrary(_AX_LIB)
    except OSError:
        return None
    fn = lib.AXIsProcessTrusted
    fn.restype = ctypes.c_bool
    return bool(fn())


def request_input_monitoring() -> None:
    """Show the macOS prompt (first time only) and open the Input Monitoring pane."""
    try:
        import Quartz

        Quartz.CGRequestListenEventAccess()
    except ImportError:
        pass
    open_settings(INPUT_MONITORING_URL)


def request_accessibility() -> None:
    """Show the macOS "would like to control this computer" prompt (it adds the app to the
    Accessibility list, switched off) and open the Accessibility pane."""
    if sys.platform == "darwin":
        try:
            import objc
            from Foundation import NSDictionary

            lib = ctypes.cdll.LoadLibrary(_AX_LIB)
            fn = lib.AXIsProcessTrustedWithOptions
            fn.restype, fn.argtypes = ctypes.c_bool, [ctypes.c_void_p]
            opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
            fn(objc.pyobjc_id(opts))
        except (ImportError, OSError, AttributeError):
            pass
    open_settings(ACCESSIBILITY_URL)


def open_settings(url: str) -> None:
    subprocess.run(["open", url], check=False)

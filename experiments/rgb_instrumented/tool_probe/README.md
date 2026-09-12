# Vivado 2016.4 startup probe

Objective: isolate the pre-banner startup CPU spin without changing the installation, system environment, hardware, or concurrent builds.

The sole Tcl input is `exit.tcl`, containing `exit`.

| Probe | Console | Vivado logging arguments | Result |
| --- | --- | --- | --- |
| 1 | Pseudo-terminal | `-nojournal -nolog` | Exit 0 in 7.43 seconds; normal banner, source, and exit output |
| 2 | Pseudo-terminal | `-log probe02.log -journal probe02.jou` | Hung before banner; own PID 5268 verified by executable, command, and creation time, then terminated after 41 seconds |
| 3 | Ordinary pipes | `-nojournal -nolog` | Exit 0 in 4.45 seconds; normal banner, source, and exit output |

Startup invocation that passed both minimal probes:

```powershell
& 'C:\Xilinx\Vivado\2016.4\bin\vivado.bat' -mode batch -source '../hardware/build.tcl' -nojournal -nolog
```

Run the example from the benchmark's `build` directory. Console output can be retained externally. A pseudo-terminal was unnecessary for the successful minimal probe.

The parent subsequently started the actual build successfully with a pseudo-terminal and explicit `vivado.log`/`vivado.jou` files, reaching synthesis and RTL elaboration. Therefore, the observed results do not establish logging or journaling as the cause, and the passing switches are not a universally isolated fix. Startup behavior remains ambiguous and could depend on other process or environment conditions. The probes do not isolate log creation from journal creation or establish the underlying runtime defect. No installation or system settings were changed, and no other process was terminated. Parent build PID 5876 and simulator processes were not touched.

Environment: Windows 11 Pro build 26200, Vivado 2016.4 build 1756540. Date: 2026-09-11.

Modified files: this README and `exit.tcl`. Next action: continue the parent's successful actual synthesis; no further startup probes are needed.

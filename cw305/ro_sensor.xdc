# Intentional combinational loops in the ring-oscillator sensor.
# XDC does not allow `if`; set_property on an empty list is a harmless no-op.
set_property ALLOW_COMBINATORIAL_LOOPS TRUE [get_nets -hierarchical -quiet *ro_loop*]

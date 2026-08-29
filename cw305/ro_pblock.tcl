# ============================================================================
# Co-locate the RO voltage sensor with the conv datapath so it reads LOCAL
# IR-drop over the conv instead of the global power grid. This is the #1
# ceiling-raiser for the RO-sensor attack channel (see RO_CHANNEL_RESULTS.md):
# the 0.33 corr wall is a locality/dilution problem, and pulling the sensor
# into the conv's power region is what attacks it.
#
# Constraints only -- cannot change functionality, only placement. Safe to try.
#
# USAGE: source this AFTER read_xdc / read_checkpoint, BEFORE place_design
#   (Vivado non-project flow), or add to the constraints set in a project build.
#   Hierarchy assumed: U_reg_bnn/U_conv (HLS bnn_conv1) and U_reg_bnn/U_ro_sensor.
# ============================================================================

# v1 sensor (ro_counter_sensor, KEEP_HIERARCHY) survives as the hierarchical cell
# U_reg_bnn/U_ro_sensor; the conv survives as U_reg_bnn/U_conv (confirmed via cells_dump).
set SENSOR [get_cells -quiet -hier -filter {NAME =~ *U_ro_sensor*}]
if {[llength $SENSOR] == 0} {
    set SENSOR [get_cells -quiet -hier -filter {NAME =~ *tap_sync* || NAME =~ *tap_prev* || NAME =~ *U_RO*}]
}
if {[llength $SENSOR] == 0} {
    puts "WARNING ro_pblock: no sensor cells (U_ro_sensor/tap_sync/U_RO) -- skipping co-location"
    return
}
set CONV [get_cells -quiet -hier -filter {NAME =~ *U_conv*}]
if {[llength $CONV] == 0} {
    set CONV [get_cells -quiet -hier -filter {NAME =~ *pacc* || NAME =~ *pg_*}]
}

# --- Co-locate sensor (+ conv if grabbable) in a compact clock-region block ----
# XC7A100T-FTG256 clock regions X0Y0..X1Y4. One-column stack (X0Y0:X0Y2) keeps the
# sensor inside the conv's droop field. Widen to X0Y0:X1Y2 if placement is tight.
create_pblock pb_core
add_cells_to_pblock pb_core $SENSOR
if {[llength $CONV] > 0} { add_cells_to_pblock pb_core $CONV }
resize_pblock pb_core -add {CLOCKREGION_X0Y0:CLOCKREGION_X0Y2}
catch { set_property CONTAIN_ROUTING 0 [get_pblocks pb_core] }
puts "ro_pblock: pb_core = [llength $SENSOR] sensor + [llength $CONV] conv cells in X0Y0:X0Y2"

# --- Option B (tighter, POST-PLACE): uncomment to shrink the sensor pblock to
#     the conv's actual placed bounding box. Run AFTER place_design, then
#     place_design again incrementally, or use as a guided re-place.
# place_design
# set xs {}; set ys {}
# foreach c $CONV {
#     set loc [get_property LOC $c]
#     if {$loc ne "" && [regexp {SLICE_X(\d+)Y(\d+)} $loc -> x y]} {
#         lappend xs $x; lappend ys $y
#     }
# }
# if {[llength $xs]} {
#     set x0 [tcl::mathfunc::min {*}$xs]; set x1 [tcl::mathfunc::max {*}$xs]
#     set y0 [tcl::mathfunc::min {*}$ys]; set y1 [tcl::mathfunc::max {*}$ys]
#     create_pblock pb_sens_tight
#     add_cells_to_pblock pb_sens_tight $SENSOR
#     resize_pblock pb_sens_tight -add "SLICE_X${x0}Y${y0}:SLICE_X${x1}Y${y1}"
#     puts "ro_pblock: sensor confined to conv bbox SLICE_X${x0}Y${y0}:SLICE_X${x1}Y${y1}"
# }

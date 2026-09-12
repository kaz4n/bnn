# USB read timing model

The stock NewAE SAM3U firmware configures a 96 MHz MCK and drives the FPGA PCK0
from MCK with divide-by-one. Reads use one setup cycle, three NRD-low cycles,
and four cycles total. READ_MODE=1 samples at the MCK edge that generates NRD
rising. See the pinned [firmware initialization](https://github.com/newaetech/cw305-artix-target/blob/e8a8066e37cc285ed6dc929512a14333ffb57f85/fw/sam3u/CW305_SAM3U_FW/src/main_cw305.c#L233-L253)
and [clock configuration](https://github.com/newaetech/cw305-artix-target/blob/e8a8066e37cc285ed6dc929512a14333ffb57f85/fw/sam3u/CW305_SAM3U_FW/src/config/conf_clock.h#L56-L73).

The SAM3U datasheet specifies 16 ns data setup before NRD high and 0 ns hold for
this 3.3 V, zero-read-hold configuration (Table 42-50). [Microchip datasheet](https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-6430-32-bit-Cortex-M3-Microcontroller-SAM3U4-SAM3U2-SAM3U1_Datasheet.pdf).

This design reads BRAM using the raw external address, avoiding an unnecessary
address-register stage. Nominally: address at C0, RAM read launched at C1,
external capture at C4. The XDC consequently models a three-cycle output path
with the paired two-cycle hold correction. It budgets 17 ns at the output:
the 16 ns receiver setup plus a provisional 1 ns net board/skew allowance.
The 2 ns input arrival assumption is inherited from the stock FPGA example.

The acquisition driver serializes all transactions. RGB streaming stops before
status or data is read; no autonomous workload runs while reading. Consequently
status, count, configuration and buffer contents remain stable through the read
transaction. Concurrent/live reads during pixel streaming are outside this host
interface contract. The output exception applies to that slow peripheral bus,
not to internal capture paths, which retain a single-cycle requirement.

The board/skew allowance and input delay are engineering assumptions, not
measurements of this PCB. A passing timing report establishes closure under
this model. The wrapper testbench checks the nominal one-plus-three-cycle read
waveform; physical captures additionally check every returned sample against
the defined sampling operation. Neither is full board-level timing signoff
over temperature, voltage and firmware variants. A different peripheral timing
profile requires revisiting the model.

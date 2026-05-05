#pragma once

// Hardware adapter selector for AIR56 on UNO Q.
//
// Production builds must provide the FOC/inverter symbols declared in
// air56_unoq_hw_port.h. The mock adapter is intentionally opt-in and is only
// for protocol loopback or compile-smoke checks.

#if defined(AIR56_UNOQ_USE_MOCK_HW)
#include "air56_unoq_hw_mock.h"
#else
#include "air56_unoq_hw_port.h"
#endif

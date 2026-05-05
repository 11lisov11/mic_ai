#pragma once

#include <stdint.h>

#include "air56_unoq_config.h"

// Mock adapter for link loopback and firmware compile smoke only.
// Do not enable AIR56_UNOQ_USE_MOCK_HW for a motor-connected build.

static float g_air56_mock_id_ref_amp = AIR56_UNOQ_ID_REF_BASE_A;

static inline float air56_hw_read_omega_meas_rad_s(void) {
  return 0.0f;
}

static inline float air56_hw_read_omega_ref_rad_s(void) {
  return 0.0f;
}

static inline float air56_hw_read_id_amp(void) {
  return g_air56_mock_id_ref_amp;
}

static inline float air56_hw_read_iq_amp(void) {
  return 0.0f;
}

static inline float air56_hw_read_vdc_volt(void) {
  return 24.0f;
}

static inline float air56_hw_read_irms_amp(void) {
  return 0.0f;
}

static inline float air56_hw_read_pin_watt(void) {
  return 0.0f;
}

static inline uint16_t air56_hw_read_status_bits(void) {
  return 0u;
}

static inline void air56_hw_apply_id_ref_amp(float id_ref_amp) {
  g_air56_mock_id_ref_amp = id_ref_amp;
}

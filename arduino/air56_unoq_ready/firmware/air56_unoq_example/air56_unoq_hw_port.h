#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Board integration contract.
//
// The real STM32U585 FOC/inverter project must implement these functions.
// Keep this file unchanged and add a board-side translation unit that maps
// these calls to the actual current loop, speed loop, ADC scaling, encoder or
// observer, DC bus measurement, input power estimator, and fault flags.
float air56_foc_get_omega_meas_rad_s(void);
float air56_foc_get_omega_ref_rad_s(void);
float air56_foc_get_id_amp(void);
float air56_foc_get_iq_amp(void);
float air56_foc_get_vdc_volt(void);
float air56_foc_get_irms_amp(void);
float air56_foc_get_pin_watt(void);
uint16_t air56_foc_get_status_bits(void);
void air56_foc_set_id_ref_amp(float id_ref_amp);

#ifdef __cplusplus
}
#endif

static inline float air56_hw_read_omega_meas_rad_s(void) {
  return air56_foc_get_omega_meas_rad_s();
}

static inline float air56_hw_read_omega_ref_rad_s(void) {
  return air56_foc_get_omega_ref_rad_s();
}

static inline float air56_hw_read_id_amp(void) {
  return air56_foc_get_id_amp();
}

static inline float air56_hw_read_iq_amp(void) {
  return air56_foc_get_iq_amp();
}

static inline float air56_hw_read_vdc_volt(void) {
  return air56_foc_get_vdc_volt();
}

static inline float air56_hw_read_irms_amp(void) {
  return air56_foc_get_irms_amp();
}

static inline float air56_hw_read_pin_watt(void) {
  return air56_foc_get_pin_watt();
}

static inline uint16_t air56_hw_read_status_bits(void) {
  return air56_foc_get_status_bits();
}

static inline void air56_hw_apply_id_ref_amp(float id_ref_amp) {
  air56_foc_set_id_ref_amp(id_ref_amp);
}

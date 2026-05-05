#ifndef UNO_Q_CONTROL_H
#define UNO_Q_CONTROL_H

#include <stdint.h>

typedef struct {
  uint8_t enable_ai;
  int16_t id_ref_q10;
  uint8_t gated;
} unoq_gate_result_t;

static inline int16_t unoq_clamp_i16(int16_t value, int16_t lo, int16_t hi) {
  if (value < lo) {
    return lo;
  }
  if (value > hi) {
    return hi;
  }
  return value;
}

static inline uint8_t unoq_status_fault(uint16_t status, uint16_t fault_mask) {
  if (fault_mask == 0u) {
    return status != 0u;
  }
  return (status & fault_mask) != 0u;
}

static inline unoq_gate_result_t unoq_apply_gates(
    int32_t speed_err_q10,
    int32_t speed_tol_q10,
    uint16_t status,
    uint16_t fault_mask,
    int16_t id_ref_base_q10,
    int16_t id_ref_cmd_q10,
    uint8_t disable_on_guard,
    uint8_t disable_on_fault) {
  const uint8_t fault = unoq_status_fault(status, fault_mask);
  const uint8_t guard = (speed_err_q10 > speed_tol_q10);
  uint8_t enable_ai = 1u;
  if ((fault && disable_on_fault) || (guard && disable_on_guard)) {
    enable_ai = 0u;
  }
  if (fault || guard) {
    unoq_gate_result_t out = {enable_ai, id_ref_base_q10, 1u};
    return out;
  }
  unoq_gate_result_t out = {enable_ai, id_ref_cmd_q10, 0u};
  return out;
}

static inline int16_t unoq_rate_limit(
    int16_t prev_q10,
    int16_t target_q10,
    int16_t max_delta_q10) {
  if (max_delta_q10 <= 0) {
    return prev_q10;
  }
  int32_t delta = (int32_t)target_q10 - (int32_t)prev_q10;
  if (delta > max_delta_q10) {
    delta = max_delta_q10;
  } else if (delta < -max_delta_q10) {
    delta = -max_delta_q10;
  }
  return (int16_t)(prev_q10 + delta);
}

#endif

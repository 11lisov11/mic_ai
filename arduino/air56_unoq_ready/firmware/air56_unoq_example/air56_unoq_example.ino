#include <Arduino.h>

#include "air56_unoq_config.h"
#include "air56_unoq_hw.h"
#include "uno_q_control.h"
#include "uno_q_protocol.h"

#define AIR56_UNOQ_LINK Serial1
#define AIR56_UNOQ_DEBUG Serial

static uint32_t g_last_tx_ms = 0u;
static uint32_t g_last_cmd_ms = 0u;
static int16_t g_last_id_ref_q10 = (int16_t)(AIR56_UNOQ_ID_REF_BASE_A * UNO_Q_CURRENT_SCALE);
static int16_t g_last_cmd_id_ref_q10 = (int16_t)(AIR56_UNOQ_ID_REF_BASE_A * UNO_Q_CURRENT_SCALE);
static uint8_t g_last_cmd_enable_ai = 0u;

static bool read_exact(uint8_t *dst, size_t len) {
  if (AIR56_UNOQ_LINK.available() < (int)len) {
    return false;
  }
  size_t got = AIR56_UNOQ_LINK.readBytes(reinterpret_cast<char *>(dst), len);
  return got == len;
}

static bool read_command(unoq_command_t *cmd) {
  uint8_t payload[sizeof(unoq_command_t)] = {0};
  if (!read_exact(payload, sizeof(payload))) {
    return false;
  }
  memcpy(cmd, payload, sizeof(*cmd));
  if (AIR56_UNOQ_ENABLE_CRC) {
    uint16_t crc_expected = cmd->crc;
    cmd->crc = 0u;
    uint16_t crc_actual = unoq_crc16_ccitt(reinterpret_cast<const uint8_t *>(cmd), sizeof(*cmd));
    cmd->crc = crc_expected;
    if (crc_expected != crc_actual) {
      return false;
    }
  }
  if (cmd->enable_ai > 1u) {
    return false;
  }
  return true;
}

static void write_telemetry() {
  unoq_telemetry_t telem;
  telem.t_ms = millis();
  telem.omega_meas_q10 = unoq_float_to_i16_sat(air56_hw_read_omega_meas_rad_s() * UNO_Q_OMEGA_SCALE);
  telem.omega_ref_q10 = unoq_float_to_i16_sat(air56_hw_read_omega_ref_rad_s() * UNO_Q_OMEGA_SCALE);
  telem.id_q10 = unoq_float_to_i16_sat(air56_hw_read_id_amp() * UNO_Q_CURRENT_SCALE);
  telem.iq_q10 = unoq_float_to_i16_sat(air56_hw_read_iq_amp() * UNO_Q_CURRENT_SCALE);
  telem.vdc_q8 = unoq_float_to_u16_sat(air56_hw_read_vdc_volt() * UNO_Q_VDC_SCALE);
  telem.i_rms_q10 = unoq_float_to_i16_sat(air56_hw_read_irms_amp() * UNO_Q_CURRENT_SCALE);
  telem.p_in_q2 = unoq_float_to_i16_sat(air56_hw_read_pin_watt() * UNO_Q_POWER_SCALE);
  telem.status = air56_hw_read_status_bits();
  AIR56_UNOQ_LINK.write(reinterpret_cast<const uint8_t *>(&telem), sizeof(telem));
}

void setup() {
  AIR56_UNOQ_DEBUG.begin(115200);
  AIR56_UNOQ_LINK.begin(921600);
  air56_hw_apply_id_ref_amp(AIR56_UNOQ_ID_REF_BASE_A);
  AIR56_UNOQ_DEBUG.println("AIR56 UNO Q example started");
}

void loop() {
  const uint32_t now_ms = millis();
  if ((now_ms - g_last_tx_ms) >= AIR56_UNOQ_TELEMETRY_PERIOD_MS) {
    g_last_tx_ms = now_ms;
    write_telemetry();
  }

  unoq_command_t cmd;
  bool have_cmd = read_command(&cmd);
  if (have_cmd) {
    g_last_cmd_ms = now_ms;
    g_last_cmd_id_ref_q10 = cmd.id_ref_q10;
    g_last_cmd_enable_ai = cmd.enable_ai ? 1u : 0u;
  }

  const int16_t id_ref_base_q10 = (int16_t)lroundf(AIR56_UNOQ_ID_REF_BASE_A * UNO_Q_CURRENT_SCALE);
  const int16_t id_ref_min_q10 = (int16_t)lroundf(AIR56_UNOQ_ID_REF_MIN_A * UNO_Q_CURRENT_SCALE);
  const int16_t id_ref_max_q10 = (int16_t)lroundf(AIR56_UNOQ_ID_REF_MAX_A * UNO_Q_CURRENT_SCALE);
  const int16_t max_delta_q10 = (int16_t)lroundf(AIR56_UNOQ_SLEW_A_PER_CYCLE * UNO_Q_CURRENT_SCALE);

  const int16_t omega_ref_q10 = unoq_float_to_i16_sat(air56_hw_read_omega_ref_rad_s() * UNO_Q_OMEGA_SCALE);
  const int16_t omega_meas_q10 = unoq_float_to_i16_sat(air56_hw_read_omega_meas_rad_s() * UNO_Q_OMEGA_SCALE);
  const int32_t speed_delta_q10 = (int32_t)omega_ref_q10 - (int32_t)omega_meas_q10;
  const int32_t speed_err_q10 = speed_delta_q10 >= 0 ? speed_delta_q10 : -speed_delta_q10;
  const float omega_ref_abs = fabsf((float)omega_ref_q10 / UNO_Q_OMEGA_SCALE);
  const float speed_tol = max(AIR56_UNOQ_SPEED_TOL_ABS_RAD_S, AIR56_UNOQ_SPEED_TOL_REL * omega_ref_abs);
  const int32_t speed_tol_q10 = (int32_t)unoq_float_to_i16_sat(speed_tol * UNO_Q_OMEGA_SCALE);

  uint8_t timeout = (now_ms - g_last_cmd_ms) > AIR56_UNOQ_COMMAND_TIMEOUT_MS;
  int16_t requested_q10 = id_ref_base_q10;
  uint8_t enable_ai = 0u;
  if (!timeout) {
    // Hold the last valid Linux command between 10 ms packets; timeout owns fallback.
    requested_q10 = unoq_clamp_i16(g_last_cmd_id_ref_q10, id_ref_min_q10, id_ref_max_q10);
    enable_ai = g_last_cmd_enable_ai ? 1u : 0u;
  } else {
    g_last_cmd_enable_ai = 0u;
  }

  unoq_gate_result_t gate = unoq_apply_gates(
      speed_err_q10,
      speed_tol_q10,
      air56_hw_read_status_bits(),
      AIR56_UNOQ_FAULT_MASK,
      id_ref_base_q10,
      requested_q10,
      AIR56_UNOQ_DISABLE_ON_GUARD,
      AIR56_UNOQ_DISABLE_ON_FAULT);

  int16_t next_q10 = enable_ai ? gate.id_ref_q10 : id_ref_base_q10;
  next_q10 = unoq_clamp_i16(next_q10, id_ref_min_q10, id_ref_max_q10);
  next_q10 = unoq_rate_limit(g_last_id_ref_q10, next_q10, max_delta_q10);
  g_last_id_ref_q10 = next_q10;

  air56_hw_apply_id_ref_amp((float)next_q10 / UNO_Q_CURRENT_SCALE);
}

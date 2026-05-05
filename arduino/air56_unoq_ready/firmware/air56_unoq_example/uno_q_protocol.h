#ifndef UNO_Q_PROTOCOL_H
#define UNO_Q_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Omega uses 128 counts per rad/s so int16 covers the AIR56 nominal speed.
#define UNO_Q_OMEGA_SCALE 128
#define UNO_Q_CURRENT_SCALE 1024
#define UNO_Q_VDC_SCALE 256
#define UNO_Q_POWER_SCALE 4

#define UNO_Q_CRC16_POLY 0x1021u
#define UNO_Q_CRC16_INIT 0xFFFFu

static inline int16_t unoq_float_to_i16_sat(float value) {
  if (!isfinite(value)) {
    return 0;
  }
  if (value > 32767.0f) {
    return 32767;
  }
  if (value < -32768.0f) {
    return -32768;
  }
  return (int16_t)lroundf(value);
}

static inline uint16_t unoq_float_to_u16_sat(float value) {
  if (!isfinite(value)) {
    return 0u;
  }
  if (value > 65535.0f) {
    return 65535u;
  }
  if (value < 0.0f) {
    return 0u;
  }
  return (uint16_t)lroundf(value);
}

#if defined(__GNUC__)
#define UNO_Q_PACKED __attribute__((packed))
#else
#define UNO_Q_PACKED
#endif

typedef struct UNO_Q_PACKED {
  uint32_t t_ms;
  int16_t omega_meas_q10;
  int16_t omega_ref_q10;
  int16_t id_q10;
  int16_t iq_q10;
  uint16_t vdc_q8;
  int16_t i_rms_q10;
  int16_t p_in_q2;
  uint16_t status;
} unoq_telemetry_t;

typedef struct UNO_Q_PACKED {
  uint32_t t_ms;
  uint8_t enable_ai;
  int16_t id_ref_q10;
  uint16_t crc;
} unoq_command_t;

#ifdef __cplusplus
static_assert(sizeof(unoq_telemetry_t) == 20, "unoq_telemetry_t ABI must stay 20 bytes");
static_assert(sizeof(unoq_command_t) == 9, "unoq_command_t ABI must stay 9 bytes");
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(unoq_telemetry_t) == 20, "unoq_telemetry_t ABI must stay 20 bytes");
_Static_assert(sizeof(unoq_command_t) == 9, "unoq_command_t ABI must stay 9 bytes");
#endif

static inline uint16_t unoq_crc16_ccitt(const uint8_t *data, size_t len) {
  uint16_t crc = UNO_Q_CRC16_INIT;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)(data[i] << 8);
    for (uint8_t b = 0; b < 8; ++b) {
      if (crc & 0x8000u) {
        crc = (uint16_t)((crc << 1) ^ UNO_Q_CRC16_POLY);
      } else {
        crc = (uint16_t)(crc << 1);
      }
    }
  }
  return crc;
}

#ifdef __cplusplus
}
#endif

#endif

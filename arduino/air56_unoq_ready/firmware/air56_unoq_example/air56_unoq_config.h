#pragma once

#include <Arduino.h>

static constexpr float AIR56_UNOQ_ID_REF_BASE_A = 1.35f;
static constexpr float AIR56_UNOQ_ID_REF_MIN_A = 1.10f;
static constexpr float AIR56_UNOQ_ID_REF_MAX_A = 1.70f;

static constexpr uint32_t AIR56_UNOQ_TELEMETRY_PERIOD_MS = 10u;
static constexpr uint32_t AIR56_UNOQ_COMMAND_TIMEOUT_MS = 100u;
static constexpr float AIR56_UNOQ_SLEW_A_PER_CYCLE = 0.08f;

static constexpr float AIR56_UNOQ_SPEED_TOL_REL = 0.08f;
static constexpr float AIR56_UNOQ_SPEED_TOL_ABS_RAD_S = 0.0f;
static constexpr uint16_t AIR56_UNOQ_FAULT_MASK = 0u;
static constexpr uint8_t AIR56_UNOQ_DISABLE_ON_GUARD = 0u;
static constexpr uint8_t AIR56_UNOQ_DISABLE_ON_FAULT = 1u;

static constexpr uint8_t AIR56_UNOQ_ENABLE_CRC = 1u;

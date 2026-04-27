/*
  fake_flight_sensor.ino  v3.0
  ============================
  Streams synthetic rocket telemetry over USB Serial in CSV format,
  calibrated to match the RocketPy training data distribution.

  SIGN CONVENTIONS (must match training data!):
    - v_vel:  NEGATIVE during ascent  (RocketPy convention)
    - v_acc:  POSITIVE during powered ascent, NEGATIVE during coast
    - altitude: 0 at launch pad (AGL), increases upward
    - Target apogee: ~3100 m AGL  (~10,200 ft)

  PROTOCOL:
    1. Arduino boots -> sends "# READY\n" and waits.
    2. Pi sends 'G' -> Arduino runs 10-second countdown then launches.
    3. After apogee -> sends "# APOGEE alt=Xm\n" then "# READY\n".
    4. Pi can trigger another flight by sending 'G' again.

  OUTPUT FORMAT (25 ms intervals):
    t,altitude,v_vel,v_acc,t_vel,h_vel,pitch,dynp,mach,pressure
*/

// ── Physical constants ────────────────────────────────────────────────────────
const float G         = 9.80665f;
const float DT        = 0.025f;      // seconds per step
const float BURN_TIME = 4.70f;       // s
// Calibrated to produce ~3100 m AGL apogee (~10,200 ft, matching training data)
const float THRUST    = 540.0f;      // N  (tuned for ~3100 m apogee)
const float MASS_WET  = 7.00f;       // kg
const float PROP_MASS = 1.80f;       // kg
const float CD        = 0.45f;
const float AREA      = 0.0079f;     // m^2
const float P_SEA     = 101325.0f;
const float T_SEA     = 288.15f;
const float L_RATE    = 0.0065f;     // K/m
const float R_GAS     = 287.058f;
const float GAMMA     = 1.40f;

// ── Simple LCG PRNG ───────────────────────────────────────────────────────────
static unsigned long _seed = 12345UL;

float rnd_unit() {
  _seed = _seed * 1664525UL + 1013904223UL;
  return ((float)(_seed & 0x7FFFFFFF)) / 2147483648.0f - 1.0f;
}

float gauss(float sigma) {
  float u = 0.0f;
  while (u == 0.0f) u = (rnd_unit() + 1.0f) * 0.5f;
  float v = (rnd_unit() + 1.0f) * 0.5f;
  return sqrt(-2.0f * log(u)) * cos(6.28318f * v) * sigma;
}

// ── ISA atmosphere ────────────────────────────────────────────────────────────
void isa(float alt_msl, float *P, float *rho) {
  float T = T_SEA - L_RATE * alt_msl;
  if (T < 216.65f) T = 216.65f;
  *P   = P_SEA * pow(T / T_SEA, G / (L_RATE * R_GAS));
  *rho = *P / (R_GAS * T);
}

float speed_of_sound(float alt_msl) {
  float T = T_SEA - L_RATE * alt_msl;
  if (T < 216.65f) T = 216.65f;
  return sqrt(GAMMA * R_GAS * T);
}

// ── Flight state ──────────────────────────────────────────────────────────────
typedef enum { STATE_READY, STATE_COUNTDOWN, STATE_FLIGHT, STATE_DONE } State;
State state = STATE_READY;

float t_flight   = 0.0f;
float alt_agl    = 0.0f;    // altitude above ground level (m)
float v_up       = 0.0f;    // upward velocity (positive = ascending)
float h_vel      = 0.0f;

unsigned long last_step_ms    = 0;
unsigned long countdown_start = 0;
int countdown_last            = -1;

// ── Reset ─────────────────────────────────────────────────────────────────────
void reset_flight() {
  t_flight = 0.0f;
  alt_agl  = 0.0f;
  v_up     = 0.0f;
  h_vel    = 0.0f;
  _seed    = (unsigned long) millis();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("# READY");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {

  // STATE: READY
  if (state == STATE_READY || state == STATE_DONE) {
    if (Serial.available() > 0) {
      Serial.read();   // consume any byte
      reset_flight();
      countdown_start = millis();
      countdown_last  = -1;
      state = STATE_COUNTDOWN;
      Serial.println("# COUNTDOWN");
    }
    return;
  }

  // STATE: COUNTDOWN
  if (state == STATE_COUNTDOWN) {
    unsigned long elapsed = millis() - countdown_start;
    int remaining = 10 - (int)(elapsed / 1000);
    if (remaining >= 0 && remaining != countdown_last) {
      countdown_last = remaining;
      Serial.print("# T-");
      Serial.println(remaining + 1);
    }
    if (elapsed >= 10000UL) {
      Serial.println("# LIFTOFF");
      last_step_ms = millis();
      state = STATE_FLIGHT;
    }
    return;
  }

  // STATE: FLIGHT
  unsigned long now = millis();
  if ((now - last_step_ms) < (unsigned long)(DT * 1000.0f)) return;
  last_step_ms = now;

  t_flight += DT;

  // Mass (decreases linearly during burn)
  float frac = (t_flight < BURN_TIME) ? t_flight / BURN_TIME : 1.0f;
  float mass = MASS_WET - PROP_MASS * frac;

  // Atmosphere at current altitude MSL (assume pad is near sea level)
  float P, rho;
  isa(alt_agl, &P, &rho);
  float a_snd = speed_of_sound(alt_agl);

  // Drag (opposes motion)
  float speed = sqrt(v_up * v_up + h_vel * h_vel);
  float drag  = 0.5f * rho * speed * speed * CD * AREA;

  // Thrust (during motor burn only)
  float thrust = (t_flight <= BURN_TIME) ? THRUST : 0.0f;

  // Net upward acceleration (positive = up)
  float accel = (thrust - drag) / mass - G;

  // Integrate
  float v_up_new  = v_up + accel * DT;
  float alt_new   = alt_agl + v_up * DT + 0.5f * accel * DT * DT;
  if (alt_new < 0.0f) alt_new = 0.0f;

  // Horizontal drift
  float h_drift = 0.5f * 0.1f * (1.0f - exp(-t_flight * 0.5f));
  h_vel = h_drift;

  // Pitch off vertical
  float pitch_c = 2.0f * (t_flight / BURN_TIME) * exp(-t_flight * 0.3f);

  // Derived quantities
  float speed_new = sqrt(v_up_new * v_up_new + h_vel * h_vel);
  float dynp_c    = 0.5f * rho * speed_new * speed_new;
  float mach_c    = speed_new / a_snd;

  // ── OUTPUT (RocketPy sign conventions!) ──────────────────────────────────
  // v_vel:  NEGATIVE during ascent (negate v_up_new)
  // v_acc:  POSITIVE during powered flight (accel is already positive then)
  // altitude: AGL (starts at 0)
  float out_vvel = -(v_up_new + gauss(1.5f));   // negate + noise
  float out_vacc = accel + gauss(0.5f);          // same sign + noise
  float out_alt  = alt_new + gauss(1.0f);
  float out_tvel = fabs(speed_new + gauss(1.5f));
  float out_hvel = h_vel + gauss(1.0f);
  float out_pitch = pitch_c + gauss(0.5f);
  float out_dynp = dynp_c * (1.0f + gauss(0.02f));
  float out_mach = mach_c * (1.0f + gauss(0.015f));
  float out_pres = P + gauss(20.0f);

  Serial.print(t_flight, 3); Serial.print(',');
  Serial.print(out_alt,  2); Serial.print(',');
  Serial.print(out_vvel, 3); Serial.print(',');
  Serial.print(out_vacc, 3); Serial.print(',');
  Serial.print(out_tvel, 3); Serial.print(',');
  Serial.print(out_hvel, 3); Serial.print(',');
  Serial.print(out_pitch,4); Serial.print(',');
  Serial.print(out_dynp, 2); Serial.print(',');
  Serial.print(out_mach, 4); Serial.print(',');
  Serial.println(out_pres,1);

  // Update state
  v_up    = v_up_new;
  alt_agl = alt_new;

  // Apogee: upward velocity crosses zero after burnout
  if (v_up_new <= 0.0f && t_flight > BURN_TIME + 0.5f) {
    Serial.print("# APOGEE alt=");
    Serial.print(alt_new, 1);
    Serial.println("m");
    Serial.println("# READY");
    state = STATE_DONE;
  }
}

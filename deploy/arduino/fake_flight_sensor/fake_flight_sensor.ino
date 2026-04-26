/*
  fake_flight_sensor.ino
  ======================
  Streams synthetic rocket telemetry over USB Serial in CSV format.

  Each line (every 25 ms) sends:
      T,altitude,v_vel,v_acc,t_vel,h_vel,pitch,dynp,mach,pressure

  The Pi reads this with flight_demo.py --serial /dev/ttyUSB0

  USAGE
  -----
  1.  Open in Arduino IDE, select your board (Uno / Nano / Mega / etc.)
  2.  Upload
  3.  On the Pi: python flight_demo.py --serial /dev/ttyUSB0 --baud 115200

  The sketch blocks for 10 s (countdown) then fires.
  Motor burn is ~4.7 s.  After burnout, coast to simulated apogee.

  Physical model is a simplified Euler integrator matching the Python
  generator.  Numbers are realistic for a mid-power L-class rocket.
*/

// ── Physical constants ────────────────────────────────────────────────────────
const float G          = 9.80665f;
const float DT         = 0.025f;      // seconds per step
const float LAUNCH_ALT = 1200.0f;    // m (pad altitude above MSL)
const float BURN_TIME  = 4.70f;      // s
const float THRUST     = 1800.0f;    // N
const float MASS_WET   = 7.20f;      // kg
const float PROP_MASS  = 1.80f;      // kg
const float CD         = 0.45f;
const float AREA       = 0.0079f;    // m²  (10 cm diameter)
const float RHO_SEA    = 1.225f;
const float P_SEA      = 101325.0f;
const float T_SEA      = 288.15f;
const float L_RATE     = 0.0065f;    // K/m
const float R_GAS      = 287.058f;
const float GAMMA      = 1.40f;

// ── Noise (approximate match to Python noise injection) ───────────────────────
// Using a simple LCG PRNG so we don't need <random>
static unsigned long _seed = 12345UL;

float rnd_unit() {
  _seed = _seed * 1664525UL + 1013904223UL;
  return ((float)(_seed & 0x7FFFFFFF)) / 2147483648.0f - 1.0f;
}

float gauss(float sigma) {
  // Box-Muller (cheap version using two uniform samples)
  float u = 0.0f;
  while (u == 0.0f) u = (rnd_unit() + 1.0f) * 0.5f;  // (0,1]
  float v = (rnd_unit() + 1.0f) * 0.5f;
  float z = sqrt(-2.0f * log(u)) * cos(6.28318f * v);
  return z * sigma;
}

// ── State ─────────────────────────────────────────────────────────────────────
float t_flight = 0.0f;
float altitude = LAUNCH_ALT;
float v_vel    = 0.0f;    // vertical velocity m/s
float h_vel    = 0.0f;    // horizontal velocity m/s
float pitch    = 0.0f;    // degrees

bool launched  = false;
bool apogee    = false;

unsigned long last_step_ms = 0;

// ── ISA pressure/density ──────────────────────────────────────────────────────
void isa(float alt, float *P, float *rho) {
  float T = T_SEA - L_RATE * alt;
  if (T < 216.65f) T = 216.65f;
  *P   = P_SEA * pow(T / T_SEA, G / (L_RATE * R_GAS));
  *rho = *P / (R_GAS * T);
}

float speed_of_sound(float alt) {
  float T = T_SEA - L_RATE * alt;
  if (T < 216.65f) T = 216.65f;
  return sqrt(GAMMA * R_GAS * T);
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);

  // Send a header comment so the Pi can identify the stream
  Serial.println("# RocketPy Fake Flight Sensor v1.0");
  Serial.println("# Fields: t,altitude,v_vel,v_acc,t_vel,h_vel,pitch,dynp,mach,pressure");

  // 10 second countdown
  for (int i = 10; i >= 1; i--) {
    Serial.print("# T-");
    Serial.println(i);
    delay(1000);
  }

  Serial.println("# LIFTOFF");
  launched = true;
  last_step_ms = millis();
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  if (!launched || apogee) return;

  unsigned long now = millis();
  if ((now - last_step_ms) < (unsigned long)(DT * 1000.0f)) return;
  last_step_ms = now;

  t_flight += DT;

  // Mass
  float frac   = min(t_flight / BURN_TIME, 1.0f);
  float mass   = MASS_WET - PROP_MASS * frac;

  // ISA
  float P, rho;
  isa(altitude, &P, &rho);
  float a_snd = speed_of_sound(altitude);

  // Total velocity
  float t_vel = sqrt(v_vel * v_vel + h_vel * h_vel);

  // Drag + thrust
  float drag   = 0.5f * rho * t_vel * t_vel * CD * AREA;
  float thrust = (t_flight <= BURN_TIME) ? THRUST : 0.0f;
  float accel  = (thrust - drag) / mass - G;

  // Integrate
  float v_vel_new = v_vel + accel * DT;
  float alt_new   = altitude + v_vel * DT + 0.5f * accel * DT * DT;

  // Horizontal drift
  float h_drift = 4.0f * 0.05f * (1.0f - exp(-t_flight * 0.3f));
  h_vel = h_drift;

  // Pitch
  float pitch_clean = 1.5f * (t_flight / BURN_TIME) * exp(-t_flight * 0.3f);

  // Aero
  float t_vel_new  = sqrt(v_vel_new * v_vel_new + h_vel * h_vel);
  float dynp_clean = 0.5f * rho * t_vel_new * t_vel_new;
  float mach_clean = t_vel_new / a_snd;

  // Apply noise
  float alt_n  = alt_new   + gauss(1.0f);
  float vv_n   = v_vel_new + gauss(1.5f);
  float va_n   = accel     + gauss(0.5f);
  float tv_n   = fabs(t_vel_new + gauss(1.5f));
  float hv_n   = h_vel     + gauss(1.0f);
  float pt_n   = pitch_clean + gauss(0.5f);
  float dq_n   = dynp_clean * (1.0f + gauss(0.02f));
  float mc_n   = mach_clean * (1.0f + gauss(0.015f));
  float pr_n   = P + gauss(20.0f);

  // Print CSV
  Serial.print(t_flight, 3); Serial.print(',');
  Serial.print(alt_n,    2); Serial.print(',');
  Serial.print(vv_n,     3); Serial.print(',');
  Serial.print(va_n,     3); Serial.print(',');
  Serial.print(tv_n,     3); Serial.print(',');
  Serial.print(hv_n,     3); Serial.print(',');
  Serial.print(pt_n,     4); Serial.print(',');
  Serial.print(dq_n,     2); Serial.print(',');
  Serial.print(mc_n,     4); Serial.print(',');
  Serial.println(pr_n,   1);

  // Advance state
  v_vel    = v_vel_new;
  altitude = alt_new;
  pitch    = pitch_clean;

  // Stop at apogee
  if (v_vel_new <= 0.0f && t_flight > BURN_TIME + 0.5f) {
    Serial.print("# APOGEE  alt=");
    Serial.print(alt_new, 1);
    Serial.println("m");
    apogee = true;
  }
}

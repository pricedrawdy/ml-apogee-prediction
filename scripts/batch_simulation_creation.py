import numpy as np
import pandas as pd
from pathlib import Path
from rocketpy import Environment, SolidMotor, Rocket, Flight, TrapezoidalFins, EllipticalFins, RailButtons, NoseCone, Tail, Parachute
import datetime
import matplotlib.pyplot as plt

from rocket_parameter_loader import load_parameters


# Load parameters from JSON
params = load_parameters()
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "rocket-info"
output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

motor_params = params["motors"]
nosecone_params = params["nosecones"]
fins_params = params["trapezoidal_fins"]["0"]
tail_params = params["tails"]["0"]
parachute_params = params["parachutes"]
rocket_params = params["rocket"]
rail_button_params = params["rail_buttons"]
env_params = params["environment"]

# Handle cross-platform path parsing (parameters.json may contain Windows-style paths)
from pathlib import PureWindowsPath
thrust_source_name = PureWindowsPath(motor_params["thrust_source"]).name
thrust_source_path = data_dir / thrust_source_name
motor = SolidMotor(
    thrust_source=str(thrust_source_path),
    dry_mass=motor_params["dry_mass"],
    center_of_dry_mass_position=motor_params["center_of_dry_mass_position"],
    dry_inertia=motor_params["dry_inertia"],
    grains_center_of_mass_position=motor_params["grains_center_of_mass_position"],
    grain_number=motor_params["grain_number"],
    grain_density=motor_params["grain_density"],
    grain_outer_radius=motor_params["grain_outer_radius"],
    grain_initial_inner_radius=motor_params["grain_initial_inner_radius"],
    grain_initial_height=motor_params["grain_initial_height"],
    grain_separation=motor_params["grain_separation"],
    nozzle_radius=motor_params["nozzle_radius"],
    nozzle_position=motor_params["nozzle_position"],
    throat_radius=motor_params["throat_radius"],
    reshape_thrust_curve=False,  # Not implemented in Rocket-Serializer
    interpolation_method="linear",
    coordinate_system_orientation=motor_params["coordinate_system_orientation"],
)

# Nose Cone
nosecone = NoseCone(
    length=nosecone_params["length"],
    kind=nosecone_params["kind"],
    base_radius=nosecone_params["base_radius"],
    rocket_radius=rocket_params["radius"],
    name=nosecone_params["name"],
)

# Fins
trapezoidal_fins = {}
trapezoidal_fins[0] = TrapezoidalFins(
    n=fins_params["number"],
    root_chord=fins_params["root_chord"],
    tip_chord=fins_params["tip_chord"],
    span=fins_params["span"],
    cant_angle=fins_params["cant_angle"],
    sweep_length=fins_params["sweep_length"],
    sweep_angle=fins_params["sweep_angle"],
    rocket_radius=rocket_params["radius"],
    name=fins_params["name"],
)

# Tails
tails = {}
tails[0] = Tail(
    top_radius=tail_params["top_radius"],
    bottom_radius=tail_params["bottom_radius"],
    length=tail_params["length"],
    rocket_radius=rocket_params["radius"],
    name=tail_params["name"],
)

# Parachutes
parachutes = {}
for key, p in parachute_params.items():
    trigger = p["deploy_altitude"] if p["deploy_event"] == "altitude" else p["deploy_event"]
    parachutes[int(key)] = Parachute(
        name=p["name"],
        cd_s=p["cds"],
        trigger=trigger,
        sampling_rate=100,
    )

# Sweep parameters
wind_speeds = np.arange(0, 13, 1)  # 1 m/s granularity: 0-12 m/s
temps = np.arange(280, 315, 2.5)
launch_angles = np.arange(0, 8, 1)
#masses = np.arange(12, 14.5, 0.5)

# Output: time steps (adjust as needed)
timesteps = np.arange(0, 27.025, 0.025)

# Column names
colnames = [
    "Wind Speed (m/s)", "Temperature (K)", "Launch Angle (deg)", "Apogee altitude (m)", "Apogee time (s)"
]
colnames += [f"Vertical velocity (m/s) @ {t:.2f}s" for t in timesteps]
colnames += [f"Vertical acceleration (m/s^2) @ {t:.2f}s" for t in timesteps]
colnames += [f"Total velocity (m/s) @ {t:.2f}s" for t in timesteps]
colnames += [f"Altitude (m) @ {t:.2f}s" for t in timesteps]
colnames += [f"Horizontal velocity (m/s) @ {t:.2f}s" for t in timesteps]
colnames += [f"Pitch angle (deg) @ {t:.2f}s" for t in timesteps]
colnames += [f"Dynamic pressure (Pa) @ {t:.2f}s" for t in timesteps]
colnames += [f"Mach number @ {t:.2f}s" for t in timesteps]
colnames += [f"Pressure (Pa) @ {t:.2f}s" for t in timesteps]

all_rows = []

# Base environment setup (location and date)
base_env = Environment()
base_env.set_location(latitude=env_params["latitude"], longitude=env_params["longitude"])
base_env.set_elevation(env_params["elevation"])
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
base_env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))

# Maximum altitude for wind profile (meters above ground)
MAX_ALT = 5000

for wind in wind_speeds:
    for temp in temps:
        # Create fresh environment for each wind/temperature combination
        # RocketPy requires set_atmospheric_model() to properly configure atmosphere
        env = Environment()
        env.set_location(latitude=env_params["latitude"], longitude=env_params["longitude"])
        env.set_elevation(env_params["elevation"])
        env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
        
        # Set custom atmospheric model with wind and temperature
        # wind_u is the east-west component (positive = from west, i.e., headwind for heading=0)
        # wind_v is the north-south component
        env.set_atmospheric_model(
            type="custom_atmosphere",
            pressure=None,  # Use International Standard Atmosphere pressure profile
            temperature=float(temp),  # Constant temperature in Kelvin
            wind_u=[(0, float(wind)), (MAX_ALT, float(wind))],  # Constant headwind profile
            wind_v=[(0, 0), (MAX_ALT, 0)],  # No crosswind
        )

        for angle in launch_angles:
            rocket = Rocket(
                radius=rocket_params["radius"],
                mass=rocket_params["mass"],
                inertia=rocket_params["inertia"],
                power_off_drag=str(data_dir / PureWindowsPath(rocket_params["drag_curve"]).name),
                power_on_drag=str(data_dir / PureWindowsPath(rocket_params["drag_curve"]).name),
                center_of_mass_without_motor=rocket_params["center_of_mass_without_propellant"],
                coordinate_system_orientation=rocket_params["coordinate_system_orientation"],
            )
            rocket.add_surfaces(
                surfaces=[nosecone, trapezoidal_fins[0], tails[0]],
                positions=[nosecone_params["position"], fins_params["position"], tail_params["position"]],
            )
            rocket.add_motor(motor, position=motor_params["position"])
            rail_buttons = rocket.set_rail_buttons(
                upper_button_position=rail_button_params["upper_position"],
                lower_button_position=rail_button_params["lower_position"],
                angular_position=rail_button_params["angular_position"],
            )
            # Flight
            flight = Flight(
                rocket=rocket,
                environment=env,
                rail_length=4.318,
                inclination=90-angle,
                heading=0,
                terminate_on_apogee=False,
                max_time=600,
            )

            # Actual apogee
            apogee_alt = flight.apogee - env.elevation

            # Apogee time
            apogee_time = flight.apogee_time

            # Mask for valid times
            valid_mask = timesteps <= apogee_time

            # Flight variables - messy timestamps
            # NOTE: For some reason RocketPy flight functions return almost all variables as 2D arrays, a time array and the variable array
            # Thus, we only choose to grab the variable array (y_array), as the time array is the same as flight.time
            stream_velocity_z = flight.stream_velocity_z.y_array 
            stream_velocity_x = flight.stream_velocity_x.y_array
            stream_velocity_y = flight.stream_velocity_y.y_array
            time = flight.time
            stream_acc_v = flight.az.y_array
            altitude = flight.altitude.y_array
            
            # New features from RocketPy
            pitch_angle = flight.theta.y_array  # nutation angle (pitch) in degrees
            dynamic_pressure = flight.dynamic_pressure.y_array
            mach_number = flight.mach_number.y_array
            pressure = flight.pressure.y_array


            # Ensure all arrays are the same length
            min_len = min(len(time), len(stream_velocity_z), len(stream_velocity_x), len(stream_velocity_y))
            time = time[:min_len]
            stream_velocity_z = stream_velocity_z[:min_len]
            stream_velocity_x = stream_velocity_x[:min_len]
            stream_velocity_y = stream_velocity_y[:min_len]
            altitude = altitude[:min_len]
            pitch_angle = pitch_angle[:min_len]
            dynamic_pressure = dynamic_pressure[:min_len]
            mach_number = mach_number[:min_len]
            pressure = pressure[:min_len]

            # Convert all data to fixed, clean timestamps, using interpolation
            v_vel = np.interp(timesteps, time, stream_velocity_z)
            v_acc = np.interp(timesteps, time, stream_acc_v)
            alt_interp = np.interp(timesteps, time, altitude) # Interpolate altitude

            # Calculate total velocity magnitude
            t_vel = np.interp(
                timesteps,
                time,
                np.sqrt(
                    stream_velocity_x ** 2 +
                    stream_velocity_y ** 2 +
                    stream_velocity_z ** 2
                )
            )
            
            # Calculate horizontal velocity magnitude
            h_vel = np.interp(
                timesteps,
                time,
                np.sqrt(stream_velocity_x ** 2 + stream_velocity_y ** 2)
            )
            
            # Interpolate new features
            pitch_interp = np.interp(timesteps, time, pitch_angle)
            dynp_interp = np.interp(timesteps, time, dynamic_pressure)
            mach_interp = np.interp(timesteps, time, mach_number)
            pres_interp = np.interp(timesteps, time, pressure)

            # Pad with NaN after apogee
            v_vel_padded = np.where(valid_mask, v_vel, np.nan)
            v_acc_padded = np.where(valid_mask, v_acc, np.nan)
            t_vel_padded = np.where(valid_mask, t_vel, np.nan)
            alt_interp_padded = np.where(valid_mask, alt_interp, np.nan)
            h_vel_padded = np.where(valid_mask, h_vel, np.nan)
            pitch_padded = np.where(valid_mask, pitch_interp, np.nan)
            dynp_padded = np.where(valid_mask, dynp_interp, np.nan)
            mach_padded = np.where(valid_mask, mach_interp, np.nan)
            pres_padded = np.where(valid_mask, pres_interp, np.nan)

            row = [wind, temp, angle, apogee_alt, apogee_time]
            row += v_vel_padded.tolist()
            row += v_acc_padded.tolist()
            row += t_vel_padded.tolist()
            row += alt_interp_padded.tolist()
            row += h_vel_padded.tolist()
            row += pitch_padded.tolist()
            row += dynp_padded.tolist()
            row += mach_padded.tolist()
            row += pres_padded.tolist()
            all_rows.append(row)

# DataFrame and Save
output_path = output_dir / "batch_dataset_v1.csv"
df = pd.DataFrame(all_rows, columns=colnames)
df.to_csv(output_path, index=False)
print(f"Simulation data saved as {output_path}")


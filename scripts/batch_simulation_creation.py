import numpy as np
import pandas as pd
from rocketpy import Environment, SolidMotor, Rocket, Flight, TrapezoidalFins, EllipticalFins, RailButtons, NoseCone, Tail, Parachute
import datetime
import matplotlib.pyplot as plt

motor = SolidMotor(
    thrust_source='thrust_source.csv',
    dry_mass=0,
    center_of_dry_mass_position=0,
    dry_inertia=[0, 0, 0],
    grains_center_of_mass_position=0,
    grain_number=1,
    grain_density=1115.3675185630957,
    grain_outer_radius=0.049,
    grain_initial_inner_radius=0.0245,
    grain_initial_height=0.597,
    grain_separation=0,
    nozzle_radius=0.036750000000000005,
    nozzle_position=-0.2985,
    throat_radius=0.0245,
    reshape_thrust_curve=False,  # Not implemented in Rocket-Serializer
    interpolation_method='linear',
    coordinate_system_orientation='nozzle_to_combustion_chamber',
)

# Nose Cone
nosecone = NoseCone(
    length=0.9270999999999999,
    kind='Von Karman',
    base_radius=0.078359,
    rocket_radius=0.078359,
    name='0.9270999999999999',
)

# Fins
trapezoidal_fins = {}
trapezoidal_fins[0] = TrapezoidalFins(
    n=3,
    root_chord=0.29999939999999997,
    tip_chord=0.100076,
    span=0.20246339999999996,
    cant_angle=0.0,
    sweep_length= 0.18542,
    sweep_angle= None,
    rocket_radius=0.078359,
    name='Main Fins',
)

# Tails
tails = {}
tails[0] = Tail(
    top_radius=0.078359,
    bottom_radius=0.05505,
    length=0.1016,
    rocket_radius=0.078359,
    name='Boattail',
)

# Parachutes
parachutes = {}
parachutes[0] = Parachute(
    name='Main Bag with Main Chute',
    cd_s=10.274,
    trigger=365.760,
    sampling_rate=100, 
)
parachutes[1] = Parachute(
    name='Drogue Bag with Drogue Parachute',
    cd_s=0.467,
    trigger='apogee',
    sampling_rate=100, 
)

# Sweep parameters
wind_speeds = np.arange(0, 12, 3)
temps = np.arange(273, 307, 10)
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

all_rows = []

env = Environment()
env.set_location(latitude=32.9428, longitude=-106.912)
env.set_elevation(60.96)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))

'''wind = 6
temp = 293
angle = 4'''

for wind in wind_speeds:
    env.wind_speed = wind    

    for temp in temps:
        env.temperature = temp

        for angle in launch_angles:
            rocket = Rocket(
                radius=0.078359,
                mass=21.215,
                inertia=[0.014, 0.014, 4.458],
                power_off_drag='drag_curve.csv',
                power_on_drag='drag_curve.csv',
                center_of_mass_without_motor=2.122,
                coordinate_system_orientation='nose_to_tail',
            )
            rocket.add_surfaces(surfaces=[nosecone, trapezoidal_fins[0], tails[0]], positions=[0.0, 3.0275276000000004, 3.324225])
            rocket.add_motor(motor, position= 3.1035067693124514)
            rail_buttons = rocket.set_rail_buttons(
                upper_button_position=2.578,
                lower_button_position=4.208,
                angular_position=0.000,
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


            # Ensure all arrays are the same length
            min_len = min(len(time), len(stream_velocity_z), len(stream_velocity_x), len(stream_velocity_y))
            time = time[:min_len]
            stream_velocity_z = stream_velocity_z[:min_len]
            stream_velocity_x = stream_velocity_x[:min_len]
            stream_velocity_y = stream_velocity_y[:min_len]
            altitude = altitude[:min_len] 

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

            # Pad with NaN after apogee
            v_vel_padded = np.where(valid_mask, v_vel, np.nan)
            v_acc_padded = np.where(valid_mask, v_acc, np.nan)
            t_vel_padded = np.where(valid_mask, t_vel, np.nan)
            alt_interp_padded = np.where(valid_mask, alt_interp, np.nan)

            row = [wind, temp, angle, apogee_alt, apogee_time]
            row += v_vel_padded.tolist()
            row += v_acc_padded.tolist()
            row += t_vel_padded.tolist()
            row += alt_interp_padded.tolist()
            all_rows.append(row)

# DataFrame and Save
df = pd.DataFrame(all_rows, columns=colnames)
df.to_csv("batch_dataset_v1.csv", index=False)
print("Simulation data saved as batch_dataset_v1.csv")

import numpy as np
import pandas as pd
from pathlib import Path
from rocketpy import Environment, SolidMotor, Rocket, Flight, TrapezoidalFins, NoseCone, Tail, Parachute
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

from rocket_parameter_loader import load_parameters


# Timesteps for interpolation (shared across all simulations)
TIMESTEPS = np.arange(0, 27.025, 0.025)
MAX_ALT = 5000  # Maximum altitude for wind profile (meters above ground)


def run_single_simulation(args):
    """
    Run a single rocket flight simulation with the given parameters.
    
    This function creates all RocketPy objects fresh to ensure process safety.
    Returns a row of data for the output DataFrame.
    """
    wind, temp, angle, params_dict, data_dir_str, tomorrow_tuple = args
    
    # Convert paths back from strings (needed for multiprocessing)
    data_dir = Path(data_dir_str)
    
    # Unpack cached parameters
    motor_params = params_dict["motors"]
    nosecone_params = params_dict["nosecones"]
    fins_params = params_dict["trapezoidal_fins"]["0"]
    tail_params = params_dict["tails"]["0"]
    rocket_params = params_dict["rocket"]
    rail_button_params = params_dict["rail_buttons"]
    env_params = params_dict["environment"]
    
    # Handle cross-platform path parsing
    from pathlib import PureWindowsPath
    thrust_source_name = PureWindowsPath(motor_params["thrust_source"]).name
    thrust_source_path = data_dir / thrust_source_name
    
    # Create motor (fresh for each simulation)
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
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation=motor_params["coordinate_system_orientation"],
    )
    
    # Create nose cone
    nosecone = NoseCone(
        length=nosecone_params["length"],
        kind=nosecone_params["kind"],
        base_radius=nosecone_params["base_radius"],
        rocket_radius=rocket_params["radius"],
        name=nosecone_params["name"],
    )
    
    # Create fins
    trapezoidal_fins = TrapezoidalFins(
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
    
    # Create tail
    tail = Tail(
        top_radius=tail_params["top_radius"],
        bottom_radius=tail_params["bottom_radius"],
        length=tail_params["length"],
        rocket_radius=rocket_params["radius"],
        name=tail_params["name"],
    )
    
    # Create environment
    env = Environment()
    env.set_location(latitude=env_params["latitude"], longitude=env_params["longitude"])
    env.set_elevation(env_params["elevation"])
    env.set_date(tomorrow_tuple)
    
    # Set custom atmospheric model with wind and temperature
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=float(temp),
        wind_u=[(0, float(wind)), (MAX_ALT, float(wind))],
        wind_v=[(0, 0), (MAX_ALT, 0)],
    )
    
    # Create rocket
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
        surfaces=[nosecone, trapezoidal_fins, tail],
        positions=[nosecone_params["position"], fins_params["position"], tail_params["position"]],
    )
    rocket.add_motor(motor, position=motor_params["position"])
    rocket.set_rail_buttons(
        upper_button_position=rail_button_params["upper_position"],
        lower_button_position=rail_button_params["lower_position"],
        angular_position=rail_button_params["angular_position"],
    )
    
    # Run flight simulation
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=4.318,
        inclination=90 - angle,
        heading=0,
        terminate_on_apogee=False,
        max_time=600,
    )
    
    # Extract flight data
    apogee_alt = flight.apogee - env.elevation
    apogee_time = flight.apogee_time
    valid_mask = TIMESTEPS <= apogee_time
    
    # Flight variables
    stream_velocity_z = flight.stream_velocity_z.y_array
    stream_velocity_x = flight.stream_velocity_x.y_array
    stream_velocity_y = flight.stream_velocity_y.y_array
    time_arr = flight.time
    stream_acc_v = flight.az.y_array
    altitude = flight.altitude.y_array
    pitch_angle = flight.theta.y_array
    dynamic_pressure = flight.dynamic_pressure.y_array
    mach_number = flight.mach_number.y_array
    pressure = flight.pressure.y_array
    
    # Ensure all arrays are the same length
    min_len = min(len(time_arr), len(stream_velocity_z), len(stream_velocity_x), 
                  len(stream_velocity_y), len(altitude), len(pitch_angle),
                  len(dynamic_pressure), len(mach_number), len(pressure))
    time_arr = time_arr[:min_len]
    stream_velocity_z = stream_velocity_z[:min_len]
    stream_velocity_x = stream_velocity_x[:min_len]
    stream_velocity_y = stream_velocity_y[:min_len]
    altitude = altitude[:min_len]
    pitch_angle = pitch_angle[:min_len]
    dynamic_pressure = dynamic_pressure[:min_len]
    mach_number = mach_number[:min_len]
    pressure = pressure[:min_len]
    
    # Interpolate to fixed timesteps
    v_vel = np.interp(TIMESTEPS, time_arr, stream_velocity_z)
    v_acc = np.interp(TIMESTEPS, time_arr, stream_acc_v[:min_len])
    alt_interp = np.interp(TIMESTEPS, time_arr, altitude)
    
    t_vel = np.interp(
        TIMESTEPS, time_arr,
        np.sqrt(stream_velocity_x**2 + stream_velocity_y**2 + stream_velocity_z**2)
    )
    h_vel = np.interp(
        TIMESTEPS, time_arr,
        np.sqrt(stream_velocity_x**2 + stream_velocity_y**2)
    )
    
    pitch_interp = np.interp(TIMESTEPS, time_arr, pitch_angle)
    dynp_interp = np.interp(TIMESTEPS, time_arr, dynamic_pressure)
    mach_interp = np.interp(TIMESTEPS, time_arr, mach_number)
    pres_interp = np.interp(TIMESTEPS, time_arr, pressure)
    
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
    
    # Build result row
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
    
    return row


def main():
    # Load parameters from JSON
    params = load_parameters()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "rocket-info"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sweep parameters
    wind_speeds = np.arange(0, 13, 1)  # 0-12 m/s
    temps = np.arange(280, 315, 2.5)   # 280-312.5 K
    launch_angles = np.arange(0, 8, 1) # 0-7 degrees
    
    # Calculate tomorrow's date tuple for environment
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_tuple = (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    
    # Column names
    colnames = [
        "Wind Speed (m/s)", "Temperature (K)", "Launch Angle (deg)", 
        "Apogee altitude (m)", "Apogee time (s)"
    ]
    colnames += [f"Vertical velocity (m/s) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Vertical acceleration (m/s^2) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Total velocity (m/s) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Altitude (m) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Horizontal velocity (m/s) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Pitch angle (deg) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Dynamic pressure (Pa) @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Mach number @ {t:.2f}s" for t in TIMESTEPS]
    colnames += [f"Pressure (Pa) @ {t:.2f}s" for t in TIMESTEPS]
    
    # Generate all parameter combinations
    param_combos = [
        (wind, temp, angle, params, str(data_dir), tomorrow_tuple)
        for wind in wind_speeds
        for temp in temps
        for angle in launch_angles
    ]
    
    total_sims = len(param_combos)
    num_workers = min(multiprocessing.cpu_count(), 12)  # Cap at 12 workers
    
    print(f"Running {total_sims} simulations using {num_workers} parallel workers...")
    start_time = time.time()
    
    all_rows = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_simulation, combo): combo for combo in param_combos}
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                row = future.result()
                all_rows.append(row)
                completed += 1
                
                # Progress update every 100 simulations
                if completed % 100 == 0 or completed == total_sims:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (total_sims - completed) / rate if rate > 0 else 0
                    print(f"  Progress: {completed}/{total_sims} ({100*completed/total_sims:.1f}%) - "
                          f"Est. remaining: {remaining:.0f}s")
            except Exception as e:
                combo = futures[future]
                print(f"  Simulation failed for wind={combo[0]}, temp={combo[1]}, angle={combo[2]}: {e}")
    
    elapsed_total = time.time() - start_time
    print(f"\nCompleted {completed} simulations in {elapsed_total:.1f}s ({completed/elapsed_total:.1f} sims/sec)")
    
    # Sort rows by wind, temp, angle to maintain consistent ordering
    all_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    
    # DataFrame and Save
    output_path = output_dir / "batch_dataset_v1.csv"
    df = pd.DataFrame(all_rows, columns=colnames)
    df.to_csv(output_path, index=False)
    print(f"Simulation data saved as {output_path}")


if __name__ == "__main__":
    main()

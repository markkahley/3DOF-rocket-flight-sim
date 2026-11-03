import matplotlib.pyplot as plt
import numpy as np
import yaml

#Motor Data Gathering
def load_motor_file(filepath):
    motor_times = []
    motor_thrusts = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue

        line_parts = line.split()

        if not header_found:
            prop_mass_kg = float(line_parts[4])
            total_mass_kg = float(line_parts[5])
            header_found = True
        else:
            motor_times.append(float(line_parts[0]))
            motor_thrusts.append(float(line_parts[1]))
    motor = {
        'prop_mass': prop_mass_kg,
        'total_mass': total_mass_kg,
        'dry_mass': total_mass_kg - prop_mass_kg,
        'burn_time': motor_times[-1],
        'times': np.array(motor_times),
        'thrusts': np.array(motor_thrusts)
    }
    return motor

#Loading Given Config
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

#Standard Atmosphere for Troposphere/Tropopause/Stratosphere
def get_air_density(altitude):
    temp_0km = 288.15 #K
    lapse_rate1 = -0.0065 #K/m
    lapse_rate2 = 0.001 #K/m
    pressure_0km = 101325 #Pa

    #Boundary Values between Troposphere and Tropopause
    temp_11km = temp_0km + lapse_rate1 * 11000
    pressure_11km = pressure_0km * (temp_11km/temp_0km) ** (-g / (R * lapse_rate1))

    # Boundary Values between Tropopause and Stratosphere
    temp_20km = temp_11km
    pressure_20km = pressure_11km * np.exp(-g * (20000 - 11000) / (R * temp_11km))

    if altitude < 11000:
        temp = temp_0km + lapse_rate1 * altitude
        pressure = pressure_0km * (temp/temp_0km) ** (-g / (R * lapse_rate1))
    elif altitude < 20000:
        temp = temp_11km
        pressure = pressure_11km * np.exp(-g * (altitude - 11000) / (R * temp_11km))
    else:
        temp = temp_20km + lapse_rate2 * (altitude - 20000)
        pressure = pressure_20km * (temp/temp_20km) ** (-g / (R * lapse_rate2))

    rho = pressure / (R * temp)
    return rho

#Main Trajectory Simulation
def run_simulation(config_file):

    #Load Config and Motor Data
    if isinstance(config_file, dict):
        config = config_file
    else:
        config = load_config(config_file)
    motor = load_motor_file(config['motor_file'])

    #Rocket Properties and Field Conditions from the Config File
    rocket_structure_mass = config['rocket']['structure_mass']
    Cd = config['rocket']['Cd']
    diameter = config['rocket']['diameter']
    angle_deg = config['launch_conditions']['angle_deg']
    launch_azimuth_deg = config['launch_conditions']['launch_azimuth_from_east']
    wind_speed = config['launch_conditions']['wind_speed']
    wind_azimuth_deg = config['launch_conditions']['wind_azimuth_from_east']

    dry_mass = rocket_structure_mass + motor['dry_mass'] #kg
    initial_prop_mass = motor['prop_mass'] #kg
    burn_time = motor['burn_time'] #s

    #Recovery Configuration
    recovery_type = config['recovery']['type']

    parachute_Cd = None
    parachute_area = None
    drogue_Cd = None
    drogue_area = None
    main_Cd = None
    main_area = None
    main_deploy_altitude = None

    if recovery_type == 'single':
        parachute_area = config['recovery']['parachute_area']
        parachute_Cd = config['recovery']['parachute_Cd']
    elif recovery_type == 'dual':
        drogue_area = config['recovery']['drogue_area']
        drogue_Cd = config['recovery']['drogue_Cd']
        main_area = config['recovery']['main_area']
        main_Cd = config['recovery']['main_Cd']
        main_deploy_altitude = config['recovery']['main_deploy_altitude']

    #Initial Conditions
    velocity_x = 0.0 #m/s
    velocity_y = 0.0 #m/s
    velocity_z = 0.0  # m/s
    position_x = 0.0 #m (Defined as distance in a Eastward direction)
    position_y = 0.0 #m (Defined as distance in vertical direction)
    position_z = 0.0 #m (Defined as distance in a Northward direction)
    time = 0.0 #s
    max_altitude_warning_shown = False
    apogee_reached = False
    main_deployed = False

    #Establishing Lists
    times = []
    positions_x = []
    positions_y = []
    positions_z = []
    velocities_x = []
    velocities_y = []
    velocities_z = []

    #Main Calculations
    angle_rad = np.radians(angle_deg)
    wind_azimuth_rad = np.radians(wind_azimuth_deg)
    launch_azimuth_rad = np.radians(launch_azimuth_deg)
    wind_speed_x = wind_speed * np.cos(wind_azimuth_rad)
    wind_speed_z = wind_speed * np.sin(wind_azimuth_rad)
    A = np.pi * (diameter / 2) ** 2   #m^2

    while position_y >= 0:
        times.append(time)
        positions_x.append(position_x)
        positions_y.append(position_y)
        positions_z.append(position_z)
        velocities_x.append(velocity_x)
        velocities_y.append(velocity_y)
        velocities_z.append(velocity_z)

        #Checking Recovery Deployment Status
        if not apogee_reached and velocity_y < 0:
            apogee_reached = True

        if recovery_type == 'dual' and apogee_reached and not main_deployed and position_y < main_deploy_altitude:
            main_deployed = True

        #Acceleration Calculations for Thrust and Gravity
        if time < burn_time:
            current_prop = initial_prop_mass - ((time / burn_time) * initial_prop_mass)
            current_mass = dry_mass + current_prop

            thrust_force = np.interp(time, motor['times'], motor['thrusts'])
            thrust_accel = thrust_force / current_mass
            thrust_horizontal = thrust_accel * np.sin(angle_rad)
            accel_x = thrust_horizontal * np.cos(launch_azimuth_rad)
            accel_z = thrust_horizontal * np.sin(launch_azimuth_rad)
            accel_y = thrust_accel * np.cos(angle_rad) - g
        else:
            current_mass = dry_mass
            accel_x = 0.0
            accel_y = -g
            accel_z = 0.0

        velocity_relative_x = velocity_x - wind_speed_x
        velocity_relative_y = velocity_y
        velocity_relative_z = velocity_z - wind_speed_z

        #Acceleration Adjustments for Drag
        speed = np.sqrt(velocity_relative_x**2 + velocity_relative_y**2 + velocity_relative_z**2)

        rho = get_air_density(position_y)
        if position_y > 30000 and not max_altitude_warning_shown:
            print(
                f"Warning: Rocket exceeded 30,000 m at t={time:.2f}s. This means the rocket is nearing the edge of the simulations "
                f"modeled atmosphere layers (32,000 m limit).")
            max_altitude_warning_shown = True

        if recovery_type == 'ballistic':
            current_Cd = Cd
            current_A = A
        elif recovery_type == 'single':
            current_Cd = parachute_Cd if apogee_reached else Cd
            current_A = parachute_area if apogee_reached else A
        elif recovery_type == 'dual':
            if main_deployed:
                current_Cd = main_Cd
                current_A = main_area
            elif apogee_reached:
                current_Cd = drogue_Cd
                current_A = drogue_area
            else:
                current_Cd = Cd
                current_A = A
        else:
            raise ValueError(f'Unknown recovery type: "{recovery_type}"')

        drag_force = 0.5 * rho * speed**2 * current_Cd * current_A
        drag_accel = drag_force / current_mass
        if speed > 0:
            accel_x -= drag_accel * (velocity_relative_x / speed)
            accel_y -= drag_accel * (velocity_relative_y / speed)
            accel_z -= drag_accel * (velocity_relative_z / speed)

        velocity_x = velocity_x + accel_x * dt
        velocity_y = velocity_y + accel_y * dt
        velocity_z = velocity_z + accel_z * dt
        position_x = position_x + velocity_x * dt
        position_y = position_y + velocity_y * dt
        position_z = position_z + velocity_z * dt
        time = time + dt

    #Calculating Apogee
    apogee = max(positions_y)
    apogee_index = positions_y.index(apogee)
    apogee_time = times[apogee_index]
    apogee_position_x = positions_x[apogee_index]
    apogee_position_z = positions_z[apogee_index]

    return {
        'apogee': apogee,
        'apogee_time': apogee_time,
        'apogee_position_x': apogee_position_x,
        'apogee_position_z': apogee_position_z,
        'positions_x': positions_x,
        'positions_y': positions_y,
        'positions_z': positions_z,
        'times': times,
    }

#Adds Randomness to the Base Config
def generate_random_config(base_config, rng):
    config = base_config.copy()
    config['rocket'] = base_config['rocket'].copy()
    config['launch_conditions'] = base_config['launch_conditions'].copy()

    #Establishing nominal values for the center of the spread
    nominal_wind = base_config['launch_conditions']['wind_speed']
    nominal_gust = base_config['launch_conditions']['wind_gust_speed']
    nominal_wind_azimuth = base_config['launch_conditions']['wind_azimuth_from_east']
    nominal_angle = base_config['launch_conditions']['angle_deg']
    nominal_launch_azimuth = base_config['launch_conditions']['launch_azimuth_from_east']
    nominal_Cd = base_config['rocket']['Cd']

    #Wind Randomization: mean = nominal, std = 1m/s (Truncated Normal)
    wind_std = max((nominal_gust - nominal_wind) / 3.0, 0.1)
    wind = rng.normal(loc=nominal_wind, scale=wind_std)
    wind = max(wind, 0.0)

    #Wind Azimuth: mean = nominal, std = 10 degrees (Normal)
    wind_azimuth = rng.normal(loc=nominal_wind_azimuth, scale=10.0)

    #Launch Angle: mean = nominal, std = 1 degrees (Normal)
    launch_angle = rng.normal(loc=nominal_angle, scale=1.0)

    # Launch Azimuth: mean = nominal, std = 1 degrees (Normal)
    launch_azimuth = rng.normal(loc=nominal_launch_azimuth, scale=1.0)

    # Cd: mean = nominal, std = 5% of nominal (Normal)
    Cd = rng.normal(loc=nominal_Cd, scale=0.05 * nominal_Cd)

    config['launch_conditions']['wind_speed'] = wind
    config['launch_conditions']['wind_azimuth_from_east'] = wind_azimuth
    config['launch_conditions']['angle_deg'] = launch_angle
    config['launch_conditions']['launch_azimuth_from_east'] = launch_azimuth
    config['rocket']['Cd'] = Cd

    return config

#Runs One Simulation of the Config and Prints/Plots Info
def run_one_simulation(config_file):
    result = run_simulation(config_file)

    apogee_index = result['positions_y'].index(result['apogee'])
    landing_index = -1

    print(f'--- Apogee ---')
    print(f'Time: {result['times'][apogee_index]:.2f} s')
    print(f'X (East): {result['positions_x'][apogee_index]:.1f} m')
    print(f'Y (Altitude): {result['apogee']:.1f} m')
    print(f'Z (North): {result['positions_z'][apogee_index]:.1f} m')

    print(f'--- Impact ---')
    print(f'Time: {result['times'][landing_index]:.2f} s')
    print(f'X (East): {result['positions_x'][landing_index]:.1f} m')
    print(f'Y (Altitude): {result['positions_y'][landing_index]:.1f} m')
    print(f'Z (North): {result['positions_z'][landing_index]:.1f} m')

    #3D Trajectory Plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(result['positions_x'], result['positions_z'], result['positions_y'])
    ax1.scatter(result['apogee_position_x'], result['positions_z'][result['positions_y'].index(result['apogee'])], result['apogee'], color='red', zorder=5, label=f'Apogee: {result["apogee"]:.1f} m')
    ax1.scatter(result['positions_x'][landing_index], result['positions_z'][landing_index], result['positions_y'][landing_index], color='green', zorder=5, label='Impact')

    x_range = max(result['positions_x']) - min(result['positions_x'])
    z_range = max(result['positions_z']) - min(result['positions_z'])
    horizontal_range = max(x_range, z_range, 1.0)
    x_mid = (max(result['positions_x']) + min(result['positions_x'])) / 2
    z_mid = (max(result['positions_z']) + min(result['positions_z'])) / 2
    ax1.set_xlim(x_mid - horizontal_range/2, x_mid + horizontal_range/2)
    ax1.set_ylim(z_mid - horizontal_range/2, z_mid + horizontal_range/2)

    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_zlabel("Altitude (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend()

    #Altitude vs Time Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(result['times'], result['positions_y'])
    ax2.scatter(result['times'][apogee_index], result['apogee'], color='red', zorder=5, label=f'Apogee: {result["apogee"]:.1f} m')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Altitude vs Time")
    ax2.grid(True)
    ax2.legend()

    #Top-Down Ground Track (East vs North)
    fig3, ax3 = plt.subplots()
    ax3.plot(result['positions_x'], result['positions_z'])
    ax3.scatter(result['positions_x'][landing_index], result['positions_z'][landing_index], color='green', zorder=5, label='Impact')
    ax3.set_xlabel("East (m)")
    ax3.set_ylabel("North (m)")
    ax3.set_title("Ground Track (Top-Down View)")
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.legend()

    plt.show()

    return result

#Rus Monte Carlo Analysis on the Config
def run_monte_carlo(base_config, n_runs):
    rng = np.random.default_rng(seed)

    apogees = []
    landing_positions_x = []
    landing_positions_z = []

    for i in range(n_runs):
        trial_config = generate_random_config(base_config, rng)
        result = run_simulation(trial_config)

        apogees.append(result['apogee'])
        landing_positions_x.append(result['positions_x'][-1])
        landing_positions_z.append(result['positions_z'][-1])

    apogees = np.array(apogees)
    landing_positions_x = np.array(landing_positions_x)
    landing_positions_z = np.array(landing_positions_z)

    # Summary Statistics
    apogee_mean = np.mean(apogees)
    apogee_std = np.std(apogees)
    landing_x_mean = np.mean(landing_positions_x)
    landing_x_std = np.std(landing_positions_x)
    landing_z_mean = np.mean(landing_positions_z)
    landing_z_std = np.std(landing_positions_z)

    print(f"--- Monte Carlo Results (n={n_runs}) ---")
    print(f"Apogee: {apogee_mean:.1f} m ± {apogee_std:.1f} m")
    print(f"Landing East (x): {landing_x_mean:.1f} m ± {landing_x_std:.1f} m")
    print(f"Landing North (z): {landing_z_mean:.1f} m ± {landing_z_std:.1f} m")

    #Landing Dispersion Plot
    fig, ax = plt.subplots()
    ax.scatter(landing_positions_x, landing_positions_z, alpha=0.3, color='blue', label=f'Landings (n={n_runs})')
    ax.scatter(landing_x_mean, landing_z_mean, color='red', marker='x', s=100, zorder=6, label='Mean Landing')
    ax.scatter(0, 0, color='red', marker='^', zorder=5, label='Launch Point')

    x_range = max(landing_positions_x) - min(landing_positions_x)
    z_range = max(landing_positions_z) - min(landing_positions_z)
    horizontal_range = max(x_range, z_range, 1.0)
    horizontal_range = horizontal_range * 1.1  # 10% padding for edges
    x_mid = (max(landing_positions_x) + min(landing_positions_x)) / 2
    z_mid = (max(landing_positions_z) + min(landing_positions_z)) / 2
    ax.set_xlim(x_mid - horizontal_range / 2, x_mid + horizontal_range / 2)
    ax.set_ylim(z_mid - horizontal_range / 2, z_mid + horizontal_range / 2)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Monte Carlo Landing Dispersion (n={n_runs})")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.show()

    return {
        'apogees': apogees,
        'landing_positions_x': landing_positions_x,
        'landing_positions_z': landing_positions_z,
    }

#Constants
g = 9.81 #m/s^2
dt = 0.01 #s
R = 287.05  # J/(kg*K)
seed = 4

#CHOSEN CONFIG FILE ***CHANGE THIS TO WHATEVER .yaml FILE YOU WANT ANALYZED***
#CONFIG_FILE = 'configs/PLUS_ULTRA_1.0.yaml'
#CONFIG_FILE = 'configs/DUAL_DEPLOYMENT_TEST.yaml'
#CONFIG_FILE = 'configs/BALLISTIC_TEST.yaml'
#CONFIG_FILE = 'configs/RED_demo.yaml'
CONFIG_FILE = 'configs/ICARUS_demo.yaml'

if __name__ == "__main__":
    result = run_one_simulation(CONFIG_FILE)

    base_config = load_config(CONFIG_FILE)
    run_monte_carlo(base_config, 75)
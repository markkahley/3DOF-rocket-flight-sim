from rocket_sim import run_simulation, get_air_density

def test_plus_ultra_apogee():
    result = run_simulation('configs/PLUS_ULTRA_1.0.yaml')
    assert abs(result['apogee'] - 373) < 5

def test_air_density_sea_level():
    assert abs(get_air_density(0) - 1.225) < 0.01

def test_air_density_boundaries():
    below_11km = get_air_density(10999)
    above_11km = get_air_density(11001)
    assert abs(below_11km - above_11km) < 0.1
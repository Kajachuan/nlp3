from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def write_catalog(filename: str, items: list[dict]) -> None:
    if len(items) != 50:
        raise ValueError(f"{filename} must contain exactly 50 entries, got {len(items)}")
    path = RAW_DIR / filename
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stock(index: int, base: int = 100) -> int:
    return base + ((index * 137) % 4800)


def price(index: int, base: float, step: float, mod: int = 17) -> float:
    return round(base + step * (index % mod), 4)


def resistors() -> list[dict]:
    e24 = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
    multipliers = [1, 10, 100]
    packages = ["0402", "0603", "0805", "1206"]
    items = []
    for i in range(50):
        value = e24[i % len(e24)] * multipliers[(i // len(e24)) % len(multipliers)]
        package = packages[i % len(packages)]
        tolerance = "1%" if i % 3 else "0.1%"
        power = {"0402": 0.0625, "0603": 0.1, "0805": 0.125, "1206": 0.25}[package]
        label = f"{value:g} ohm" if value < 1000 else f"{value / 1000:g} kOhm"
        items.append(
            {
                "sku": f"R-E24-{value:g}-{package}-{tolerance.replace('%', 'P').replace('.', 'D')}-{i + 1:02d}",
                "name": f"Resistencia SMD E24 {label} {package} {tolerance}",
                "category": "resistor",
                "description": "Resistencia SMD de la serie E24 para polarizacion, divisores, pull-up, pull-down y limitacion de corriente.",
                "specs": {
                    "series": "E24",
                    "resistance_ohm": value,
                    "tolerance": tolerance,
                    "power_w": power,
                    "package": package,
                    "temperature_coefficient_ppm_c": 100 if tolerance == "1%" else 25,
                },
                "stock": stock(i, 900),
                "price_usd": price(i, 0.006, 0.001),
                "recommended_use": "Seleccion rapida de valores estandar para prototipos, redes analogicas y entradas digitales.",
            }
        )
    return items


def smd_ceramic_capacitors() -> list[dict]:
    values = [
        ("1 pF", 1e-12),
        ("2.2 pF", 2.2e-12),
        ("4.7 pF", 4.7e-12),
        ("10 pF", 1e-11),
        ("22 pF", 2.2e-11),
        ("47 pF", 4.7e-11),
        ("100 pF", 1e-10),
        ("220 pF", 2.2e-10),
        ("470 pF", 4.7e-10),
        ("1 nF", 1e-9),
        ("2.2 nF", 2.2e-9),
        ("4.7 nF", 4.7e-9),
        ("10 nF", 1e-8),
        ("22 nF", 2.2e-8),
        ("47 nF", 4.7e-8),
        ("100 nF", 1e-7),
        ("220 nF", 2.2e-7),
        ("470 nF", 4.7e-7),
        ("1 uF", 1e-6),
        ("2.2 uF", 2.2e-6),
        ("4.7 uF", 4.7e-6),
        ("10 uF", 1e-5),
        ("22 uF", 2.2e-5),
        ("47 uF", 4.7e-5),
        ("100 uF", 1e-4),
    ]
    packages = ["0402", "0603", "0805", "1206"]
    dielectrics = ["C0G", "X7R", "X5R", "Y5V"]
    voltages = [6.3, 10, 16, 25, 50]
    items = []
    for i in range(50):
        label, farads = values[i % len(values)]
        package = packages[i % len(packages)]
        dielectric = dielectrics[i % len(dielectrics)]
        voltage = voltages[i % len(voltages)]
        items.append(
            {
                "sku": f"C-SMD-{label.replace(' ', '').replace('.', 'D')}-{dielectric}-{package}-{str(voltage).replace('.', 'D')}V-{i + 1:02d}",
                "name": f"Capacitor ceramico SMD {label} {dielectric} {package} {voltage} V",
                "category": "capacitor",
                "description": "Capacitor ceramico SMD para desacople, filtrado, temporizacion y redes de senal.",
                "specs": {
                    "capacitance_f": farads,
                    "voltage_v": voltage,
                    "dielectric": dielectric,
                    "package": package,
                    "mounting": "SMD",
                },
                "stock": stock(i, 1200),
                "price_usd": price(i, 0.01, 0.004),
                "recommended_use": "Desacople cerca de integrados, filtros pasivos y estabilizacion de alimentacion.",
            }
        )
    return items


def through_hole_electrolytics() -> list[dict]:
    values = ["1 uF", "2.2 uF", "4.7 uF", "10 uF", "22 uF", "47 uF", "100 uF", "220 uF", "470 uF", "1000 uF"]
    farads = [1e-6, 2.2e-6, 4.7e-6, 1e-5, 2.2e-5, 4.7e-5, 1e-4, 2.2e-4, 4.7e-4, 1e-3]
    voltages = [6.3, 10, 16, 25, 35, 50, 63, 100]
    pitches = [2.0, 2.5, 3.5, 5.0, 7.5]
    items = []
    for i in range(50):
        idx = i % len(values)
        voltage = voltages[i % len(voltages)]
        pitch = pitches[i % len(pitches)]
        items.append(
            {
                "sku": f"CE-TH-{values[idx].replace(' ', '')}-{str(voltage).replace('.', 'D')}V-P{str(pitch).replace('.', 'D')}-{i + 1:02d}",
                "name": f"Capacitor electrolitico through hole {values[idx]} {voltage} V",
                "category": "capacitor",
                "description": "Capacitor electrolitico radial through hole para filtrado de fuente, reserva de energia y suavizado de ripple.",
                "specs": {
                    "capacitance_f": farads[idx],
                    "voltage_v": voltage,
                    "mounting": "through_hole",
                    "lead_pitch_mm": pitch,
                    "type": "aluminum_electrolytic",
                },
                "stock": stock(i, 300),
                "price_usd": price(i, 0.04, 0.018),
                "recommended_use": "Filtrado de entrada/salida en reguladores, fuentes DC y etapas de potencia de baja frecuencia.",
            }
        )
    return items


def connectors_and_headers() -> list[dict]:
    families = ["pin_header", "socket_header", "jst_ph", "jst_xh", "dupont", "usb", "terminal_block", "fpc"]
    pitches = [1.0, 1.25, 2.0, 2.54, 3.5, 5.08]
    items = []
    for i in range(50):
        family = families[i % len(families)]
        pins = 2 + (i % 24)
        pitch = pitches[i % len(pitches)]
        gender = "male" if i % 2 == 0 else "female"
        angle = "right_angle" if i % 5 == 0 else "straight"
        items.append(
            {
                "sku": f"CONN-{family.upper()}-{pins}P-{str(pitch).replace('.', 'D')}MM-{gender[:1].upper()}-{angle[:2].upper()}-{i + 1:02d}",
                "name": f"Conector {family.replace('_', ' ')} {pins} pines {pitch} mm {gender} {angle}",
                "category": "connector",
                "description": "Conector o header para interconexion de placas, sensores, cables y modulos de prototipado.",
                "specs": {
                    "family": family,
                    "pins": pins,
                    "pitch_mm": pitch,
                    "gender": gender,
                    "orientation": angle,
                    "mounting": "through_hole" if family not in {"usb", "fpc"} else "smd",
                },
                "stock": stock(i, 250),
                "price_usd": price(i, 0.05, 0.025),
                "recommended_use": "Conexion desmontable para prototipos, buses de senal, alimentacion auxiliar y placas de expansion.",
            }
        )
    return items


def analog_devices_ics() -> list[dict]:
    bases = ["AD620", "AD8226", "AD8421", "AD8605", "AD8628", "AD5683R", "AD7190", "ADuM1201", "ADG704", "LTC6655"]
    functions = ["instrumentation_amplifier", "precision_op_amp", "adc", "dac", "digital_isolator", "analog_switch", "voltage_reference"]
    packages = ["SOIC-8", "MSOP-8", "TSSOP-16", "LFCSP", "SOT-23-5"]
    items = []
    for i in range(50):
        base = bases[i % len(bases)]
        function = functions[i % len(functions)]
        package = packages[i % len(packages)]
        channels = 1 + (i % 4)
        items.append(
            {
                "sku": f"ADI-{base}-{package}-{channels}CH-{i + 1:02d}",
                "name": f"Analog Devices {base} {function.replace('_', ' ')} {package}",
                "category": "integrated_circuit",
                "description": "Integrado Analog Devices para medicion analogica, precision, conversion de datos o aislacion de senales.",
                "specs": {
                    "manufacturer": "Analog Devices",
                    "part_family": base,
                    "function": function,
                    "channels": channels,
                    "package": package,
                    "supply_v": "2.7-5.5" if i % 2 else "3.0-15",
                },
                "stock": stock(i, 40),
                "price_usd": price(i, 1.2, 0.35),
                "recommended_use": "Front-end analogico, adquisicion de datos, instrumentacion, aislacion o referencias de precision.",
            }
        )
    return items


def texas_instruments_ics() -> list[dict]:
    bases = ["LM358", "TLV9002", "INA219", "ADS1115", "DAC8562", "TPS62160", "DRV8833", "SN74LVC1T45", "LMV321", "BQ24075"]
    functions = ["op_amp", "current_monitor", "adc", "dac", "buck_regulator", "motor_driver", "level_translator", "battery_charger"]
    packages = ["SOIC-8", "VSSOP-10", "SOT-23-5", "QFN-16", "TSSOP-14"]
    items = []
    for i in range(50):
        base = bases[i % len(bases)]
        function = functions[i % len(functions)]
        package = packages[i % len(packages)]
        items.append(
            {
                "sku": f"TI-{base}-{package}-{i + 1:02d}",
                "name": f"Texas Instruments {base} {function.replace('_', ' ')} {package}",
                "category": "integrated_circuit",
                "description": "Integrado Texas Instruments para procesamiento analogico, potencia, conversion de datos, drivers o logica.",
                "specs": {
                    "manufacturer": "Texas Instruments",
                    "part_family": base,
                    "function": function,
                    "package": package,
                    "supply_v": "1.8-5.5" if i % 3 else "4.5-17",
                    "interface": "I2C" if function in {"current_monitor", "adc", "dac", "battery_charger"} else "analog/gpio",
                },
                "stock": stock(i, 55),
                "price_usd": price(i, 0.45, 0.28),
                "recommended_use": "Diseno de fuentes, adquisicion, control de motores, sensado de corriente y acondicionamiento analogico.",
            }
        )
    return items


def bosch_bm_sensors() -> list[dict]:
    bases = ["BMA400", "BMA456", "BME280", "BME680", "BME688", "BMI160", "BMI270", "BMP280", "BMP388", "BMP390"]
    functions = ["accelerometer", "environmental_sensor", "pressure_sensor", "imu", "gas_environmental_sensor"]
    interfaces = ["I2C", "SPI", "I2C/SPI"]
    packages = ["LGA-10", "LGA-12", "LGA-14", "LGA-16"]
    items = []
    for i in range(50):
        base = bases[i % len(bases)]
        function = functions[i % len(functions)]
        interface = interfaces[i % len(interfaces)]
        package = packages[i % len(packages)]
        items.append(
            {
                "sku": f"BOSCH-{base}-{interface.replace('/', '')}-{package}-{i + 1:02d}",
                "name": f"Bosch Sensortec {base} {function.replace('_', ' ')}",
                "category": "sensor",
                "description": "Sensor Bosch de la linea BM/BMP/BME para movimiento, presion, ambiente o gas en equipos embebidos.",
                "specs": {
                    "manufacturer": "Bosch Sensortec",
                    "part_family": base,
                    "function": function,
                    "interface": interface,
                    "package": package,
                    "supply_v": "1.71-3.6",
                },
                "stock": stock(i, 70),
                "price_usd": price(i, 1.1, 0.22),
                "recommended_use": "Wearables, estaciones ambientales, navegacion inercial, monitoreo de presion y sensores IoT.",
            }
        )
    return items


def mcu_boards() -> list[dict]:
    families = ["Arduino", "STM32", "ESP32", "ESP8266", "RP2040"]
    boards = ["UNO R3", "Nano", "Mega 2560", "STM32F103 Blue Pill", "STM32F401 Black Pill", "ESP32 DevKitC", "ESP32-S3 DevKit", "ESP8266 NodeMCU", "RP2040 Pico", "Arduino MKR WiFi 1010"]
    wireless = ["none", "none", "none", "none", "none", "WiFi/Bluetooth", "WiFi/Bluetooth", "WiFi", "none", "WiFi"]
    items = []
    for i in range(50):
        board = boards[i % len(boards)]
        family = families[i % len(families)]
        flash_mb = [0.032, 0.256, 1, 2, 4, 8, 16][i % 7]
        gpio = [14, 20, 34, 48, 54][i % 5]
        items.append(
            {
                "sku": f"MCU-{family.upper()}-{board.upper().replace(' ', '-').replace('/', '')}-{i + 1:02d}",
                "name": f"Placa MCU {board}",
                "category": "microcontroller",
                "description": "Placa de desarrollo MCU para prototipado, control embebido, IoT, sensores y educacion tecnica.",
                "specs": {
                    "line": family,
                    "board": board,
                    "gpio": gpio,
                    "flash_mb": flash_mb,
                    "wireless": wireless[i % len(wireless)],
                    "supply_v": "3.3" if family in {"STM32", "ESP32", "ESP8266", "RP2040"} else "5",
                },
                "stock": stock(i, 35),
                "price_usd": price(i, 3.5, 0.65),
                "recommended_use": "Prototipos rapidos, adquisicion de datos, control de actuadores, conectividad IoT y pruebas de firmware.",
            }
        )
    return items


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    catalogs = {
        "resistors_e24_values.json": resistors(),
        "smd_ceramic_capacitors.json": smd_ceramic_capacitors(),
        "through_hole_electrolytic_capacitors.json": through_hole_electrolytics(),
        "connectors_and_headers.json": connectors_and_headers(),
        "analog_devices_integrated_circuits.json": analog_devices_ics(),
        "texas_instruments_integrated_circuits.json": texas_instruments_ics(),
        "bosch_bm_sensors.json": bosch_bm_sensors(),
        "mcu_arduino_stm_esp.json": mcu_boards(),
    }
    for filename, items in catalogs.items():
        write_catalog(filename, items)
        print(f"Wrote {filename}: {len(items)} entries")


if __name__ == "__main__":
    main()

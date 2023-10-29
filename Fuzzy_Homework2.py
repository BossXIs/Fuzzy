import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# กำหนด อินพุตและเอาท์ของระบบคือ คุณภาพของอากาศ, ความชื้น, และระดับการทำงานของเครื่องฟอกอากาศ
air_quality = ctrl.Antecedent(np.linspace(0, 100, 101), 'air_quality')
humidity = ctrl.Antecedent(np.linspace(0, 100, 101), 'humidity')
fan_speed = ctrl.Consequent(np.linspace(0, 100, 101), 'fan_speed')

# กำหนด membership functions ของคุณภาพอากาศ
air_quality['poor'] = fuzz.trapmf(air_quality.universe, [0, 0, 20, 40])
air_quality['moderate'] = fuzz.trimf(air_quality.universe, [30, 50, 70])
air_quality['good'] = fuzz.trapmf(air_quality.universe, [60, 80, 100, 100])

# กำหนด membership functions ของความชื้น
humidity['underrate'] = fuzz.trapmf(humidity.universe, [0, 0, 10, 32])
humidity['good'] = fuzz.trimf(humidity.universe, [30, 40, 50])
humidity['overrate'] = fuzz.trapmf(humidity.universe, [48, 70, 100, 100])

# กำหนด membership functions ของระดับการทำงานของเครื่องฟอกอากาศ
fan_speed['low'] = fuzz.trapmf(fan_speed.universe, [0, 0, 20, 40])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [20, 50, 70])
fan_speed['high'] = fuzz.trapmf(fan_speed.universe, [60, 85, 100, 100])

# กำหนดความสัมพันธ์ระหว่างคุณภาพอากาศ, ความชื้น, และ ระดับการทำงานของเครื่องฟอกอากาศ โดยใช้กฎ fuzzy
Rule_1 = ctrl.Rule(air_quality['poor'] & humidity['underrate'], fan_speed['high'])
Rule_2 = ctrl.Rule(air_quality['poor'] & humidity['good'], fan_speed['medium'])
Rule_3 = ctrl.Rule(air_quality['poor'] & humidity['overrate'], fan_speed['high'])
Rule_4 = ctrl.Rule(air_quality['moderate'] & humidity['underrate'], fan_speed['medium'])
Rule_5 = ctrl.Rule(air_quality['moderate'] & humidity['good'], fan_speed['low'])
Rule_6 = ctrl.Rule(air_quality['moderate'] & humidity['overrate'], fan_speed['medium'])
Rule_7 = ctrl.Rule(air_quality['good'] & humidity['underrate'], fan_speed['low'])
Rule_8 = ctrl.Rule(air_quality['good'] & humidity['good'], fan_speed['low'])
Rule_9 = ctrl.Rule(air_quality['good'] & humidity['overrate'], fan_speed['medium'])


# เพิ่มกฎ fuzzy ลงในระบบโดยใช้ Mamdani-style
fuzzy = ctrl.ControlSystem([Rule_1, Rule_2, Rule_3, Rule_4, Rule_5, Rule_6, Rule_7, Rule_8, Rule_9])

# สร้างระบบ Fuzzy Logic โดยใช้กฎที่กำหนดไว้
fuzzy = ctrl.ControlSystemSimulation(fuzzy)

# Set inputs สำหรับ air quality and humidity
input_air_quality = 80 #ปรับค่าได้ (0-100)
input_humidity = 50 #ปรับค่าได้ (0-100)

# กำหนดอินพุทในระบบ Fuzzy Logic
fuzzy.input['air_quality'] = input_air_quality
fuzzy.input['humidity'] = input_humidity

# ประมวลผลระบบ Fuzzy
fuzzy.compute()

# แสดงผลลัพธ์ 
print(f"Air Quality: {input_air_quality}")
print(f"Percentage Humidity: {input_humidity}")
print("Fan Speed: ", fuzzy.output['fan_speed'])

air_quality.view(sim=fuzzy)
humidity.view(sim=fuzzy)
fan_speed.view(sim=fuzzy)

plt.show()

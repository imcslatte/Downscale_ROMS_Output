import yaml

print('FORECAST CONFIG')
# Reading YAML from a file
with open('C:\\work\\roms\\NOPP\\forecast\\forecast_config.yml', 'r') as file:
#with open('./test.yml', 'r') as file:
    fconfig = yaml.safe_load(file)

print(fconfig)
# Accessing values
#value = data['key']

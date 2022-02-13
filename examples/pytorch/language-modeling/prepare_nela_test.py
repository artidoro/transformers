with open('/data/artidoro/data/nela-covid-2020-valid.json') as infile:
    lines = [line.strip() for line in infile.readlines()]

valid = lines[:3000]
test = lines[3000:]

with open('/data/artidoro/data/nela-covid-2020-valid.json', 'w') as outfile:
    for line in valid:
        outfile.write(f'{line}\n')

with open('/data/artidoro/data/nela-covid-2020-test.json', 'w') as outfile:
    for line in test:
        outfile.write(f'{line}\n')
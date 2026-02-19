import os
import time
import subprocess
import re
import argparse

os.system('clear')

parser = argparse.ArgumentParser(description='Track and display the CPU temperature')
parser.add_argument('-i','--interval', help='Set the refresh rate in seconds', type=int, default=1)
args = parser.parse_args()

	
#gets the array with all of the logged readings and get min, max and averages them
def calculateResults(resultsArray):
	maxReading = max(resultsArray)
	minReading = min(resultsArray)	
	average = float("{:.1f}".format(sum(resultsArray)/len(resultsArray)))
	return({'max': maxReading, 'min': minReading, 'average': average})

def printTempResults(tempArray):
	results = calculateResults(tempArray)
	print("Average CPU Temperature: ", results['average'])
	print("Max CPU Temperature: ", results['max'])
	print("Min CPU Temperature: ", results['min'])

def printFanResults(fanArray):
	results = calculateResults(fanArray)
	print("Average Fan Speed: ", results['average'])
	print("Max Fan Speed: ", results['max'])
	print("Min Fan Speed: ", results['min'])


class System:
	tempReadings = []
	fanReadings = []

	@classmethod
	def getRawData(cls):
		#getting the output of the 'sensors' command 
		return str(subprocess.check_output('sensors'))

	@classmethod
	def parseCpuTemp(cls,rawData):
		#parsing out the CPU temperature using regex, output looks like: Core X:    +55.55°C
		pattern = re.compile(r'(?i)core...\s+\+(\d+\.\d)')
		match = pattern.search(rawData)
		#converting it into a float for further calculations
		return float(match.group(1))

	@classmethod
	def parseFanSpeed(cls,rawData):
		#pattern = re.compile(r'(?i)fan.+(\d+)')
		pattern = re.compile(r'(\d+) RPM')
		match = pattern.search(rawData)
		return int(match.group(1))

	@classmethod
	def getCurrentData(cls):
		#Returns a dictionary with the parsed data
		rawData = cls.getRawData()
		currentData = {
			'cpuTemp': cls.parseCpuTemp(rawData),
			'fanSpeed': cls.parseFanSpeed(rawData)
		}
		return currentData

	@classmethod
	def writeData(cls,currentData):
		cls.tempReadings.append(currentData['cpuTemp'])
		cls.fanReadings.append(currentData['fanSpeed'])

def main():
    while True:
        current_data = System.getCurrentData()
        print(current_data)
        time.sleep(args.interval)

if __name__ == "__main__":
	main()
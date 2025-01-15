
class Holder:

    def __init__(self, name):
        print("Generated Holder")

        self.name = name
        self.samples = []

    def addSample(self, sample):
        self.samples.append(sample)

    def removeSample(self, sampleIndex):
        self.samples.pop(sampleIndex)

    def getSampleList(self):
        return self.samples

class Sample:

    def __init__(self,name, temperature=0,angles=[], time=0, process="None", interface=None):
        print("Generated Sample")

        self.name = name
        self.temperature = temperature
        self.angles = angles
        self.time = time
        self.process = process
        
        self.interface = interface

        if(self.interface != None):
            command = "$SampleList.append(Sample(" + "'" + self.name + "'" + "))"
            self.interface.executeCommand(command)

    def changeTemperature(self, temperature):
        self.temperature = temperature

    def changeAngles(self, anglesq):
        self.angles = angles


    def getData(self):
        return [self.name, self.temperature, self.angles]
from ardSerial import *

goodPorts = {}
connectPort(goodPorts)

task = ["ksit", 1]
send(goodPorts, task)

import serial

file=open('file.txt','r')
f=file.readlines()
#print(f)
newlist=[]
buf=[]
x=[]
y=[]
z=[]
for line in f[1:]:
    line=line.rstrip('\n')
    buf=line.split(' ')
    x.append(0.15+int(buf[0])/3000)
    y.append(int(buf[1])/3000)
    z.append(-0.01*int(buf[2]))


ports=serial.tools.list_ports.comports()
serialInst=serial.Serial()

portsList=[]

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

val=input("Select Port: COM")
portVar="COM" +str(val)
print(portVar)

serialInst.baudrate=9600
serialInst.port=portVar
serialInst.open()

for set in [x,y,z]:
     command=set[0]
     serialInst.write(command.encode('float'))
     command=set[1]
     serialInst.write(command.encode('float'))
     command=set[2]
     serialInst.write(command.encode('float'))
     
     while(serialInst.in_waiting()==0):
          continue
     response=serialInst.read()
     if(response.decode('Ascii')!='a'):
          set=set-1

   

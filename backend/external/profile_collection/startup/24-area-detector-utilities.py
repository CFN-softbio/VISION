#TODO: integrate PV to turn on the power supply. ##'XF:11BMB-VA{Chm:Det}UserButton'

print(f'Loading {__file__}')

import telnetlib
import paramiko
# import numpy as np
# SSH connection details
# hostname = 'xf11bm-pilatus800k2'
hostname = 'xf11bm-pilatus800k'
username = 'det'
password = 'Pilatus2'
# Create an SSH client
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())



telnet_command = 'telnet localhost 20002'

def restartWAXS():


    try:
        # Connect to the SSH server
        client.connect(hostname=hostname,username=username, password=password)


        command_to_run = './start_camserver'  # Replace this with your desired command
        stdin, stdout, stderr = client.exec_command(command_to_run)
    #    output = stdout.read().decode()
    #    print(output)
        # Start an interactive shell session
        ssh_shell = client.invoke_shell()

        # Send the Telnet command to the shell
        ssh_shell.send(telnet_command + '\n')

        # Create a Telnet session on the SSH server
        tn = telnetlib.Telnet()

        # Attach the shell transport to the Telnet session
        tn.sock = ssh_shell

        ssh_shell.send('\x18\x18')
        # Read and monitor the output of the Telnet command

        start_time=time.time()

        while time.time()-start_time<30:
            output = tn.read_very_eager().decode()
            if output:
                print(output, end='')

            # Check if the Telnet connection is closed
            if tn.eof:
                break
            # if time.time()-start_time>3:
            #     break
        
        # Close the SSH connection
        client.close()

    finally:
        # Close the SSH connection
        client.close()

#restartWAXS after pumping the vacuum below 0.5mbar
def startWAXS():

    caput(' XF:11BMB-VA{Chm:Det}UserButton', 1)

    #telnet and restart camserver
    restartWAXS()
    #set exposure time twice to garantee the EPICS connection
    pilatus800.cam.acquire_time.set(1.3)
    time.sleep(3)
    pilatus800.cam.acquire_time.set(1.7)



        

# try:
#      # Connect to the SSH server
#     client.connect(hostname=hostname,username=username, password=password)


#     command_to_run = './start_camserver'  # Replace this with your desired command
#     stdin, stdout, stderr = client.exec_command(command_to_run)
# #    output = stdout.read().decode()
# #    print(output)
#     # Start an interactive shell session
#     ssh_shell = client.invoke_shell()

#     # Send the Telnet command to the shell
#     ssh_shell.send(telnet_command + '\n')

#     # Create a Telnet session on the SSH server
#     tn = telnetlib.Telnet()

#     # Attach the shell transport to the Telnet session
#     tn.sock = ssh_shell

#     ssh_shell.send('\x18\x18')
#     # Read and monitor the output of the Telnet command

#     start_time=time.time()

#     while time.time()-start_time<30:
#         output = tn.read_very_eager().decode()
#         if output:
#             print(output, end='')

#         # Check if the Telnet connection is closed
#         if tn.eof:
#             break
#         # if time.time()-start_time>180:
#         #     command_to_close = './killall telnet'  # Replace this with your desired command
#         #     stdin, stdout, stderr = client.exec_command(command_to_close)
#         #     output=stdout.read().decode()
#         #     print(output)
#             # tn.write(b'x1d')D
#         # if time.time()-start_time>3:
#         #     break
    
#     tn.close()

# finally:
#     # Close the SSH connection
#     # client.close()
#     client.close()

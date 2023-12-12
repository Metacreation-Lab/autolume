from pythonosc import udp_client

# Replace these with your specific values
target_ip = "127.0.0.1"  # Replace with the IP of the receiving OSC application
target_port = 1338  # Replace with the port the receiving application is listen>
address = "/test"  # Replace with the OSC address you want to use

# Create a list of 512 floats
vector_of_floats = [float(i) for i in range(1, 513)]

#Send a boolean
b = False

#Send string
s = '/a'

# Create an OSC client
client = udp_client.SimpleUDPClient(target_ip, target_port)

# Send the 512-length vector of floats
client.send_message(address, vector_of_floats)

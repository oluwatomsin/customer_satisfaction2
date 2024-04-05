import subprocess
from pyngrok import ngrok


port = 1234  # Replace 1234 with the desired port number
process = subprocess.Popen(['zenml', 'up', '--blocking', '--port', str(port)])

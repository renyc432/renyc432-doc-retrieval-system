    
import socket
import threading
import socketserver
import time

import http.server

HOST, PORT = "localhost", 8080

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()


# =============================================================================
# 
# class MyTCPHandler(socketserver.BaseRequestHandler):
# 
#     
# # =============================================================================
# #     def __init__(self, host, opt):
# #         self.environment = {}
# #         self.environment['NoMode'] = {'points' : 0}
# #         self.environment['Occupancy'] = {'occupancy' : 0, 'points' : 0}
# #         self.host = host
# #         self.port = opt.port
# #         self.opt = opt
# #         self.state = self.environment[opt.mode if opt.mode else 'NoMode']
# #         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #         self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# #         self.sock.bind((self.host, self.port))
# #         self.lock = threading.Lock()
# # =============================================================================
#         
#     #def __init__(self):
#         
#         
#         
#     def handle(self):
#         # self.request is the TCP socket connected to the client
#         # if data size is larger than 1024, then call recv() multiple times
#         #self.data = self.request.recv(1024).strip()
#         #print("{} wrote:".format(self.client_address[0]))
#         #print(self.data)
#     
#         # just send back the same data, but upper-cased
# 
#         
#         for i in range(10):
#             number_byte=bytes(str(i),"utf-8")
#             self.request.sendall(number_byte)
#             time.sleep(10)
#             
#             
#     def display_answer(self):
#         pass
#             
#             
# with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
#     # Activate the server; this will keep running until you
#     # interrupt the program with Ctrl-C
#     print('serving at port', PORT)
#     server.serve_forever()
# =============================================================================

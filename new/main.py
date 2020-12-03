import spines
import book_getter
import socket

# Full pipeline
def run():
    # TODO: INTEGRATE WITH APP HERE! WILL JUST HARDCODE FOR NOW
    imgpath = '../data/example5.jpg'
    
    # Get names
    names = spines.run(imgpath)
    
    # For each book, return isbn and name
    for n in names:
        pro = book_getter.run(n)
        json = pro[0]
        old_isbn = pro[1]
        # Catch error
        if 'errorMessage' in json:
            print('Error - {} isbn {} not found'.format(json['errorMessage'], old_isbn))
        else:
            name = json['book']['title']
            author = json['book']['authors'][0]
            isbn = json['book']['isbn']
            
            print('{} - {} isbn: {}'.format(name, author, isbn))
 
# Networking
def net():
    s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    port = 12345                # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port
    
    s.listen(5)                 # Now wait for client connection.
    while True:
       c, addr = s.accept()     # Establish connection with client.
       print('Got connection from', addr)
       c.send('Thank you for connecting')
       c.close()                # Close the connection
            
run()
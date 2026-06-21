"""
Production WSGI server for Flask app using Waitress
Run: python wsgi.py
"""
from app import app
from waitress import serve

if __name__ == '__main__':
    print('✓ Starting Waitress WSGI server on 0.0.0.0:5000')
    print('✓ Production-ready with connection pooling')
    print('✓ Multi-threaded: 10 workers')
    print()
    
    # Production settings
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=10,  # Number of worker threads
        connection_limit=256,  # Max concurrent connections
        recv_bytes=8192,  # Socket receive buffer
        send_bytes=18000,  # Socket send buffer
        _quiet=False
    )

from flask import Flask, request
import json, datetime

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "remote_addr": request.remote_addr,
        "headers": dict(request.headers),
    }
    print(f"\n[{info['timestamp']}] Source IP: {info['remote_addr']}")
    for h in ['X-Forwarded-For','X-Real-IP','Via','Forwarded']:
        if request.headers.get(h):
            print(f"  {h}: {request.headers.get(h)}")
    return json.dumps(info, indent=2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

# Start (or restart) the OpenAI-compatible server inside the persistent kernel.
exec(open('/content/runs/load.py').read(),globals())
import threading,os,socketserver
from bonsai_tpu.engine import Engine
from bonsai_tpu.server import serve
NCTX=int(os.environ.get('BONSAI_NCTX','65536'))
E=Engine(None,n_ctx=NCTX,kv_dtype='bfloat16',preloaded=(G,CFG,PARAMS),block=int(os.environ.get('BONSAI_BLOCK','8')))
print('warmup s',round(E.warmup(),1),flush=True)
socketserver.TCPServer.allow_reuse_address=True
HTTPD=serve(E,'127.0.0.1',int(os.environ.get('BONSAI_PORT','8000')),default_thinking=os.environ.get('BONSAI_THINK','0')=='1')
threading.Thread(target=HTTPD.serve_forever,daemon=True).start()
print('SERVING on 127.0.0.1:%s n_ctx=%d'%(os.environ.get('BONSAI_PORT','8000'),NCTX),E.memory(),flush=True)

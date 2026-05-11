#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path

def post(url, body, timeout=300):
    data=json.dumps(body).encode(); req=urllib.request.Request(url,data=data,method='POST'); req.add_header('Content-Type','application/json')
    t0=time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw=r.read(); elapsed=time.perf_counter()-t0
    return json.loads(raw), elapsed

def extract(resp):
    msg=(resp.get('choices') or [{}])[0].get('message') or {}
    return msg.get('content') or '', msg.get('reasoning_content') or '', resp.get('usage') or {}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('model'); ap.add_argument('--out', required=True); ap.add_argument('--turns',type=int,default=3); ap.add_argument('--max-tokens',type=int,default=128)
    a=ap.parse_args(); rows=[]
    messages=[{'role':'user','content':'Write a concise but complete numbered list of practical ways local LLM inference can be made faster. Include exactly eight numbered points.'}]
    for i in range(a.turns):
        body={'model':a.model,'messages':messages,'max_tokens':a.max_tokens,'temperature':0.0,'stream':False,'enable_thinking':False}
        resp,elapsed=post(a.base.rstrip('/')+'/v1/chat/completions', body)
        content,reasoning,usage=extract(resp)
        ct=usage.get('completion_tokens') or 0
        rows.append({'turn':i+1,'elapsed_sec':elapsed,'completion_tokens':ct,'rough_tps':ct/elapsed if elapsed>0 else None,'content_chars':len(content),'reasoning_chars':len(reasoning),'head':content[:240],'tail':content[-240:],'usage':usage})
        messages.append({'role':'assistant','content':content or 'ok'})
        messages.append({'role':'user','content':'Continue the same list with more concrete implementation details, keeping the same numbering style.'})
    out={'base':a.base,'model':a.model,'turns':rows,'mean_rough_tps':sum(r['rough_tps'] or 0 for r in rows)/len(rows)}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
if __name__=='__main__': main()

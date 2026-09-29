# ADR 002: SSE token streaming over plain request/response

## Status
Accepted.

## Context
For a 24-token reply, this repo's own measurements (see README) show
~1.5-2s of total generation time on the CPU baseline. A plain
request/response client sees nothing for that entire window, then the full
reply appears at once. A streaming client sees the first token much sooner
(`time_to_first_token_s` in the server's own `usage` payload) and then a
steady trickle of tokens.

Total generation time is unchanged either way -- streaming does not make the
model faster. What changes is *perceived* latency: humans judge
responsiveness heavily by time-to-first-byte, not total completion time. A
chat UI that starts rendering text within a few hundred milliseconds feels
fast even if the full answer takes several seconds, while the same total
time with nothing shown until the end feels slow and can make users think
the request failed.

## Decision
`serving/server.py`'s `/v1/chat/completions` streams via Server-Sent Events
by default (`stream: true`), using `transformers.TextIteratorStreamer` to
pull tokens out of a background generation thread as they're produced. A
`stream: false` path also exists, for callers (like `bench/load_test.py`)
that want one wall-clock number per request instead of a token stream --
those are two different, both-legitimate things to measure, and conflating
them would make neither number mean anything precise.

We chose SSE over WebSockets for this specific endpoint because the traffic
is unidirectional (server produces tokens, client doesn't need to send
anything mid-stream) and HTTP-native: it works through ordinary reverse
proxies and load balancers without special upgrade handling, and every HTTP
client library already knows how to consume it.

## Consequences
- The server's own timing (`time_to_first_token_s`, `total_time_s`) is
  reported per-request in the final SSE chunk, which is what makes the
  README's latency numbers self-documenting rather than externally
  re-measured after the fact.
- A reverse proxy or load balancer in front of this service must not buffer
  the response body (e.g. disable `proxy_buffering` on nginx) or the
  streaming benefit is silently lost -- this is called out explicitly in
  docs/PRODUCTION.md.

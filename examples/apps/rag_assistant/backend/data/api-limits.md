# Lumen API limits

## Rate limits

The HTTP collector accepts up to 500 requests per second per workspace. Requests over the
limit return `429 Too Many Requests` with a `Retry-After` header in seconds.

## Batch size

Send up to 1,000 events per request, with a maximum body size of 1 MB. Larger requests fail
with `413 Payload Too Large`.

## Authentication

Every request needs an `x-lumen-key` header. Keys are created under **Settings > API keys**
and can be revoked at any time. A revoked key returns `401 Unauthorized` within one minute.

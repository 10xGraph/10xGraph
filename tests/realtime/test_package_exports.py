"""The realtime package must re-export the Phase 1 public surface (SDK-first usage)."""

import tenxgraph.core.realtime as rt




def test_queue_constructs_from_top_level_import():
    q = rt.LiveInputQueue()
    q.send_text("hi")
    item = q.get_nowait()
    assert isinstance(item, rt.LiveInput)
    assert item.kind == "text"

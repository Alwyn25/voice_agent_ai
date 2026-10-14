"""Knowledge lookup as an explicit tool.

Retrieval also runs automatically when the customer's words look factual, but the
model sometimes needs to look something up on its own initiative — for example after
the customer names a product mid-conversation.
"""
from app.engine.rag.retrieval import retrieve
from app.engine.tools.types import ToolContext


async def knowledge_lookup(ctx: ToolContext, params: dict) -> dict:
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    # Honour the session's simulated date, so a rewound clock applies to the model's
    # own lookups too — not just the automatic retrieval in the planner.
    result = await retrieve(ctx.db, tenant_id=ctx.tenant_id, query=query,
                            top_k=int(params.get("limit", 4)),
                            doc_type=params.get("doc_type"),
                            on_date=(ctx.extra or {}).get("as_of_date"),
                            # Scope the model's own lookup to the resolved project, so it can
                            # never surface another project's document than the one the call is
                            # about. None (tenant-wide) keeps the pre-multi-project behaviour.
                            project_id=ctx.project_id)
    if not result.chunks:
        # THE FALLBACK, REWRITTEN (Voice Agent Feedback #3–#6). The old guidance here told
        # the model to "offer a callback" on every miss, and it obeyed to the letter:
        # "daycare?", "dog parks?", "generator?", "bricks?" all got "I'll find out and
        # arrange a callback" — the worst possible answer from a sales agent, and in each
        # case the fact was actually in the brochure the search simply failed to surface.
        #
        # The honest fallback is: say plainly you do not have that specific detail, do NOT
        # invent one, and do NOT reflexively promise a callback. A callback is a real
        # commitment the sales team must honour; it belongs to a customer who ASKED to be
        # called, not to a question the search fumbled. Keep the caller moving on what the
        # agent can actually do now.
        return {"found": False, "query": query,
                "passages": [],
                "guidance": ("The knowledge base returned nothing for this specific query. "
                             "Do NOT answer from memory or guess a value. Tell the customer "
                             "plainly you don't have that exact detail to hand. Do NOT offer "
                             "a callback as a reflex — only schedule one if they explicitly "
                             "ask to be called back. Prefer a next step you can act on now: "
                             "answer any nearby part of their question you DO have, or invite "
                             "them to see it on a site visit. Try a different phrasing of the "
                             "lookup first if their wording may differ from the documents'.")}
    return {"found": True, "query": query,
            "passages": [c.as_context() for c in result.chunks]}

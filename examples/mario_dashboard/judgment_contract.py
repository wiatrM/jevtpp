"""Bind bounded model slots to the exact candidate snapshot that was scored."""
import math


def bind_plan_judgment(result, request):
    candidates = request.get('plan_candidates', [])
    scores = result.get('slot_scores')
    if not isinstance(scores, list) or len(scores) != 7 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError('Invalid model plan distribution')
    if abs(sum(scores) - 1) > .02:
        raise ValueError('Model plan distribution does not sum to one')
    if len(candidates) > 6 or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Invalid or duplicate plan candidate IDs')
    for i, candidate in enumerate(candidates):
        if candidate.get('slot') != i or not isinstance(candidate['id'], str) or not candidate['id']:
            raise ValueError('Plan candidate slots must be contiguous and named')
    slot = result.get('selected_slot')
    if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot <= 6:
        raise ValueError('Invalid selected plan slot')
    # A plan preference (including defer) must never carry legacy goal authority
    # into the separate goal gate. Metadata remains available for diagnostics.
    result={key:value for key,value in result.items() if key not in {
        'active_goal','model_goal','raw_model_goal','goal_scores','goal_probabilities'}}
    return {**result, 'judgment_kind': 'plan', 'plan_context': request['context_id'],
            'plan_facts': request.get('context_facts', {}),
            'model_plan': candidates[slot]['id'] if slot < len(candidates) else '',
            'plan_scores': {c['id']: scores[i] for i, c in enumerate(candidates)},
            'candidate_labels': {c['id']: f"#{i} " + c.get('skill', c['id']) + ': ' + c['action'] for i,c in enumerate(candidates)},
            'defer_score': scores[6], 'absent_slot_selected': slot < 6 and slot >= len(candidates),
            'score_semantics': 'Original model preferences, not outcome probabilities; absent slots are not renormalized'}

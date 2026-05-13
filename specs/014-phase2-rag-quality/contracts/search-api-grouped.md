# Search API Grouping Contract

When calling the search orchestrator or via the Web API with `grouped=True`:

## Request Parameters
- `query` (string): The search query.
- `grouped` (boolean): Default `False`. Set to `True` to enable grouping.
- `top_k_children` (integer): Default `3`. The maximum number of results to return per parent group.

## Response Format

The response remains a **flat JSON array** of search result objects, to avoid breaking existing LLM prompt context builders.

However, the contents of the array are constrained as follows:
1. **Grouping Identity**: Items are grouped by their logical parent (e.g., the same source document, the same project, or the same base entity).
2. **Child Limit**: No more than `top_k_children` items sharing the same parent identity will be present in the returned array.
3. **Ordering**: The groups themselves are ordered by the highest-scoring item within the group. The items within a group are ordered by their individual scores descending.

## Example

If `top_k_children=2`, and there are 5 highly relevant paragraphs from `Doc A` and 1 highly relevant paragraph from `Doc B`:

**Without Grouping**:
1. Doc A - Para 1 (Score: 0.99)
2. Doc A - Para 2 (Score: 0.98)
3. Doc A - Para 3 (Score: 0.95)
4. Doc B - Para 1 (Score: 0.90)
5. Doc A - Para 4 (Score: 0.88)

**With Grouping**:
1. Doc A - Para 1 (Score: 0.99)
2. Doc A - Para 2 (Score: 0.98)
   *(Doc A paras 3, 4, 5 are pruned)*
3. Doc B - Para 1 (Score: 0.90)
"""Diagnostic script for ContentBasedRecommender

Prints embedding shapes and sample recommendations for inspection.
"""
import sys
import random
sys.path.append(r'd:\StudyDocument\Recomendation System\Project')
import food_recomendation_system.app as app


def check(n_samples=5):
    app.ensure_loaded()
    rec = app._rec
    items = app._items_df
    try:
        emb = rec.embeddings
    except Exception:
        emb = None
    print('rec.embeddings type:', type(emb))
    try:
        if hasattr(emb, 'shape'):
            ln = emb.shape[0]
        else:
            ln = len(emb)
    except Exception:
        ln = None
    print('len embeddings:', ln)
    try:
        pos_map = app._pos_to_itemid
    except Exception:
        pos_map = None
    print('pos->itemid length:', 0 if pos_map is None else len(pos_map))

    titles = None
    if items is not None:
        titles = items['title'].astype(str).tolist()

    rng = random.Random(42)
    if ln is None:
        print('No embeddings available')
        return
    samples = rng.sample(range(ln), min(n_samples, ln))
    for s in samples:
        print('\n--- Sample item pos:', s)
        try:
            itemid = pos_map[s] if pos_map is not None else list(items.index)[s]
            print('itemID:', itemid)
            print('title:', items.loc[itemid]['title'])
        except Exception:
            pass
        try:
            idxs, scores = rec.recommend(s, top_k=10, titles=titles)
            for i, sc in zip(idxs.tolist(), scores.tolist()):
                try:
                    iid = pos_map[i] if pos_map is not None else list(items.index)[i]
                    print(f'  -> pos {i} (item {iid}) score={sc:.4f} title={items.loc[iid]["title"]}')
                except Exception:
                    print(f'  -> pos {i} score={sc:.4f} (mapping fail)')
        except Exception as e:
            print('recommend error:', e)


if __name__ == '__main__':
    check(5)

import sys
import random
import argparse
from collections import defaultdict

sys.path.append(r'd:\StudyDocument\Recomendation System\Project')
import food_recomendation_system.app as app
from food_recomendation_system.content_based import ContentBasedRecommender
from food_recomendation_system.collaborative import build_user_item_matrix, cosine_sim_matrix
import numpy as np


def pick_holdout_per_user(df, user_col='userID', seed=42, max_users=None):
    rng = random.Random(seed)
    users = df[user_col].value_counts()
    eligible = users[users >= 2].index.tolist()
    if max_users is not None and len(eligible) > max_users:
        eligible = rng.sample(eligible, max_users)
    holdouts = []
    for u in eligible:
        rows = df[df[user_col] == u]
        if 'timestamp' in rows.columns:
            chosen_idx = rows.sort_values('timestamp').index[-1]
        else:
            chosen_idx = rng.choice(rows.index.tolist())
        holdouts.append(chosen_idx)
    return holdouts


def hr_at_k(recommended_list, ground_truth, k):
    return int(ground_truth in recommended_list[:k])


def ndcg_at_k(recommended_list, ground_truth, k):
    for i, item in enumerate(recommended_list[:k], start=1):
        if item == ground_truth:
            return 1.0 / np.log2(i + 1)
    return 0.0


def minmax_normalize(d):
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    mn = vals.min()
    mx = vals.max()
    if mx - mn <= 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def evaluate(max_users=1000, k_list=(5, 10), alphas=(0.0, 0.25, 0.5, 0.75, 1.0), seed=42, cb_method='tfidf'):
    app.ensure_loaded()
    df = app._full_df
    items_df = app._items_df
    rec = app._rec

    if df is None or items_df is None or rec is None:
        print('Data or models not loaded. Run ensure_loaded in app first.')
        return

    print(f'Total interactions: {len(df)}, total items: {len(items_df)}')

    holdout_idxs = pick_holdout_per_user(df, seed=seed, max_users=max_users)
    print(f'Prepared {len(holdout_idxs)} holdouts for evaluation')

    train_df = df.drop(index=holdout_idxs)

    try:
        R, u2i, i2i = build_user_item_matrix(train_df, user_col='userID', item_col='itemID', rating_col='rating')
    except Exception as e:
        print('Failed to build user-item matrix:', e)
        return

    train_pos_to_itemid = [None] * len(i2i)
    for iid, pos in i2i.items():
        train_pos_to_itemid[pos] = iid

    try:
        item_sim = cosine_sim_matrix(R.T)
    except Exception as e:
        print('Failed to compute item-item similarity:', e)
        item_sim = None

    # rec fitted on canonical items; get mapping pos->itemid used by rec
    # Optionally re-fit the content recommender inside the evaluator with chosen method
    try:
        from food_recomendation_system.app import _pos_to_itemid as rec_pos_to_itemid
    except Exception:
        rec_pos_to_itemid = list(items_df.index)
    rec_itemid_to_pos = {itemid: idx for idx, itemid in enumerate(rec_pos_to_itemid)}

    # Always (re)fit the content recommender here with the requested cb_method
    print(f'Refitting ContentBasedRecommender with method="{cb_method}"')
    titles = [items_df.loc[itemid]['title'] if itemid in items_df.index else '' for itemid in rec_pos_to_itemid]
    descs = [items_df.loc[itemid]['description'] if itemid in items_df.index else '' for itemid in rec_pos_to_itemid]
    newrec = ContentBasedRecommender(method=cb_method)
    newrec.fit(titles, descs)
    rec = newrec

    if hasattr(rec, 'embeddings') and rec.embeddings is not None:
        emb = rec.embeddings
    else:
        print('Content recommender not available')
        return

    # popularity baseline
    pop_counts = train_df['itemID'].value_counts()
    popular_list = pop_counts.index.tolist()

    # accumulators
    results = {
        'pop': {k: [] for k in k_list},
        'cb': {k: [] for k in k_list},
        'cf': {k: [] for k in k_list},
    }
    hybrid_results = {a: {k: [] for k in k_list} for a in alphas}

    from sklearn.metrics.pairwise import cosine_similarity

    for idx in holdout_idxs:
        row = df.loc[idx]
        user = row['userID']
        test_item = row['itemID']

        user_train = train_df[train_df['userID'] == user]
        train_itemids = user_train['itemID'].unique().tolist()
        if len(train_itemids) == 0:
            continue

        # CF aggregation
        # CF aggregation (simple averaged weighted sum):
        # accumulate sum = sum sim(i,j) * r_{u,i} and count of contributions
        cf_sum = defaultdict(float)
        cf_count = defaultdict(int)
        if item_sim is not None:
            for tid in train_itemids:
                if tid in i2i:
                    pos = i2i[tid]
                    sims = item_sim[pos]
                    try:
                        r_ui = float(user_train[user_train['itemID'] == tid]['rating'].iloc[0])
                    except Exception:
                        r_ui = 1.0
                    for jpos, s in enumerate(sims):
                        iid = train_pos_to_itemid[jpos]
                        cf_sum[iid] += float(s) * r_ui
                        # count if similarity non-zero
                        if abs(float(s)) > 1e-9:
                            cf_count[iid] += 1
        # finalize cf scores as average contribution
        cf_scores = {iid: (cf_sum.get(iid, 0.0) / (cf_count.get(iid, 0) + 1e-9)) for iid in set(list(cf_sum.keys()) + list(cf_count.keys()))}

        # CB aggregation: sum similarity vectors from each train item pos
        cb_scores_arr = np.zeros(len(rec_pos_to_itemid))
        for tid in train_itemids:
            if tid in rec_itemid_to_pos:
                rpos = rec_itemid_to_pos[tid]
                if hasattr(emb, 'toarray'):
                    mat = emb
                    sims = cosine_similarity(mat[rpos], mat).flatten()
                else:
                    sims = cosine_similarity(emb[rpos:rpos+1], emb).flatten()
                cb_scores_arr += sims

        final_cb = {rec_pos_to_itemid[pos]: float(cb_scores_arr[pos]) for pos in range(len(rec_pos_to_itemid))}
        final_cf = dict(cf_scores)
        

        # remove user's train items
        for t in train_itemids:
            final_cb.pop(t, None)
            final_cf.pop(t, None)

        # normalize
        norm_cb = minmax_normalize(final_cb)
        norm_cf = minmax_normalize(final_cf)

        # ranked lists
        cb_list = [iid for iid, _ in sorted(norm_cb.items(), key=lambda x: -x[1])]
        cf_list = [iid for iid, _ in sorted(norm_cf.items(), key=lambda x: -x[1])]
        pop_list = [iid for iid in popular_list if iid not in train_itemids]

        # record baseline & methods
        for k in k_list:
            results['cb'][k].append(hr_at_k(cb_list, test_item, k))
            results['cf'][k].append(hr_at_k(cf_list, test_item, k))
            results['pop'][k].append(hr_at_k(pop_list, test_item, k))

        # hybrid sweep
        for a in alphas:
            hybrid_scores = defaultdict(float)
            # combine normalized scores
            keys = set(list(norm_cb.keys()) + list(norm_cf.keys()))
            for iid in keys:
                hybrid_scores[iid] = a * norm_cb.get(iid, 0.0) + (1 - a) * norm_cf.get(iid, 0.0)
            hybrid_list = [iid for iid, _ in sorted(hybrid_scores.items(), key=lambda x: -x[1])]
            for k in k_list:
                hybrid_results[a][k].append(hr_at_k(hybrid_list, test_item, k))

    # summarize
    print('\nEvaluation summary (Hit Rate / HR@K)')
    for method in ('pop', 'cb', 'cf'):
        print(f'-- {method.upper()} --')
        for k in k_list:
            arr = np.array(results[method][k], dtype=float)
            mean = float('nan') if arr.size == 0 else arr.mean()
            print(f'HR@{k}: {mean:.4f} (n={len(arr)})')

    print('\nHybrid sweep (HR@K)')
    for a in alphas:
        print(f'Alpha={a:.2f}')
        for k in k_list:
            arr = np.array(hybrid_results[a][k], dtype=float)
            mean = float('nan') if arr.size == 0 else arr.mean()
            print(f'  HR@{k}: {mean:.4f} (n={len(arr)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-users', type=int, default=1000, help='Max number of users to evaluate')
    parser.add_argument('--k', type=int, default=10, help='K for HR@K')
    parser.add_argument('--alphas', type=str, default='0.0,0.25,0.5,0.75,1.0', help='Comma separated alpha values for hybrid')
    parser.add_argument('--cb-method', type=str, default='tfidf', help='Content-based method: tfidf or sbert')
    parser.add_argument('--out', type=str, default=None, help='Optional CSV output path')
    args = parser.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(','))
    evaluate(max_users=args.max_users, k_list=(5, args.k), alphas=alphas, cb_method=args.cb_method)
    

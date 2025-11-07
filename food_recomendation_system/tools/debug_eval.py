import sys
sys.path.append(r'D:\StudyDocument\Recomendation System\Project')
import food_recomendation_system.app as app
from food_recomendation_system.tools.evaluate import pick_holdout_per_user
from food_recomendation_system.collaborative import build_user_item_matrix
import numpy as np

app.ensure_loaded()
df = app._full_df
print('df shape', df.shape)
h = pick_holdout_per_user(df, max_users=10)
print('holdouts sample', h[:5])
idx = h[0]
print('first holdout idx', idx)
row = df.loc[idx]
print('row user,item', row['userID'], row['itemID'])
train_df = df.drop(index=h)
print('train_df shape', train_df.shape)
R,u2i,i2i = build_user_item_matrix(train_df, user_col='userID', item_col='itemID', rating_col='rating')
print('R shape', R.shape, 'len i2i', len(i2i))
user = row['userID']
user_train = train_df[train_df['userID']==user]
print('user_train len', len(user_train))
print('train_itemids sample', user_train['itemID'].unique()[:10])
from food_recomendation_system.app import _pos_to_itemid
print('pos_to_itemid len', 0 if _pos_to_itemid is None else len(_pos_to_itemid))
from food_recomendation_system.tools.evaluate import minmax_normalize
from sklearn.metrics.pairwise import cosine_similarity

# replicate one evaluation iteration
idx = h[0]
row = df.loc[idx]
user = row['userID']
test_item = row['itemID']
user_train = train_df[train_df['userID'] == user]
train_itemids = user_train['itemID'].unique().tolist()
print('train_itemids count', len(train_itemids))

# compute item_sim
from food_recomendation_system.collaborative import cosine_sim_matrix
item_sim = cosine_sim_matrix(R.T)

# CF aggregation as in evaluate
cf_num = {}
cf_den = {}
train_pos_to_itemid = [None] * len(i2i)
for iid,pos in i2i.items():
	train_pos_to_itemid[pos] = iid
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
			cf_num[iid] = cf_num.get(iid, 0.0) + float(s) * r_ui
			cf_den[iid] = cf_den.get(iid, 0.0) + abs(float(s))

final_cf = {iid: (cf_num.get(iid, 0.0) / (cf_den.get(iid, 1e-9) + 1e-9)) for iid in set(list(cf_num.keys()) + list(cf_den.keys()))}
print('final_cf sample size', len(final_cf))

# CB aggregation
rec = app._rec
try:
	rec_emb = rec.embeddings
except Exception:
	rec_emb = None
if rec_emb is not None:
	rec_pos_to_itemid = _pos_to_itemid
	rec_itemid_to_pos = {itemid: idx for idx, itemid in enumerate(rec_pos_to_itemid)}
	cb_scores_arr = np.zeros(len(rec_pos_to_itemid))
	for tid in train_itemids:
		if tid in rec_itemid_to_pos:
			rpos = rec_itemid_to_pos[tid]
			if hasattr(rec_emb, 'toarray'):
				mat = rec_emb
				sims = cosine_similarity(mat[rpos], mat).flatten()
			else:
				sims = cosine_similarity(rec_emb[rpos:rpos+1], rec_emb).flatten()
			cb_scores_arr += sims
	final_cb = {rec_pos_to_itemid[pos]: float(cb_scores_arr[pos]) for pos in range(len(rec_pos_to_itemid))}
	print('final_cb sample size', len(final_cb))
	# compute normalized maps
	from food_recomendation_system.tools.evaluate import minmax_normalize
	norm_cb = minmax_normalize(final_cb)
	norm_cf = minmax_normalize(final_cf)
	print('norm_cb size', len(norm_cb))
	if norm_cb:
		vals = list(norm_cb.values())
		print('norm_cb min,max', min(vals), max(vals))
	print('norm_cf size', len(norm_cf))
	if norm_cf:
		vals = list(norm_cf.values())
		print('norm_cf min,max', min(vals), max(vals))
else:
	print('rec embeddings missing')

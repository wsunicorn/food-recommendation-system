from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from .data_loader import load_data
from .content_based import ContentBasedRecommender
from .collaborative import build_user_item_matrix, cosine_sim_matrix, predict_user_based
from .hybrid import weighted_hybrid
import os
import numpy as np

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.secret_key = 'food_recommendation_secret_key_2024'  # For session management

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'Dataset_for_print.csv')

_full_df = None
_items_df = None
_rec = None
# collaborative artifacts
_user_item_mat = None
_itemid_to_pos = None
_pos_to_itemid = None
_item_sim = None
_popular_list = None
_popular_counts = None
# user authentication
_valid_users = set()  # Set of valid userIDs from dataset
_user_to_index = {}  # Map userID to row index in user-item matrix
# performance cache
_user_sim_cache = None  # Cached user similarity matrix
_last_cache_time = None  # Track cache freshness


def ensure_loaded():
    global _full_df, _items_df, _rec, _user_item_mat, _itemid_to_pos, _pos_to_itemid, _item_sim
    global _valid_users, _user_to_index, _user_sim_cache, _last_cache_time, _popular_list, _popular_counts
    if _full_df is None:
        # load full interaction / recipe data
        _full_df = load_data(DATA_PATH)

        # build canonical items table keyed by itemID (one row per item)
        if 'itemID' in _full_df.columns:
            # keep first occurrence per itemID as canonical
            _items_df = _full_df.drop_duplicates(subset='itemID').set_index('itemID')
        else:
            # fallback: treat each row as unique and use its integer index as itemID
            _full_df = _full_df.reset_index().rename(columns={'index': 'itemID'})
            _items_df = _full_df.set_index('itemID')

        # build user-item matrix from full interactions if possible
        if {'userID', 'itemID', 'rating'}.issubset(_full_df.columns):
            R, u2i, i2i = build_user_item_matrix(_full_df, user_col='userID', item_col='itemID', rating_col='rating')
            _user_item_mat = R
            # i2i maps itemID -> position in R columns
            _itemid_to_pos = i2i
            # build pos->itemID list in order of columns
            pos_to_itemid = [None] * len(i2i)
            for itemid, pos in i2i.items():
                pos_to_itemid[pos] = itemid
            _pos_to_itemid = pos_to_itemid
            # compute item-item similarity from R (columns => items)
            try:
                _item_sim = cosine_sim_matrix(R.T)
            except Exception:
                _item_sim = None
            # Build user authentication data
            _valid_users = set(_full_df['userID'].unique())
            _user_to_index = u2i  # Map userID to row index in matrix
            # Pre-compute and cache user similarity matrix for performance
            print('Pre-computing user similarity matrix...')
            _user_sim_cache = cosine_sim_matrix(_user_item_mat)
            print(f'User similarity matrix cached: {_user_sim_cache.shape}')
        else:
            _user_item_mat = None
            _itemid_to_pos = None
            _pos_to_itemid = None
            _item_sim = None
            _popular_list = None

        # fit content-based recommender on canonical items (ordered by _pos_to_itemid if available)
        if _pos_to_itemid is not None:
            titles = [_items_df.loc[itemid]['title'] if itemid in _items_df.index else '' for itemid in _pos_to_itemid]
            descs = [_items_df.loc[itemid]['description'] if itemid in _items_df.index else '' for itemid in _pos_to_itemid]
        else:
            titles = _items_df['title'].astype(str).tolist()
            descs = _items_df['description'].astype(str).tolist()

        _rec = ContentBasedRecommender(method='tfidf')
        _rec.fit(titles, descs)

        # precompute popularity list from train interactions if available
        if 'itemID' in _full_df.columns:
            try:
                pop_counts = _full_df['itemID'].value_counts()
                _popular_list = pop_counts.index.tolist()
                # keep counts dictionary for scoring
                _popular_counts = pop_counts.to_dict()
            except Exception:
                _popular_list = None
                _popular_counts = None
        else:
            _popular_list = None
            _popular_counts = None


# reusable dedupe helper for items list (list of (idx, row))
def dedupe_items(items):
    seen = set()
    out = []
    for idx, row in items:
        try:
            t = str(row.get('title', '')).strip().lower()
        except Exception:
            t = ''
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append((idx, row))
    return out


@app.route('/login', methods=['GET', 'POST'])
def login():
    ensure_loaded()
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Check if username exists in dataset and password is correct
        try:
            user_id = int(username)
            if user_id in _valid_users and password == '9999':
                session['user_id'] = user_id
                session['username'] = str(user_id)
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error='Invalid username or password')
        except ValueError:
            return render_template('login.html', error='Username must be a valid user ID number')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    ensure_loaded()
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # filters and options
    selected_category = request.args.get('category', '')
    sort = request.args.get('sort', 'default')
    diversify = request.args.get('diversify', '') == '1'

    # operate on canonical items table
    df = _items_df
    if selected_category:
        df = df[df['category'] == selected_category]

    # helper: dedupe by normalized title
    def dedupe_items(items):
        seen = set()
        out = []
        for idx, row in items:
            try:
                t = str(row.get('title', '')).strip().lower()
            except Exception:
                t = ''
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append((idx, row))
        return out
    if sort == 'rating' and 'rating' in df.columns:
        try:
            df = df.sort_values(by='rating', ascending=False)
        except Exception:
            pass

    # prepare categories list with counts for sidebar (distinct items)
    categories_counts = sorted(_items_df['category'].value_counts().items(), key=lambda x: (-x[1], x[0]))
    total_all = len(_items_df)

    # pagination params
    try:
        page = int(request.args.get('page', '1'))
    except Exception:
        page = 1
    try:
        per_page = int(request.args.get('per_page', '48'))
    except Exception:
        per_page = 48
    n = per_page
    if diversify:
        # sample evenly across categories to increase diversity
        groups = df.groupby('category')
        per_group = max(1, n // max(1, len(groups)))
        rows = []
        for cat, g in groups:
            rows.extend(g.head(per_group).itertuples(index=False, name=None))
        # if not enough, pad from top
        if len(rows) < n:
            rows.extend(df.head(n - len(rows)).itertuples(index=False, name=None))
        # convert rows to list of (itemID, row) using canonical items_df
        items = []
        for r in rows[:n]:
            try:
                title = r[df.columns.get_loc('title')]
                # find itemID in items_df by title (first match)
                matches = _items_df[_items_df['title'] == title]
                if len(matches):
                    itemid = matches.index[0]
                    items.append((int(itemid), _items_df.loc[itemid]))
            except Exception:
                continue
    else:
        # df already is canonical items_df (one row per item)
        items = []
        for i, row in df.head(n).iterrows():
            items.append((int(i), row))

    # deduplicate items by title while preserving order
    items = dedupe_items(items)

    # pagination slice
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]

    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': max(1, (total + per_page - 1) // per_page),
    }

    current_user = session.get('username', 'Guest')
    return render_template('index.html', items=page_items, categories=[c for c,_ in categories_counts], categories_counts=categories_counts, selected_category=selected_category, sort=sort, diversify=diversify, pagination=pagination, total_all=total_all, current_user=current_user)


@app.route('/category/<path:category_name>')
def category_page(category_name: str):
    ensure_loaded()
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # show all items for this category
    df = _items_df[_items_df['category'] == category_name]
    items = []
    for i, row in df.iterrows():
        items.append((int(i), row))

    # pagination for category page
    try:
        page = int(request.args.get('page', '1'))
    except Exception:
        page = 1
    try:
        per_page = int(request.args.get('per_page', '48'))
    except Exception:
        per_page = 48

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': max(1, (total + per_page - 1) // per_page),
    }

    # use canonical items dataframe for category counts
    categories_sorted = sorted(_items_df['category'].value_counts().items(), key=lambda x: (-x[1], x[0]))
    return render_template('index.html', items=page_items, categories=[c for c,_ in categories_sorted], categories_counts=categories_sorted, selected_category=category_name, sort='default', diversify=False, pagination=pagination)


@app.route('/search')
def search():
    ensure_loaded()
    if 'user_id' not in session:
        return redirect(url_for('login'))
    q = request.args.get('q', '').lower().strip()
    if not q:
        return redirect(url_for('index'))
    
    # Search in both title and description for better results
    title_mask = _items_df['title'].astype(str).str.lower().str.contains(q, na=False)
    desc_mask = _items_df['description'].astype(str).str.lower().str.contains(q, na=False)
    combined_mask = title_mask | desc_mask
    
    results = []
    for i, row in _items_df[combined_mask].iterrows():
        results.append((int(i), row))
    
    # Sort by relevance: title matches first, then description matches
    results.sort(key=lambda x: (q not in x[1]['title'].lower(), x[1]['title']))
    
    # Get category counts for sidebar
    categories_counts = sorted(_items_df['category'].value_counts().items(), key=lambda x: (-x[1], x[0]))
    total_all = len(_items_df)
    current_user = session.get('username', 'Guest')
    
    return render_template('index.html', items=results[:100], categories=[c for c,_ in categories_counts], 
                         categories_counts=categories_counts, selected_category='', 
                         sort='default', diversify=False, pagination=None, total_all=total_all,
                         current_user=current_user, search_query=q, search_results_count=len(results))


@app.route('/item/<int:item_id>')
def item_page(item_id: int):
    ensure_loaded()
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # item_id is a DataFrame index label (int). Use label membership instead of positional iloc.
    if item_id not in _items_df.index:
        return "Item not found", 404
    # get the canonical item row by itemID label
    row = _items_df.loc[item_id]
    top_k = int(request.args.get('k', 16))  # Show more recommendations
    
    # Get current user
    current_user_id = session.get('user_id')
    
    # map label -> positional index for recommendation model
    # map itemID -> positional index used by _rec (if available)
    if _pos_to_itemid is not None and item_id in _itemid_to_pos:
        pos = int(_itemid_to_pos[item_id])
    else:
        # fallback: find positional index by searching _rec's fitted order
        pos = None
    # get cb scores
    if pos is not None:
        cb_idx, cb_scores = _rec.recommend(pos, top_k=top_k, titles=None)
        # cb_idx are positional indices
    else:
        # fallback: compute cosine similarity between query text and items (use titles list)
        titles = _items_df['title'].astype(str).tolist()
        # try to find positional index by matching item_id in _items_df order
        try:
            pos = list(_items_df.index).index(item_id)
        except Exception:
            pos = 0
        cb_idx, cb_scores = _rec.recommend(pos, top_k=top_k, titles=None)
    recs = []
    
    # compute PERSONALIZED CF-based recommendations using user-based collaborative filtering
    cf_scores = None
    if _user_item_mat is not None and current_user_id in _user_to_index:
        user_idx = _user_to_index[current_user_id]
        # Use CACHED user similarity matrix for performance
        try:
            if _user_sim_cache is not None:
                user_sim = _user_sim_cache
            else:
                # Fallback: compute on-the-fly if cache not available
                user_sim = cosine_sim_matrix(_user_item_mat)
            
            # Predict ratings for all items for this user using user-based CF
            # Use k=30 for better coverage
            cf_predictions = predict_user_based(_user_item_mat, user_idx, user_sim, k=30)
            # cf_predictions is an array of predicted ratings for all items (one per column in R)
            cf_scores = cf_predictions
        except Exception as e:
            # Fallback: use item-item similarity
            if _item_sim is not None and item_id in _itemid_to_pos:
                ipos = _itemid_to_pos[item_id]
                cf_scores = _item_sim[ipos]
            else:
                cf_scores = None
    elif _item_sim is not None and item_id in _itemid_to_pos:
        # Fallback: use item-item similarity
        ipos = _itemid_to_pos[item_id]
        cf_scores = _item_sim[ipos]

    # combine CB and CF into hybrid ranking
    candidates_set = set()
    TOP_N = top_k * 20  # Get more candidates than needed
    
    # from CB - content similarity
    try:
        for p in cb_idx.tolist()[:TOP_N]:
            candidates_set.add(int(p))
    except Exception:
        pass
    
    # from CF - personalized predictions
    if cf_scores is not None:
        cf_order = np.argsort(-cf_scores)
        for p in cf_order[:TOP_N]:
            candidates_set.add(int(p))

    candidates = list(candidates_set)
    
    # Filter out the query item itself
    if pos is not None and pos in candidates:
        candidates.remove(pos)

    # now compute hybrid scores for candidates
    # build score maps for candidates
    cb_map = {}
    try:
        for i, p in enumerate(cb_idx.tolist()):
            if p in candidates:
                cb_map[int(p)] = float(cb_scores[i])
    except Exception:
        pass
    cf_map = {}
    if cf_scores is not None:
        for p in candidates:
            try:
                cf_map[int(p)] = float(cf_scores[int(p)])
            except Exception:
                cf_map[int(p)] = 0.0

    # min-max normalize over candidate set
    def _minmax_map(m):
        if not m:
            return {k: 0.0 for k in candidates}
        vals = np.array(list(m.values()), dtype=float)
        mn = vals.min()
        mx = vals.max()
        if mx - mn <= 1e-9:
            return {k: 1.0 for k in m}
        return {k: (v - mn) / (mx - mn) for k, v in m.items()}

    norm_cb = _minmax_map(cb_map)
    norm_cf = _minmax_map(cf_map)

    # Hybrid scoring (70% content, 30% collaborative for better item diversity)
    alpha = 0.7
    hybrid_scores = []
    
    for p in candidates:
        cb_s = norm_cb.get(p, 0.0)
        cf_s = norm_cf.get(p, 0.0)
        
        # Weighted combination: prioritize content similarity
        hybrid_score = alpha * cb_s + (1.0 - alpha) * cf_s
        
        hybrid_scores.append((p, hybrid_score))

    # Sort by score
    hybrid_scores.sort(key=lambda x: -x[1])

    # Build recommendations (simple, no complex re-ranking)
    for p, s in hybrid_scores[:top_k]:
        try:
            label = _pos_to_itemid[p] if _pos_to_itemid is not None else list(_items_df.index)[p]
            rrow = _items_df.loc[label]
            recs.append({
                'item_id': int(label), 
                'score': float(s), 
                'title': rrow.get('title', ''), 
                'image': rrow.get('link_image_food', ''),
                'category': rrow.get('category', ''),
                'rating': rrow.get('rating', '')
            })
        except Exception:
            continue
    item = {
        'title': row.get('title', ''),
        'description': row.get('description', ''),
        'rating': row.get('rating', ''),
        'link_food': row.get('link_food', ''),
        'link_image_food': row.get('link_image_food', ''),
        'category': row.get('category', '')
    }
    
    return render_template('item.html', item=item, recs=recs, current_user=session.get('username', 'Guest'))


@app.route('/recommend/content/<int:item_id>')
def recommend_content(item_id: int):
    ensure_loaded()
    if 'user_id' not in session:
        return jsonify({'error': 'not authenticated'}), 401
    top_k = int(request.args.get('k', 10))
    # item_id is a DataFrame label; map to positional index for model, then map results back
    if item_id not in _items_df.index:
        return jsonify({'error': 'item not found'}), 404
    pos = int(_items_df.index.get_loc(item_id))
    idx_pos, scores = _rec.recommend(pos, top_k=top_k)
    items = []
    for pos_i, s in zip(idx_pos.tolist(), scores.tolist()):
        try:
            label = _items_df.index[int(pos_i)]
            items.append({'item_id': int(label), 'score': float(s), 'title': _items_df.loc[label, 'title']})
        except Exception:
            continue
    return jsonify(items)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, jsonify, request, render_template
from .data_loader import load_data
from .content_based import ContentBasedRecommender
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'Dataset_for_print.csv')

_df = None
_rec = None


def ensure_loaded():
    global _df, _rec
    if _df is None:
        _df = load_data(DATA_PATH)
        _rec = ContentBasedRecommender(method='tfidf')
        _rec.fit(_df['title'].astype(str).tolist(), _df['description'].astype(str).tolist())


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


@app.route('/')
def index():
    ensure_loaded()
    # filters and options
    selected_category = request.args.get('category', '')
    sort = request.args.get('sort', 'default')
    diversify = request.args.get('diversify', '') == '1'

    df = _df
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

    # prepare categories list with counts for sidebar
    categories_counts = sorted(_df['category'].value_counts().items(), key=lambda x: (-x[1], x[0]))
    total_all = len(_df)

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
        # convert rows to list of (index, row)
        items = []
        # need to map rows back to their indices
        for r in rows[:n]:
            # r is a namedtuple without index; find index via title match (fallback)
            try:
                title = r[df.columns.get_loc('title')]
                idx = int(df[df['title'] == title].index[0])
                items.append((idx, _df.iloc[idx]))
            except Exception:
                continue
    else:
        # use drop_duplicates on title to enumerate distinct products across the filtered df
        uniq = df.drop_duplicates(subset='title')
        items = []
        for i in uniq.index:
            row = _df.loc[int(i)]
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

    return render_template('index.html', items=page_items, categories=[c for c,_ in categories_counts], categories_counts=categories_counts, selected_category=selected_category, sort=sort, diversify=diversify, pagination=pagination, total_all=total_all)


@app.route('/category/<string:category_name>')
def category_page(category_name: str):
    ensure_loaded()
    # show all items for this category
    df = _df[_df['category'] == category_name]
    # drop duplicates by title within the category
    uniq = df.drop_duplicates(subset='title')
    items = []
    for i in uniq.index:
        items.append((int(i), _df.loc[int(i)]))

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

    return render_template('index.html', items=page_items, categories=[c for c,_ in sorted(_df['category'].value_counts().items(), key=lambda x:(-x[1], x[0]))], categories_counts=sorted(_df['category'].value_counts().items(), key=lambda x:(-x[1], x[0])), selected_category=category_name, sort='default', diversify=False, pagination=pagination)


@app.route('/search')
def search():
    ensure_loaded()
    q = request.args.get('q', '').lower()
    if not q:
        return render_template('index.html', items=[])
    mask = _df['title'].astype(str).str.lower().str.contains(q)
    results = []
    for i, row in _df[mask].head(100).iterrows():
        results.append((int(i), row))
    return render_template('index.html', items=results)


@app.route('/item/<int:item_id>')
def item_page(item_id: int):
    ensure_loaded()
    # item_id is a DataFrame index label (int). Use label membership instead of positional iloc.
    if item_id not in _df.index:
        return "Item not found", 404
    # get the row by label
    row = _df.loc[item_id]
    top_k = int(request.args.get('k', 8))
    # map label -> positional index for recommendation model
    pos = int(_df.index.get_loc(item_id))
    titles = _df['title'].astype(str).tolist()
    idx_pos, scores = _rec.recommend(pos, top_k=top_k, titles=titles)
    recs = []
    # map returned positional indices back to DataFrame labels
    for pos_i, s in zip(idx_pos.tolist(), scores.tolist()):
        try:
            label = _df.index[int(pos_i)]
            rrow = _df.loc[label]
            recs.append({'item_id': int(label), 'score': float(s), 'title': rrow.get('title', ''), 'image': rrow.get('link_image_food', '')})
        except Exception:
            continue
    item = {
        'title': row.get('title', ''),
        'description': row.get('description', ''),
        'rating': row.get('rating', ''),
        'link_food': row.get('link_food', ''),
        'link_image_food': row.get('link_image_food', '')
    }
    return render_template('item.html', item=item, recs=recs)


@app.route('/recommend/content/<int:item_id>')
def recommend_content(item_id: int):
    ensure_loaded()
    top_k = int(request.args.get('k', 10))
    # item_id is a DataFrame label; map to positional index for model, then map results back
    if item_id not in _df.index:
        return jsonify({'error': 'item not found'}), 404
    pos = int(_df.index.get_loc(item_id))
    idx_pos, scores = _rec.recommend(pos, top_k=top_k)
    items = []
    for pos_i, s in zip(idx_pos.tolist(), scores.tolist()):
        try:
            label = _df.index[int(pos_i)]
            items.append({'item_id': int(label), 'score': float(s), 'title': _df.loc[label, 'title']})
        except Exception:
            continue
    return jsonify(items)


if __name__ == '__main__':
    app.run(debug=True)

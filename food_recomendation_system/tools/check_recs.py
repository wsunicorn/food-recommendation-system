import sys
sys.path.append(r'd:\StudyDocument\Recomendation System\Project')
import food_recomendation_system.app as app

app.ensure_loaded()

_df = app._df
_rec = app._rec

titles = _df['title'].astype(str).tolist()
idx, scores = _rec.recommend(0, top_k=10, titles=titles)
print('Returned indices:', idx)
print('Returned titles:')
for i, s in zip(idx, scores):
    print(i, _df.iloc[int(i)]['title'], f'{s:.3f}')
